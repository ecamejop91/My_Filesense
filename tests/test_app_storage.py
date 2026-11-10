"""Real storage permissions, path separation, and failure without shared fallbacks."""
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import app_storage as storage


class PrivateStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.paths = storage.StoragePaths(self.root / 'account-a' / 'FileSense')

    def test_storage_is_independent_of_launch_folder_and_environment_username(self):
        expected = storage.user_data_dir()
        with patch.dict(os.environ, LOCALAPPDATA=str(self.root / 'shared'), HOME=str(self.root / 'other')):
            self.assertEqual(storage.user_data_dir(), expected)
        self.assertNotEqual(expected.parent, Path(__file__).resolve().parents[1])

    def test_runtime_caches_and_temporary_files_are_in_private_account_storage(self):
        old_temp = tempfile.tempdir
        self.addCleanup(setattr, tempfile, 'tempdir', old_temp)
        old_mask = os.umask(0o077)
        os.umask(old_mask)
        self.addCleanup(os.umask, old_mask)
        with patch.dict(os.environ, HF_HOME=str(self.root / 'shared-cache')):
            storage.configure_runtime(self.paths)
            for name in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'SENTENCE_TRANSFORMERS_HOME',
                         'XDG_CACHE_HOME', 'TIKTOKEN_CACHE_DIR', 'TORCH_HOME', 'TEMP', 'TMP', 'TMPDIR'):
                self.assertTrue(Path(os.environ[name]).is_relative_to(self.paths.root))
            with tempfile.NamedTemporaryFile() as handle:
                self.assertTrue(Path(handle.name).is_relative_to(self.paths.temporary))

    def test_permission_failure_is_reported_without_creating_a_fallback(self):
        with patch.object(storage, 'protect_path', side_effect=storage.StorageSecurityError('denied')):
            with self.assertRaisesRegex(storage.StorageSecurityError, 'denied'):
                storage.prepare_storage(self.paths)
        self.assertFalse((self.root / 'vector_store_notebook').exists())

    def test_mac_root_uses_the_account_home_instead_of_the_home_environment(self):
        pwd = SimpleNamespace(getpwuid=lambda uid: SimpleNamespace(pw_dir='/Users/account-a'))
        with patch.object(storage.sys, 'platform', 'darwin'), \
                patch.object(storage.os, 'getuid', return_value=123, create=True), \
                patch.dict(sys.modules, pwd=pwd), patch.dict(os.environ, HOME='/Users/account-b'):
            self.assertEqual(storage.user_data_dir(), Path('/Users/account-a/Library/Application Support/FileSense'))

    def test_mac_removes_extended_acls_and_sets_owner_only_modes(self):
        path = self.root / 'private'
        info = SimpleNamespace(st_uid=123, st_mode=stat.S_IFDIR | 0o700)
        with patch.object(storage.sys, 'platform', 'darwin'), \
                patch.object(storage.os, 'getuid', return_value=123, create=True), \
                patch.object(Path, 'lstat', return_value=info), \
                patch.object(storage.os, 'chmod') as chmod, \
                patch.object(storage.subprocess, 'run') as command:
            storage._secure_posix(path, True)
        command.assert_called_once_with(['/bin/chmod', '-N', str(path)], check=True, capture_output=True)
        chmod.assert_called_once_with(path, 0o700, follow_symlinks=False)

    def test_another_posix_accounts_folder_is_not_taken_over(self):
        with patch.object(storage.os, 'getuid', return_value=123, create=True), \
                patch.object(Path, 'lstat', return_value=SimpleNamespace(st_uid=456)), \
                patch.object(storage.os, 'chmod') as chmod:
            with self.assertRaisesRegex(storage.StorageSecurityError, 'different account'):
                storage._secure_posix(self.root, True)
        chmod.assert_not_called()

    def test_link_or_junction_is_rejected_before_changing_its_target(self):
        path = self.root / 'link'
        fake = Mock(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        with patch.object(Path, 'lstat', return_value=fake), \
                patch.object(storage, '_secure_windows') as windows, \
                patch.object(storage, '_secure_posix') as posix:
            with self.assertRaisesRegex(storage.StorageSecurityError, 'junction'):
                storage.protect_path(path, directory=True)
        windows.assert_not_called()
        posix.assert_not_called()

    def test_parent_components_do_not_change_permissions_outside_managed_storage(self):
        with patch.object(storage, 'PATHS', self.paths), \
                patch.object(storage, 'private_directory') as secure, \
                patch.object(storage, 'protect_path') as protect:
            storage.prepare_managed_file(self.paths.root / 'library/../../outside.sqlite3')
        secure.assert_not_called()
        protect.assert_not_called()

    def test_different_account_roots_do_not_share_library(self):
        other = storage.StoragePaths(self.root / 'account-b' / 'FileSense')
        storage.prepare_storage(self.paths)
        storage.prepare_storage(other)
        from document_database import clear_database, load_index, save_index
        import numpy as np
        a = dict(doc_id='doc_00000', pdf_path=str(self.root / 'a.pdf'), method='none', char_count=0)
        b = dict(a, pdf_path=str(self.root / 'b.pdf'))
        save_index([a], [], np.empty((0, 3)), self.paths.library, 'all-MiniLM-L6-v2')
        save_index([b], [], np.empty((0, 3)), other.library, 'all-MiniLM-L6-v2')
        clear_database(self.paths.library)
        self.assertEqual(load_index(self.paths.library)[0], [])
        self.assertEqual(load_index(other.library)[0], [b])

    @unittest.skipUnless(sys.platform == 'win32', 'Windows DACL enforcement')
    def test_windows_permissions_exclude_other_ordinary_accounts(self):
        import win32security
        storage.prepare_storage(self.paths)
        file = self.paths.library / 'private.txt'
        file.write_text('private')
        storage.protect_path(file)
        forbidden = {str(win32security.CreateWellKnownSid(sid)) for sid in (
            win32security.WinWorldSid, win32security.WinAuthenticatedUserSid, win32security.WinBuiltinUsersSid)}
        for path in (self.paths.root, self.paths.library, file):
            descriptor = win32security.GetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT,
                                                           win32security.DACL_SECURITY_INFORMATION)
            acl = descriptor.GetSecurityDescriptorDacl()
            self.assertIsNotNone(acl)
            self.assertEqual(acl.GetAceCount(), 3)
            self.assertTrue(descriptor.GetSecurityDescriptorControl()[0] & win32security.SE_DACL_PROTECTED)
            for index in range(acl.GetAceCount()):
                self.assertNotIn(str(acl.GetAce(index)[2]), forbidden)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows account ownership')
    def test_another_windows_accounts_folder_is_not_taken_over(self):
        import win32security
        descriptor = Mock()
        descriptor.GetSecurityDescriptorOwner.return_value = win32security.CreateWellKnownSid(win32security.WinAnonymousSid)
        with patch.object(win32security, 'GetFileSecurity', return_value=descriptor), \
                patch.object(win32security, 'SetNamedSecurityInfo') as change:
            with self.assertRaisesRegex(storage.StorageSecurityError, 'different account'):
                storage._secure_windows(self.root, True)
        change.assert_not_called()

    @unittest.skipIf(sys.platform == 'win32', 'POSIX permissions')
    def test_posix_permissions_and_ownership(self):
        storage.prepare_storage(self.paths)
        file = self.paths.library / 'private.txt'
        file.write_text('private')
        storage.protect_path(file)
        self.assertEqual(stat.S_IMODE(self.paths.root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)
        self.assertEqual(file.stat().st_uid, os.getuid())
