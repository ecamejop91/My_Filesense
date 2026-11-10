"""Account-private runtime storage, independent of where FileSense is launched."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import sys
import subprocess
import tempfile
from uuid import UUID


class StorageSecurityError(RuntimeError):
    """Private storage could not be established; never fall back to a shared path."""


def user_data_dir():
    if sys.platform == 'win32':
        # Resolve the signed-in account's known folder, not an inherited or
        # user-supplied LOCALAPPDATA value or a directory beside the executable.
        guid = ctypes.create_string_buffer(UUID('F1B32785-6FBA-4FCF-9D55-7B8E7F157091').bytes_le, 16)
        pointer = ctypes.c_void_p()
        result = ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None,
                                                          ctypes.byref(pointer))
        if result != 0:
            raise StorageSecurityError('Windows could not locate your private application data folder.')
        try:
            base = Path(ctypes.wstring_at(pointer))
        finally:
            ctypes.windll.ole32.CoTaskMemFree(pointer)
        return base / 'FileSense'
    import pwd
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    return home / ('Library/Application Support' if sys.platform == 'darwin' else '.local/share') / 'FileSense'


@dataclass(frozen=True)
class StoragePaths:
    root: Path

    @property
    def library(self):
        return self.root / 'library'

    @property
    def cache(self):
        return self.root / 'cache'

    @property
    def temporary(self):
        return self.root / 'tmp'

    @property
    def logs(self):
        return self.root / 'logs'


PATHS = StoragePaths(user_data_dir())


def _secure_windows(path, directory):
    import ntsecuritycon
    import win32api
    import win32con
    import win32security

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    finally:
        token.Close()
    descriptor = win32security.GetFileSecurity(str(path), win32security.OWNER_SECURITY_INFORMATION)
    if descriptor.GetSecurityDescriptorOwner() != user:
        raise StorageSecurityError(f'Private storage is owned by a different account: {path}')
    allowed = (user, win32security.CreateWellKnownSid(win32security.WinLocalSystemSid),
               win32security.CreateWellKnownSid(win32security.WinBuiltinAdministratorsSid))
    acl = win32security.ACL()
    flags = win32con.OBJECT_INHERIT_ACE | win32con.CONTAINER_INHERIT_ACE if directory else 0
    for sid in allowed:
        acl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, flags, ntsecuritycon.FILE_ALL_ACCESS, sid)
    win32security.SetNamedSecurityInfo(
        str(path), win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None, None, acl, None)
    actual = win32security.GetNamedSecurityInfo(
        str(path), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION)
    dacl = actual.GetSecurityDescriptorDacl()
    if (dacl is None or dacl.GetAceCount() != len(allowed)
            or not actual.GetSecurityDescriptorControl()[0] & win32security.SE_DACL_PROTECTED):
        raise StorageSecurityError(f'Windows did not retain private permissions: {path}')
    for index in range(dacl.GetAceCount()):
        (kind, ace_flags), rights, sid = dacl.GetAce(index)
        if (kind != win32security.ACCESS_ALLOWED_ACE_TYPE or ace_flags != flags
                or sid not in allowed or rights != ntsecuritycon.FILE_ALL_ACCESS):
            raise StorageSecurityError(f'Windows could not verify private permissions: {path}')


def _secure_posix(path, directory):
    info = path.lstat()
    if info.st_uid != os.getuid():
        raise StorageSecurityError(f'Private storage is owned by a different account: {path}')
    if sys.platform == 'darwin':
        # macOS extended ACLs can grant access beyond POSIX mode bits.
        subprocess.run(['/bin/chmod', '-N', str(path)], check=True, capture_output=True)
    mode = 0o700 if directory else 0o600
    os.chmod(path, mode, follow_symlinks=False)
    if stat.S_IMODE(path.lstat().st_mode) != mode:
        raise StorageSecurityError(f'Could not verify private permissions: {path}')


def protect_path(path, *, directory=False):
    path = Path(os.path.abspath(path))
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise StorageSecurityError(f'Private storage cannot use a symbolic link or junction: {path}')
        if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
            raise StorageSecurityError(f'Unexpected private storage file type: {path}')
        (_secure_windows if sys.platform == 'win32' else _secure_posix)(path, directory)
    except StorageSecurityError:
        raise
    except Exception as exc:
        raise StorageSecurityError(f'Could not protect FileSense data at {path}: {exc}') from exc
    return path


def private_directory(path):
    path = Path(os.path.abspath(path))
    try:
        if not path.exists() and not path.is_symlink():
            if not path.parent.exists():
                private_directory(path.parent)
            path.mkdir(mode=0o700, exist_ok=True)
        return protect_path(path, directory=True)
    except StorageSecurityError:
        raise
    except OSError as exc:
        raise StorageSecurityError(f'Could not create private FileSense storage at {path}: {exc}') from exc


def prepare_storage(paths=PATHS):
    private_directory(paths.root)
    for folder in (paths.library, paths.temporary, paths.logs, paths.cache):
        private_directory(folder)
        # Repair older explicit file permissions as well as directory inheritance.
        # Public model caches can legitimately contain Hugging Face symlinks.
        if folder != paths.cache:
            for base, directories, files in os.walk(folder, followlinks=False):
                for name in directories:
                    protect_path(Path(base) / name, directory=True)
                for name in files:
                    protect_path(Path(base) / name)
    return paths


def configure_runtime(paths=PATHS):
    """Call before importing model/tokenizer libraries in a desktop launch."""
    prepare_storage(paths)
    if sys.platform != 'win32':
        os.umask(0o077)
    for variable, name in (('HF_HOME', 'huggingface'), ('TIKTOKEN_CACHE_DIR', 'tiktoken'),
                           ('TORCH_HOME', 'torch'),
                           ('HF_HUB_CACHE', 'huggingface/hub'),
                           ('HUGGINGFACE_HUB_CACHE', 'huggingface/hub'),
                           ('SENTENCE_TRANSFORMERS_HOME', 'sentence-transformers')):
        os.environ[variable] = str(private_directory(paths.cache / name).resolve())
    os.environ['XDG_CACHE_HOME'] = str(paths.cache.resolve())
    for variable in ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE'):
        os.environ.pop(variable, None)
    for variable in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[variable] = str(paths.temporary.resolve())
    tempfile.tempdir = str(paths.temporary.resolve())
    return paths


def prepare_managed_file(path):
    """Protect managed database paths, leaving explicit script/export paths alone."""
    path = Path(os.path.abspath(path))
    logical_root = Path(os.path.abspath(PATHS.root))
    physical_root = logical_root.resolve()
    if path.is_relative_to(logical_root) or path.is_relative_to(physical_root):
        private_directory(PATHS.root)
        # Check every component before any database connection can follow it.
        current = logical_root if path.is_relative_to(logical_root) else physical_root
        for part in path.parent.relative_to(current).parts:
            current = private_directory(current / part)
        if path.exists() or path.is_symlink():
            protect_path(path)
        for suffix in ('-wal', '-shm', '-journal'):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists() or sidecar.is_symlink():
                protect_path(sidecar)
