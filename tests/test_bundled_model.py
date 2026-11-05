"""Portable model selection and packaging, without downloads or real weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import bundle_model as bundle
import model_runtime as runtime


class BundledModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.folder = self.root / 'models' / 'all-MiniLM-L6-v2'
        patcher = patch.object(runtime, 'BUNDLED_MODEL_DIR', self.folder)
        patcher.start()
        self.addCleanup(patcher.stop)
        runtime._models.clear()

    def files(self, folder=None):
        folder = folder or self.folder
        for name in runtime.BUNDLED_MODEL_FILES:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture model bytes')
        (folder / 'README.md').write_text('license: apache-2.0\nFixture card', encoding='utf-8')
        return folder

    def manifest(self):
        manifest = dict(schema_version=1, model_id=bundle.MODEL_ID, revision=bundle.MODEL_REVISION,
                        embedding_dimension=384, files={
                            path.relative_to(self.folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in self.folder.rglob('*') if path.is_file()})
        (self.folder / 'bundle.json').write_text(json.dumps(manifest), encoding='utf-8')
        return manifest

    def test_bundled_model_preferred_and_aliases_share_instance(self):
        self.files()
        factory = Mock(return_value=object())
        first = runtime.get_model('all-MiniLM-L6-v2', factory)
        second = runtime.get_model(bundle.MODEL_ID, factory)
        self.assertIs(first, second)
        factory.assert_called_once_with(str(self.folder), local_files_only=True)

    def test_present_incomplete_bundle_fails_before_any_factory_or_download(self):
        self.folder.mkdir(parents=True)
        factory = Mock()
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            runtime.get_model(bundle.MODEL_ID, factory)
        factory.assert_not_called()

    def test_failed_bundled_load_never_falls_back_to_hub(self):
        self.files()
        factory = Mock(side_effect=OSError('damaged weights'))
        with self.assertRaisesRegex(OSError, 'damaged'):
            runtime.get_model(bundle.MODEL_ID, factory)
        factory.assert_called_once_with(str(self.folder), local_files_only=True)

    def test_missing_bundle_retains_cache_and_first_download_workflow(self):
        factory = Mock(side_effect=[OSError('no cached model'), object()])
        runtime.get_model(bundle.MODEL_ID, factory)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args_list[0].kwargs, {'local_files_only': True})
        self.assertEqual(factory.call_args_list[1].args, (bundle.MODEL_ID,))

    def test_custom_model_names_are_not_replaced_by_default_bundle(self):
        self.files()
        factory = Mock(return_value=object())
        runtime.get_model('my-custom-model', factory)
        factory.assert_called_once_with('my-custom-model', local_files_only=True)

    def test_verification_detects_modified_files(self):
        self.files()
        self.manifest()
        bundle.verify_bundle(self.folder)
        (self.folder / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            bundle.verify_bundle(self.folder)

    def test_verification_rejects_paths_outside_model_directory(self):
        self.files()
        manifest = self.manifest()
        manifest['files']['../../outside'] = 'bad'
        (self.folder / 'bundle.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Invalid file path'):
            bundle.verify_bundle(self.folder)

    def test_packaging_copies_bytes_preserves_card_and_is_idempotent(self):
        source = self.files(self.root / 'cached-snapshot')
        model = Mock()
        model.get_sentence_embedding_dimension.return_value = 384
        with patch('huggingface_hub.snapshot_download', return_value=source) as snapshot, \
                patch('sentence_transformers.SentenceTransformer', return_value=model) as factory:
            first = bundle.bundle_model(self.folder)
            second = bundle.bundle_model(self.folder)
        self.assertEqual(first, second)
        self.assertEqual(snapshot.call_count, 1)
        self.assertTrue(snapshot.call_args.kwargs['local_files_only'])
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(factory.call_args.kwargs['local_files_only'])
        self.assertEqual((self.folder / 'README.md').read_bytes(), (source / 'README.md').read_bytes())
        self.assertFalse((self.folder / 'model.safetensors').is_symlink())
        self.assertEqual(bundle.verify_bundle(self.folder)['embedding_dimension'], 384)


if __name__ == '__main__':
    unittest.main()
