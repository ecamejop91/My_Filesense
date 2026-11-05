"""Explicit account imports preserve sources and reject unsafe or nonempty targets."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from app_storage import StoragePaths
import document_database as database
import document_tag_store as tags
from library_import import import_library


class LibraryImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'old-app' / 'vector_store_notebook' / database.DATABASE_NAME
        self.target = self.root / 'account' / 'library' / database.DATABASE_NAME
        patcher = patch('library_import.PATHS', StoragePaths(self.root / 'account'))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.doc = dict(doc_id='doc_00000', pdf_path=str(self.root / 'invoice.pdf'), method='text', char_count=12)
        self.chunk = dict(self.doc, page=1, chunk_id=0, text='Invoice paid')
        database.save_index([self.doc], [self.chunk], np.array([[1., 0., 0.]]), self.source.parent, 'all-MiniLM-L6-v2')
        tags.save_tag_definition('Private category', 'invoice', self.source)

    def test_explicit_import_preserves_text_vectors_tags_and_original_database(self):
        before = self.source.read_bytes()
        result = import_library(self.source, self.target)
        self.assertEqual(result, dict(documents=1, chunks=1))
        docs, chunks, vectors = database.load_index(self.target.parent)
        self.assertEqual(docs, [self.doc])
        self.assertEqual(chunks[0]['text'], 'Invoice paid')
        np.testing.assert_array_equal(vectors, [[1., 0., 0.]])
        self.assertIn('Private category', tags.load_tag_definitions(self.target))
        self.assertEqual(self.source.read_bytes(), before)

    def test_new_account_does_not_automatically_import_an_old_project_library(self):
        before = self.source.read_bytes()
        with patch.object(database, 'PROJECT_DIR', self.source.parent.parent):
            self.assertEqual(database.load_index(self.target.parent)[0], [])
        self.assertEqual(self.source.read_bytes(), before)

    def test_nonempty_target_is_never_overwritten(self):
        import_library(self.source, self.target)
        before = self.target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'already contains data'):
            import_library(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), before)

    def test_relative_source_paths_are_anchored_to_old_application(self):
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute("UPDATE documents SET pdf_path='Files/invoice.pdf'")
        import_library(self.source, self.target)
        self.assertEqual(database.load_index(self.target.parent)[0][0]['pdf_path'],
                         str(self.source.parent.parent / 'Files/invoice.pdf'))

    def test_untrusted_triggers_are_rejected_and_destination_remains_empty(self):
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute('CREATE TRIGGER unexpected AFTER INSERT ON documents BEGIN DELETE FROM chunks; END')
        with self.assertRaisesRegex(ValueError, 'triggers'):
            import_library(self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_invalid_vectors_are_rejected_before_target_is_written(self):
        with closing(sqlite3.connect(self.source)) as connection, connection:
            connection.execute("UPDATE chunks SET embedding=x'0000'")
        with self.assertRaises(ValueError):
            import_library(self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_import_includes_committed_wal_changes(self):
        with closing(sqlite3.connect(self.source)) as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute("INSERT INTO tag_definitions VALUES ('WAL category', 'payment')")
            connection.commit()
            import_library(self.source, self.target)
        self.assertIn('WAL category', tags.load_tag_definitions(self.target))
