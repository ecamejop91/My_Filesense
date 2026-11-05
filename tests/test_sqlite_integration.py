import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import document_database as database
import document_search_engine as engine
import document_tag_store as tags


class SQLiteIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve() / 'store'
        self.db = self.folder / database.DATABASE_NAME
        self.docs = [dict(doc_id='doc_00000', pdf_path=str(Path(self.temp.name).resolve() / "O'Brien.pdf"), method='text', char_count=17)]
        self.chunks = [dict(doc_id='doc_00000', pdf_path=self.docs[0]['pdf_path'], page=1, chunk_id=0, text="Invoice 'paid'; ✓")]
        self.vectors = np.array([[1, 0, 0]], dtype=np.float32)

    def save(self, docs=None, chunks=None, vectors=None):
        database.save_index(self.docs if docs is None else docs, self.chunks if chunks is None else chunks,
                            self.vectors if vectors is None else vectors, self.folder, engine.MODEL_NAME)

    def legacy(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        for filename, records in [('docs.jsonl', self.docs), ('chunks.jsonl', self.chunks)]:
            (self.folder / filename).write_text(''.join(json.dumps(row) + '\n' for row in records))
        np.save(self.folder / 'vectors.npy', self.vectors)

    def test_round_trip_and_tag_coexistence(self):
        tags.save_tag_definition('Research', 'scientific paper', self.db)
        with closing(tags.connect_tag_db(self.db)) as connection:
            tags.save_file_tags(Path(self.docs[0]['pdf_path']), [{'tag': 'Research', 'score': .8}], 17, connection)
        self.save()
        docs, chunks, vectors = database.load_index(self.folder)
        self.assertEqual(docs, self.docs)
        self.assertEqual(chunks, self.chunks)
        np.testing.assert_array_equal(vectors, self.vectors)
        self.assertEqual(tags.load_tag_definitions(self.db)['Research'], 'scientific paper')
        self.assertEqual(len(tags.exact_tag_matches('research', self.db)), 1)
        self.assertFalse((self.folder / 'vectors.npy').exists())
        with closing(database.connect_database(self.db)) as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])

    def test_failed_save_preserves_previous_index(self):
        self.save()
        with self.assertRaises(sqlite3.IntegrityError):
            self.save(docs=[dict(self.docs[0], char_count=-1)])
        docs, _, vectors = database.load_index(self.folder)
        self.assertEqual(docs, self.docs)
        np.testing.assert_array_equal(vectors, self.vectors)
        with self.assertRaises(ValueError):
            self.save(vectors=np.ones((2, 3)))
        self.assertEqual(database.load_index(self.folder)[0], self.docs)

    def test_clear_removes_index_and_assignments_but_keeps_definitions_and_pdfs(self):
        pdf = Path(self.docs[0]['pdf_path'])
        pdf.write_bytes(b'original PDF contents')
        self.legacy()
        self.save()
        tags.save_tag_definition('Research', 'scientific paper', self.db)
        with closing(tags.connect_tag_db(self.db)) as connection:
            tags.save_file_tags(pdf, [{'tag': 'Research', 'score': .8}], 17, connection)
        database.clear_database(self.folder)
        docs, chunks, vectors = database.load_index(self.folder)
        self.assertEqual((docs, chunks), ([], []))
        self.assertEqual(len(vectors), 0)
        self.assertEqual(tags.load_tag_rows(self.db), [])
        self.assertEqual(tags.load_tag_definitions(self.db)['Research'], 'scientific paper')
        self.assertEqual(pdf.read_bytes(), b'original PDF contents')
        self.assertTrue((self.folder / 'docs.jsonl').exists())
        # Cleared metadata must prevent legacy backups from repopulating the index.
        self.assertEqual(database.load_index(self.folder)[0], [])
        self.save()
        self.assertEqual(database.load_index(self.folder)[0], self.docs)

    def test_clear_empty_database_does_not_import_legacy_backups(self):
        self.legacy()
        database.clear_database(self.folder)
        database.clear_database(self.folder)
        self.assertEqual(database.load_index(self.folder)[0], [])

    def test_clear_failure_rolls_back_index_deletion(self):
        self.save()
        with closing(tags.connect_tag_db(self.db)) as connection:
            tags.save_file_tags(Path(self.docs[0]['pdf_path']), [], 17, connection)
            connection.execute("CREATE TRIGGER prevent_clear BEFORE DELETE ON file_tags "
                               "BEGIN SELECT RAISE(ABORT, 'test failure'); END")
            connection.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            database.clear_database(self.folder)
        self.assertEqual(database.load_index(self.folder)[0], self.docs)
        self.assertEqual(len(tags.load_tag_rows(self.db)), 1)

    def test_rejects_orphan_chunk_and_invalid_vectors(self):
        for chunks, vectors in [([dict(self.chunks[0], doc_id='missing')], self.vectors),
                                (self.chunks, np.array([[float('nan'), 0, 0]]))]:
            with self.assertRaises(ValueError):
                self.save(chunks=chunks, vectors=vectors)

    def test_legacy_import_is_one_time_and_preserves_sources(self):
        self.legacy()
        paths = [self.folder / name for name in ('docs.jsonl', 'chunks.jsonl', 'vectors.npy')]
        before = [path.read_bytes() for path in paths]
        self.assertEqual(database.load_index(self.folder)[0], self.docs)
        self.assertEqual([path.read_bytes() for path in paths], before)
        self.save(docs=[], chunks=[], vectors=np.empty((0, 3)))
        docs, chunks, vectors = database.load_index(self.folder)
        self.assertEqual((docs, chunks, vectors.shape), ([], [], (0, 3)))

    def test_incomplete_legacy_import_can_be_retried(self):
        self.legacy()
        (self.folder / 'vectors.npy').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            database.load_index(self.folder)
        np.save(self.folder / 'vectors.npy', self.vectors)
        self.assertEqual(database.load_index(self.folder)[0], self.docs)

    def test_invalid_legacy_import_is_atomic_and_retryable(self):
        self.legacy()
        np.save(self.folder / 'vectors.npy', np.ones((2, 3)))
        with self.assertRaises(ValueError):
            database.load_index(self.folder)
        with closing(sqlite3.connect(self.db)) as connection, connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM documents').fetchone()[0], 0)
            self.assertEqual(connection.execute('SELECT count(*) FROM index_metadata').fetchone()[0], 0)
        np.save(self.folder / 'vectors.npy', self.vectors)
        self.assertEqual(database.load_index(self.folder)[0], self.docs)

    def test_legacy_relative_paths_are_anchored_to_app_folder(self):
        self.docs[0]['pdf_path'] = self.chunks[0]['pdf_path'] = 'Files/example.pdf'
        self.legacy()
        docs, chunks, _ = database.load_index(self.folder)
        expected = str(self.folder.parent / 'Files/example.pdf')
        self.assertEqual(docs[0]['pdf_path'], expected)
        self.assertEqual(chunks[0]['pdf_path'], expected)

    def test_model_mismatch_does_not_modify_index(self):
        self.save()
        with self.assertRaisesRegex(ValueError, 'original model'):
            database.load_index(self.folder, 'different-model')
        self.assertEqual(database.load_index(self.folder)[0], self.docs)

    def test_chunk_order_preserves_embedding_alignment(self):
        chunks = [dict(self.chunks[0], chunk_id=2, text='second'), dict(self.chunks[0], chunk_id=0)]
        vectors = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
        self.save(chunks=chunks, vectors=vectors)
        _, loaded, matrix = database.load_index(self.folder)
        self.assertEqual(loaded, chunks)
        np.testing.assert_array_equal(matrix, vectors)

    def test_empty_and_no_text_documents_round_trip(self):
        self.assertEqual(database.load_index(self.folder)[2].shape, (0, 0))
        self.save(docs=[dict(self.docs[0], method='encrypted', char_count=0)], chunks=[], vectors=np.empty((0, 3)))
        docs, chunks, vectors = database.load_index(self.folder)
        self.assertEqual(docs[0]['method'], 'encrypted')
        self.assertEqual((chunks, vectors.shape), ([], (0, 3)))

    def test_old_tag_schema_migrates_and_duplicate_names_stay_distinct(self):
        self.folder.mkdir()
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute('CREATE TABLE file_tags (file_name TEXT PRIMARY KEY, file_path TEXT NOT NULL, char_count INTEGER NOT NULL, tag_1 TEXT, score_1 REAL, tag_2 TEXT, score_2 REAL, tag_3 TEXT, score_3 REAL)')
            connection.execute('INSERT INTO file_tags VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                               ("O'Brien.pdf", self.docs[0]['pdf_path'], 17, 'Finance', .8, None, None, None, None))
        new_path = Path(self.temp.name).resolve() / 'another' / "O'Brien.pdf"
        with closing(tags.connect_tag_db(self.db)) as connection:
            tags.save_file_tags(new_path, [{'tag': 'Finance', 'score': .6}], 20, connection)
        self.assertEqual(len(tags.load_tag_rows(self.db)), 2)
        self.assertEqual(len(tags.exact_tag_matches('FINANCE', self.db)), 2)
        with closing(tags.connect_tag_db(self.db)) as connection:
            tags.save_file_tags(new_path, [{'tag': 'Legal', 'score': .9}], 21, connection)
        self.assertEqual(len(tags.load_tag_rows(self.db)), 2)
        self.assertEqual(len(tags.exact_tag_matches('Finance', self.db)), 1)

    def test_custom_tag_uses_parameters(self):
        name = "Owner's notes'; DROP TABLE documents; --"
        tags.save_tag_definition(name, 'private notes', self.db)
        with closing(tags.connect_tag_db(self.db)) as connection:
            tags.save_file_tags(Path(self.docs[0]['pdf_path']), [{'tag': name, 'score': .9}], 17, connection)
        self.assertEqual(len(tags.exact_tag_matches(name, self.db)), 1)
        self.save()
        self.assertEqual(database.load_index(self.folder)[0], self.docs)

    def test_update_reopen_and_search(self):
        class Model:
            def __init__(self, *args, **kwargs):
                pass
            def get_sentence_embedding_dimension(self):
                return 3
            def encode(self, texts, **kwargs):
                return np.array([[1, 0, 0] if 'invoice' in text else [0, 1, 0] for text in texts], dtype=np.float32)

        first = Path(self.temp.name).resolve() / 'one' / 'report.pdf'
        second = Path(self.temp.name).resolve() / 'two' / 'report.pdf'
        for path in (first, second):
            path.parent.mkdir()
            path.write_bytes(b'PDF fixture')
        with patch.object(engine, 'SentenceTransformer', Model), \
             patch.object(engine.tiktoken, 'get_encoding', return_value=object()), \
             patch.object(engine, 'extract_page_texts', return_value=('text', [(1, 'invoice')])), \
             patch.object(engine, 'chunk_page_text', return_value=[{'page': 1, 'text': 'invoice'}]):
            store = engine.update_vector_store([first, second, first], self.folder)
            self.assertEqual(len(store['added_doc_records']), 2)
            self.assertEqual(store['skipped_paths'], [(first, 'already indexed')])
            reopened = engine.load_vector_store(self.folder)
            result = engine.search_index('invoice', reopened['model'], reopened['chunk_records'], reopened['chunk_vectors'], reopened['docs_by_id'])
            self.assertEqual({Path(row['path']) for row in result}, {first, second})
            self.assertEqual(len(result), 2)
            self.assertTrue(all(row['score'] == 1 for row in result))
            again = engine.update_vector_store([second], self.folder)
            self.assertEqual(again['added_doc_records'], [])
            self.assertEqual(len(again['chunk_records']), 2)


if __name__ == '__main__':
    unittest.main()
