"""Avoid expensive work without losing updates, independent paths, or tag edits."""
from contextlib import ExitStack, closing
from pathlib import Path
import shutil
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pymupdf

import analysis_progress as analysis
import document_database as database
import document_search_engine as engine
import document_tag_store as tags
import model_runtime


class Model:
    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.asarray([[1., 0., 0.] if 'invoice' in t else [0., 1., 0.] for t in texts], dtype=np.float32)


class Tokenizer:
    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = self.root / 'store'
        self.db = self.store / database.DATABASE_NAME
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model = Model()
        self.factory = Mock(return_value=self.model)
        model_runtime._models.clear()
        model_runtime._categories.clear()
        self.stack.enter_context(patch.object(engine, 'SentenceTransformer', self.factory))
        self.stack.enter_context(patch('sentence_transformers.SentenceTransformer', self.factory))
        self.stack.enter_context(patch.object(engine.tiktoken, 'get_encoding', return_value=Tokenizer()))

    def pdf(self, name='invoice.pdf', text='invoice due tomorrow'):
        path = self.root / name
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((40, 40), text)
            pdf.save(path)
        return path

    def analyze(self, paths):
        worker = analysis.AnalysisWorker(paths, self.store)
        completed, failed = Mock(), Mock()
        worker.completed.connect(completed)
        worker.failed.connect(failed)
        worker.run()
        failed.assert_not_called()
        completed.assert_called_once()
        store, result = completed.call_args.args
        # The worker is incremental: it never reads the whole index back.
        self.assertIsNone(store['doc_records'])
        docs, chunks, vectors = database.load_index(self.store)
        return dict(store, doc_records=docs, chunk_records=chunks, chunk_vectors=vectors), result

    def test_unchanged_repeat_does_no_extraction_encoding_or_writes(self):
        path = self.pdf()
        self.analyze([path])
        before = self.db.read_bytes()
        calls = len(self.model.calls)
        with patch.object(engine, 'extract_document_chunks', side_effect=AssertionError('extracted twice')), \
                patch.object(engine, 'apply_index_updates', side_effect=AssertionError('rewrote index')), \
                patch.object(tags, 'save_file_tags', side_effect=AssertionError('rewrote categories')):
            store, result = self.analyze([path])
        self.assertEqual(store['added_doc_records'], [])
        self.assertEqual(len(self.model.calls), calls)
        self.factory.assert_called_once()
        self.assertEqual(before, self.db.read_bytes())
        self.assertEqual(result['skipped'], [(path, 'unchanged categories')])

    def test_unchanged_after_restart_does_not_load_model(self):
        path = self.pdf()
        self.analyze([path])
        model_runtime._models.clear()
        model_runtime._categories.clear()
        self.factory.reset_mock()
        self.analyze([path])
        self.factory.assert_not_called()

    def test_changed_content_replaces_chunks_preserves_id_and_unrelated_rows(self):
        first = self.pdf()
        second = self.pdf('travel.pdf', 'travel reservation')
        original, _ = self.analyze([first, second])
        first_id = original['doc_records'][0]['doc_id']
        second_id = original['doc_records'][1]['doc_id']
        with closing(sqlite3.connect(self.db)) as db:
            old_rows = db.execute('SELECT * FROM chunks WHERE doc_id=?', (second_id,)).fetchall()
        with pymupdf.open(first) as pdf:
            pdf[0].insert_text((40, 80), 'invoice updated amount 1000')
            pdf.saveIncr()
        store, _ = self.analyze([first, second])
        self.assertEqual(store['added_doc_records'][0]['doc_id'], first_id)
        self.assertEqual(len(store['doc_records']), 2)
        self.assertIn('updated', ' '.join(c['text'] for c in store['chunk_records'] if c['doc_id'] == first_id))
        with closing(sqlite3.connect(self.db)) as db:
            self.assertEqual(db.execute('SELECT * FROM chunks WHERE doc_id=?', (second_id,)).fetchall(), old_rows)
        loaded = database.load_index(self.store)
        np.testing.assert_array_equal(loaded[2], store['chunk_vectors'])

    def test_identical_copies_reuse_ocr_and_vectors_but_keep_both_paths(self):
        first = self.pdf(text='')
        second = self.root / 'invoice-checkpoint.pdf'
        shutil.copyfile(first, second)
        with patch.object(engine, 'extract_page_texts_with_ocr', return_value=[(1, 'invoice scanned')]) as ocr:
            store, _ = self.analyze([first, second])
        ocr.assert_called_once()
        self.assertEqual(len(store['doc_records']), 2)
        np.testing.assert_array_equal(store['chunk_vectors'][0], store['chunk_vectors'][1])
        self.assertEqual(len(tags.load_tag_rows(self.db)), 2)
        # The shared chunk is embedded once; categories reuse that vector.
        self.assertEqual(sum(call == ['invoice scanned'] for call in self.model.calls), 1)

    def test_copy_added_later_reuses_persisted_content_and_categories_after_restart(self):
        first = self.pdf()
        self.analyze([first])
        second = self.root / 'copy.pdf'
        shutil.copyfile(first, second)
        model_runtime._models.clear()
        model_runtime._categories.clear()
        self.factory.reset_mock()
        with patch.object(engine, 'extract_document_chunks', side_effect=AssertionError('duplicate extraction')):
            self.analyze([second])
        self.factory.assert_not_called()
        rows = tags.load_tag_rows(self.db)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['char_count'], rows[1]['char_count'])

    def test_tag_edit_retags_without_extraction_or_chunk_embeddings(self):
        path = self.pdf()
        self.analyze([path])
        tags.save_tag_definition('Research', 'research publication', self.db)
        with patch.object(engine, 'extract_document_chunks', side_effect=AssertionError('re-extracted PDF')), \
                patch.object(engine, 'apply_index_updates', side_effect=AssertionError('rewrote chunks')):
            store, result = self.analyze([path])
        self.assertEqual(store['added_doc_records'], [])
        self.assertEqual(len(result['tagged']), 1)
        # Initial chunks + categories; revised categories only. Documents are
        # categorized from stored chunk vectors, never re-encoded.
        self.assertEqual(len(self.model.calls), 3)
        self.factory.assert_called_once()

    def test_category_vectors_reused_for_additional_unique_imports(self):
        self.analyze([self.pdf()])
        self.analyze([self.pdf('travel.pdf', 'travel booking')])
        prototype_calls = [call for call in self.model.calls if len(call) == len(tags.TAG_PROTOTYPES)]
        self.assertEqual(len(prototype_calls), 1)
        self.factory.assert_called_once()

    def test_blank_scan_not_ocr_processed_again_when_unchanged(self):
        path = self.pdf(text='')
        with patch.object(engine, 'extract_page_texts_with_ocr', return_value=[]) as ocr:
            self.analyze([path])
            self.analyze([path])
            self.assertEqual(ocr.call_count, 1)
        self.factory.assert_not_called()

    def test_clear_removes_fingerprints_and_next_analysis_rebuilds(self):
        path = self.pdf()
        self.analyze([path])
        database.clear_database(self.store)
        self.assertEqual(database.load_processing_states(self.store), {})
        with patch.object(engine, 'extract_document_chunks', wraps=engine.extract_document_chunks) as extract:
            self.analyze([path])
        extract.assert_called_once()

    def test_incremental_write_error_rolls_back_old_chunks_and_state(self):
        path = self.pdf()
        self.analyze([path])
        old = database.load_index(self.store)
        states = database.load_processing_states(self.store)
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("CREATE TRIGGER fail_update BEFORE UPDATE ON documents BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        with pymupdf.open(path) as pdf:
            pdf[0].insert_text((40, 80), 'changed content')
            pdf.saveIncr()
        with self.assertRaises(sqlite3.IntegrityError):
            engine.update_vector_store([path], self.store)
        loaded = database.load_index(self.store)
        self.assertEqual(loaded[:2], old[:2])
        np.testing.assert_array_equal(loaded[2], old[2])
        self.assertEqual(database.load_processing_states(self.store), states)

    def test_touch_only_updates_fingerprint_without_reencoding_or_retagging(self):
        path = self.pdf()
        self.analyze([path])
        count = len(self.model.calls)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with patch.object(engine, 'extract_document_chunks', side_effect=AssertionError('Unchanged bytes')):
            store, _ = self.analyze([path])
        self.assertEqual(store['added_doc_records'], [])
        self.assertEqual(len(self.model.calls), count)

    def test_rename_and_remove_keep_unrelated_fingerprints_without_model_work(self):
        first = self.pdf()
        second = self.pdf('travel.pdf', 'travel hotel')
        self.analyze([first, second])
        old = database.load_processing_states(self.store)
        destination = self.root / 'renamed.pdf'
        first.rename(destination)
        model_runtime._models.clear()
        self.factory.reset_mock()
        self.assertTrue(engine.rename_document_in_vector_store(first, destination, self.store))
        self.analyze([destination, second])
        self.factory.assert_not_called()
        self.assertTrue(engine.remove_document_from_vector_store(destination, self.store))
        states = database.load_processing_states(self.store)
        self.assertEqual(len(states), 1)
        state = next(iter(states.values()))
        self.assertEqual(state, old[state['doc_id']])
        self.assertTrue(destination.exists())

    def test_changed_chunk_settings_rebuild_without_duplicate_rows(self):
        path = self.pdf(text='invoice ' * 30)
        first = engine.update_vector_store([path], self.store, chunk_size=60)
        again = engine.update_vector_store([path], self.store, chunk_size=120)
        self.assertEqual(len(again['doc_records']), 1)
        self.assertLess(len(again['chunk_records']), len(first['chunk_records']))
        loaded = database.load_index(self.store)
        self.assertEqual(len(loaded[1]), len(again['chunk_records']))

    def test_new_documents_share_one_embedding_batch_and_tag_without_encoding(self):
        store, result = self.analyze([self.pdf('first.pdf', 'invoice first'), self.pdf('second.pdf', 'travel second')])
        # Chunks from both files go through one encode call; the only other
        # call is the category descriptions. No document text is re-encoded.
        self.assertEqual(self.model.calls[0], ['invoice first', 'travel second'])
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(len(self.model.calls[1]), len(tags.TAG_PROTOTYPES))
        self.assertTrue({t['tags'][0]['tag'] for t in result['tagged']} <= set(tags.TAG_PROTOTYPES))
        self.assertEqual(len(result['tagged']), 2)

    def test_embedding_dimension_mismatch_fails_before_modifying_database(self):
        first = self.pdf()
        self.analyze([first])
        second = self.pdf('second.pdf', 'travel second')
        before = self.db.read_bytes()
        model = Mock()
        model.get_sentence_embedding_dimension.return_value = 4
        model.encode.return_value = np.array([[1., 0., 0., 0.]])
        with patch.object(engine, 'get_embedding_model', return_value=model):
            with self.assertRaisesRegex(ValueError, 'dimension'):
                engine.update_vector_store([second], self.store)
        self.assertEqual(self.db.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
