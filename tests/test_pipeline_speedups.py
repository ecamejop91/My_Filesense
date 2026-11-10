"""Indra OCR, 150 DPI, 50-token chunks, cross-file batches,
incremental index updates and tagging from stored chunk vectors."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pymupdf

import document_database as database
import document_search_engine as engine
import document_tag_store as tags
import model_runtime
import ocr


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


class OCRBackendTests(unittest.TestCase):
    def setUp(self):
        ocr._get_engine.cache_clear()
        self.addCleanup(ocr._get_engine.cache_clear)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pdf = Path(self.temp.name) / 'letter.pdf'
        with pymupdf.open() as doc:
            doc.new_page(width=612, height=792)  # US Letter
            doc.save(self.pdf)

    def test_default_indra_is_cached_on_windows_and_macos(self):
        package = Mock()
        for platform in ('win32', 'darwin'):
            ocr._get_engine.cache_clear()
            package.reset_mock()
            with patch.object(sys, 'platform', platform), \
                    patch.dict(sys.modules, {'indra_ocr': package}), \
                    patch.object(ocr, 'ensure_model_dir', return_value=self.pdf.parent):
                first = ocr._get_engine()
                self.assertIsInstance(first, ocr._IndraOCREngine)
                self.assertIs(ocr._get_engine(), first)
                package.load.assert_called_once()

    def test_unsupported_model_is_rejected_before_loading(self):
        with patch.object(ocr, 'ensure_model_dir') as prepare:
            with self.assertRaisesRegex(ValueError, 'Unknown OCR model'):
                ocr._get_engine('unsupported')
        prepare.assert_not_called()

    def test_backend_failure_is_actionable(self):
        with patch.object(ocr, 'ensure_model_dir', side_effect=RuntimeError('model download blocked')):
            with self.assertRaisesRegex(ocr.OCRError, 'Indra.*requirements.txt.*model download blocked'):
                ocr._get_engine()

    def test_pages_render_at_150_dpi(self):
        predictor = Mock()
        predictor.predict.return_value = [{'rec_texts': ['invoice']}]
        with patch.object(ocr, '_get_engine', return_value=predictor):
            self.assertEqual(ocr.extract_page_texts_with_ocr(self.pdf), [(1, 'invoice')])
        raster = predictor.predict.call_args.args[0]
        self.assertEqual(raster.shape, (1650, 1275))  # 11in x 8.5in at 150 DPI

    @unittest.skipUnless(sys.platform == 'win32', 'Windows OCR smoke test')
    def test_real_windows_backend_reads_rendered_text(self):
        with pymupdf.open() as doc:
            doc.new_page(width=612, height=792).insert_text((72, 100), 'Invoice 4729 total due', fontsize=20)
            doc.save(self.pdf)
        # Image-only copy so the text exists only as pixels.
        scanned = self.pdf.with_name('scanned.pdf')
        with pymupdf.open(self.pdf) as doc, pymupdf.open() as out:
            pix = doc[0].get_pixmap(dpi=150)
            out.new_page(width=612, height=792).insert_image(out[0].rect, pixmap=pix)
            out.save(scanned)
        self.assertIn('Indra', ocr.ocr_backend_name())
        pages = ocr.extract_page_texts_with_ocr(scanned)
        self.assertEqual(pages[0][0], 1)
        self.assertIn('4729', pages[0][1])


class StoreFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = self.root / 'store'
        self.db = self.store / database.DATABASE_NAME
        self.model = Model()
        model_runtime._models.clear()
        model_runtime._categories.clear()
        for patcher in [patch.object(engine, 'SentenceTransformer', Mock(return_value=self.model)),
                        patch('sentence_transformers.SentenceTransformer', Mock(return_value=self.model)),
                        patch.object(engine.tiktoken, 'get_encoding', return_value=Tokenizer())]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def pdf(self, name, text):
        path = self.root / name
        with pymupdf.open() as doc:
            doc.new_page().insert_text((40, 40), text)
            doc.save(path)
        return path


class IncrementalIndexTests(StoreFixture, unittest.TestCase):
    def test_default_reanalysis_replaces_old_chunks_without_overlap(self):
        path = self.pdf('default.pdf', 'invoice')
        text = 'abcdefghijklmnopqrstuvwxyz' * 4 + 'abcdefghijklmnopqrstu'
        with patch.object(engine, 'extract_page_texts', return_value=('text', [(1, text)])):
            old = engine.update_vector_store([path], self.store, chunk_size=200)
            current = engine.update_vector_store([path], self.store)
        self.assertEqual(len(old['chunk_records']), 1)
        chunks = current['chunk_records']
        self.assertEqual([len(Tokenizer().encode(c['text'])) for c in chunks], [50, 50, 25])
        self.assertEqual(''.join(c['text'] for c in chunks), text)
        docs, stored_chunks, vectors = database.load_index(self.store)
        self.assertEqual(len(docs), 1)
        self.assertEqual(stored_chunks, chunks)
        self.assertEqual(len(vectors), 3)

    def test_chunks_from_many_files_share_embedding_batches(self):
        paths = [self.pdf(f'f{i}.pdf', f'invoice number {i}') for i in range(5)]
        events = []
        engine.update_vector_store(paths, self.store, progress=events.append)
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(len(self.model.calls[0]), 5)
        embedding = [e for e in events if e['stage'] == 'embedding']
        self.assertEqual((embedding[-1]['current'], embedding[-1]['total']), (5, 5))

    def test_batches_are_capped(self):
        paths = [self.pdf(f'f{i}.pdf', f'invoice {i}') for i in range(5)]
        with patch.object(engine, 'EMBED_BATCH_SIZE', 2):
            engine.update_vector_store(paths, self.store)
        self.assertEqual([len(call) for call in self.model.calls], [2, 2, 1])

    def test_analysis_never_loads_the_whole_index(self):
        engine.update_vector_store([self.pdf(f'old{i}.pdf', f'travel {i}') for i in range(3)], self.store)
        new = self.pdf('new.pdf', 'invoice new')
        with patch.object(engine, 'load_index', side_effect=AssertionError('loaded whole index')), \
                patch.object(engine, 'load_document_chunks', wraps=engine.load_document_chunks) as partial:
            result = engine.update_vector_store([new], self.store, full_result=False)
        partial.assert_not_called()  # nothing to reuse, so no chunk rows were read
        self.assertIsNone(result['doc_records'])
        self.assertEqual([d['pdf_path'] for d in result['added_doc_records']], [new.as_posix()])
        self.assertEqual(result['added_chunk_vectors'].shape, (1, 3))
        docs, chunks, vectors = database.load_index(self.store)
        self.assertEqual(len(docs), 4)
        np.testing.assert_array_equal(vectors[-1], [1, 0, 0])

    def test_identical_copy_reads_only_the_matching_document(self):
        first = self.pdf('first.pdf', 'invoice shared')
        other = self.pdf('other.pdf', 'travel other')
        engine.update_vector_store([first, other], self.store)
        copy = self.root / 'copy.pdf'
        copy.write_bytes(first.read_bytes())
        calls = len(self.model.calls)
        with patch.object(engine, 'load_document_chunks', wraps=engine.load_document_chunks) as partial, \
                patch.object(engine, 'extract_document_chunks', side_effect=AssertionError('re-extracted')):
            result = engine.update_vector_store([copy], self.store, full_result=False)
        self.assertEqual(len(partial.call_args_list), 1)
        self.assertEqual(len(list(partial.call_args.args[1])), 1)
        self.assertEqual(len(self.model.calls), calls)
        np.testing.assert_array_equal(result['added_chunk_vectors'], [[1, 0, 0]])

    def test_merge_delta_matches_reloaded_index(self):
        first = self.pdf('first.pdf', 'invoice one')
        second = self.pdf('second.pdf', 'travel two')
        engine.update_vector_store([first, second], self.store)
        in_memory = engine.load_vector_store(self.store)
        with pymupdf.open(first) as doc:
            doc[0].insert_text((40, 80), 'travel changed')
            doc.saveIncr()
        third = self.pdf('third.pdf', 'invoice three')
        delta = engine.update_vector_store([first, third], self.store, full_result=False)
        merged = engine.merge_index_delta(in_memory, delta)
        reloaded = engine.load_vector_store(self.store)
        self.assertEqual(merged['doc_records'], reloaded['doc_records'])
        self.assertEqual(merged['chunk_records'], reloaded['chunk_records'])
        np.testing.assert_array_equal(merged['chunk_vectors'], reloaded['chunk_vectors'])
        self.assertEqual(merged['docs_by_id'], reloaded['docs_by_id'])

    def test_merge_into_empty_index(self):
        empty = dict(model=None, doc_records=[], chunk_records=[],
                     chunk_vectors=np.empty((0, 0), np.float32), docs_by_id={})
        delta = engine.update_vector_store([self.pdf('a.pdf', 'invoice a')], self.store, full_result=False)
        merged = engine.merge_index_delta(empty, delta)
        self.assertEqual(merged['chunk_vectors'].shape, (1, 3))
        self.assertEqual(len(merged['doc_records']), 1)


class VectorTaggingTests(StoreFixture, unittest.TestCase):
    def test_tags_come_from_stored_chunk_vectors_without_encoding_text(self):
        path = self.pdf('bill.pdf', 'invoice payment due')
        engine.update_vector_store([path], self.store)
        plan = tags.prepare_tagging([path], self.db)
        stored = tags.load_tagging_vectors(plan, self.db)
        self.assertEqual(set(stored), {str(path)})
        before = list(self.model.calls)
        with patch.object(tags, 'extract_text', side_effect=AssertionError('extracted text')):
            result = tags.tag_files([path], self.db, vectors_by_path=stored, model=self.model, plan=plan)
        new_calls = self.model.calls[len(before):]
        # Only the category descriptions are encoded (once per model).
        self.assertEqual([len(call) for call in new_calls], [len(tags.TAG_PROTOTYPES)])
        self.assertEqual(len(result['tagged']), 1)
        self.assertIn(result['tagged'][0]['tags'][0]['tag'], {'Finance', 'Professional'})
        self.assertGreater(result['tagged'][0]['char_count'], 0)

    def test_document_vector_is_normalized_mean(self):
        tagger = tags.DocumentTagger(model=self.model, definitions={'A': 'invoice', 'B': 'travel'})
        mixed = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32)
        (top,) = tagger.top_tags_from_vectors([mixed], top_k=2)
        self.assertEqual([t['tag'] for t in top], ['A', 'B'])
        self.assertAlmostEqual(top[0]['score'], 2 / np.sqrt(5), places=5)
        self.assertEqual(tagger.top_tags_from_vectors([np.empty((0, 3))]), [[]])


if __name__ == '__main__':
    unittest.main()
