from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pymupdf as fitz

import document_database as database
import document_search_engine as engine
import document_tag_store as tags
import ocr


class Tokenizer:
    def encode(self, text):
        return list(text.encode())
    def decode(self, tokens):
        return bytes(tokens).decode()


class Model:
    def __init__(self, *args, **kwargs):
        pass
    def get_sentence_embedding_dimension(self):
        return 3
    def encode(self, texts, **kwargs):
        return np.array([[1, 0, 0] if 'invoice' in text else [0, 1, 0] for text in texts], dtype=np.float32)


class OCRIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pdf = self.root / 'scan.pdf'
        with fitz.open() as doc:
            doc.new_page()
            doc.new_page()
            doc.save(self.pdf)
        self.store = self.root / 'store'
        for patcher in [patch.object(engine, 'SentenceTransformer', Model),
                        patch.object(engine.tiktoken, 'get_encoding', return_value=Tokenizer())]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_text_pdf_and_mixed_pdf_do_not_run_ocr(self):
        with fitz.open(self.pdf) as doc:
            doc[0].insert_text((40, 40), 'Searchable invoice')
            doc.saveIncr()
        with patch.object(engine, 'extract_page_texts_with_ocr') as fallback:
            method, pages, chunks = engine.extract_document_chunks(self.pdf, Tokenizer())
        self.assertEqual(method, 'text')
        self.assertTrue(chunks)
        self.assertEqual([p for p, _ in pages], [1])
        fallback.assert_not_called()

    def test_zero_chunks_triggers_fallback_even_with_extracted_text(self):
        with patch.object(engine, 'extract_page_texts', return_value=('text', [(1, '   ')])), \
             patch.object(engine, 'extract_page_texts_with_ocr', return_value=[(2, 'invoice')]) as fallback:
            method, _, chunks = engine.extract_document_chunks(self.pdf, Tokenizer())
        self.assertEqual(method, 'ocr')
        self.assertEqual(chunks, [{'page': 2, 'text': 'invoice'}])
        fallback.assert_called_once()

    def test_encrypted_pdf_does_not_run_ocr(self):
        protected = self.root / 'encrypted.pdf'
        with fitz.open(self.pdf) as doc:
            doc.save(protected, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='secret')
        with patch.object(engine, 'extract_page_texts_with_ocr') as fallback:
            self.assertEqual(engine.extract_document_chunks(protected, Tokenizer()), ('encrypted', [], []))
        fallback.assert_not_called()

    def test_old_zero_chunk_scan_retries_without_duplicates_and_searches_after_reload(self):
        old = dict(doc_id='doc_00007', pdf_path=str(self.pdf), method='none', char_count=0)
        other = dict(doc_id='doc_00003', pdf_path=str(self.root / 'other.pdf'), method='text', char_count=6)
        original_chunk = dict(doc_id=other['doc_id'], pdf_path=other['pdf_path'], page=1, chunk_id=0, text='travel')
        database.save_index([old, other], [original_chunk], np.array([[0, 1, 0]], dtype=np.float32), self.store, engine.MODEL_NAME)
        with patch.object(engine, 'extract_page_texts_with_ocr', return_value=[(2, 'invoice paid')]) as fallback:
            result = engine.update_vector_store([self.pdf, self.pdf], self.store)
            again = engine.update_vector_store([self.pdf], self.store)
        fallback.assert_called_once()
        self.assertEqual(len(result['doc_records']), 2)
        self.assertEqual(result['added_doc_records'][0]['doc_id'], old['doc_id'])
        self.assertEqual(result['added_doc_records'][0]['method'], 'ocr')
        self.assertEqual(again['added_doc_records'], [])
        loaded = engine.load_vector_store(self.store)
        found = engine.search_index('invoice', loaded['model'], loaded['chunk_records'], loaded['chunk_vectors'], loaded['docs_by_id'])
        self.assertEqual(Path(found[0]['path']), self.pdf)
        self.assertEqual(found[0]['page'], 2)
        self.assertEqual(loaded['chunk_records'][0], original_chunk)
        np.testing.assert_array_equal(loaded['chunk_vectors'], [[0, 1, 0], [1, 0, 0]])

    def test_failed_ocr_leaves_existing_index_intact_and_can_retry(self):
        old = dict(doc_id='doc_00000', pdf_path=str(self.pdf), method='none', char_count=0)
        database.save_index([old], [], np.empty((0, 3)), self.store, engine.MODEL_NAME)
        with patch.object(engine, 'extract_page_texts_with_ocr', side_effect=ocr.OCRError('runtime unavailable')):
            with self.assertRaisesRegex(ocr.OCRError, 'runtime unavailable'):
                engine.update_vector_store([self.pdf], self.store)
        self.assertEqual(database.load_index(self.store)[0], [old])
        with patch.object(engine, 'extract_page_texts_with_ocr', return_value=[(1, 'invoice')]):
            self.assertEqual(engine.update_vector_store([self.pdf], self.store)['doc_records'][0]['method'], 'ocr')

    def test_indra_revision_change_reprocesses_scan_then_reuses_unchanged_result(self):
        with patch.object(engine, 'extract_page_texts_with_ocr', return_value=[(1, 'invoice indra')]) as fallback:
            first = engine.update_vector_store([self.pdf], self.store, ocr_model='indra')
            engine.update_vector_store([self.pdf], self.store, ocr_model='indra')
            fallback.assert_called_once()
            self.assertEqual(fallback.call_args.kwargs['ocr_model'], 'indra')
        with patch.object(ocr, 'MODEL_REVISION', 'updated-indra-revision'), \
                patch.object(engine, 'extract_page_texts_with_ocr', return_value=[(1, 'invoice updated')]) as fallback:
            changed = engine.update_vector_store([self.pdf], self.store, ocr_model='indra')
            engine.update_vector_store([self.pdf], self.store, ocr_model='indra')
            fallback.assert_called_once()
            self.assertEqual(fallback.call_args.kwargs['ocr_model'], 'indra')
        self.assertEqual(first['doc_records'][0]['doc_id'], changed['doc_records'][0]['doc_id'])
        self.assertEqual([c['text'] for c in changed['chunk_records']], ['invoice updated'])

    def test_blank_result_and_full_build_have_valid_empty_matrix(self):
        with patch.object(engine, 'extract_page_texts_with_ocr', return_value=[]):
            result = engine.build_index(self.root)
        self.assertEqual(result['doc_records'][0]['method'], 'none')
        self.assertEqual(result['chunk_vectors'].shape, (0, 3))

    def test_full_build_uses_ocr(self):
        with patch.object(engine, 'extract_page_texts_with_ocr', return_value=[(2, 'invoice')]):
            result = engine.build_index(self.root)
        self.assertEqual(result['doc_records'][0]['method'], 'ocr')
        self.assertEqual(result['chunk_records'][0]['page'], 2)

    def test_pdf_renderer_keeps_page_numbers_and_filters_empty_lines(self):
        predictor = Mock()
        predictor.predict.side_effect = [[{'rec_texts': ['', '  ']}], [{'rec_texts': [' invoice ', 'paid']}]]
        with patch.object(ocr, '_get_engine', return_value=predictor):
            result = ocr.extract_page_texts_with_ocr(self.pdf, ocr_model='indra')
        self.assertEqual(result, [(2, 'invoice\npaid')])
        for call in predictor.predict.call_args_list:
            raster = call.args[0]
            self.assertEqual(raster.ndim, 2)
            self.assertLessEqual(max(raster.shape[:2]), 3001)
            self.assertTrue(raster.flags.c_contiguous)

    def test_page_failure_is_visible_instead_of_saving_partial_ocr(self):
        predictor = Mock()
        predictor.predict.side_effect = [[{'rec_texts': ['invoice']}], RuntimeError('bad page')]
        with patch.object(ocr, '_get_engine', return_value=predictor):
            with self.assertRaisesRegex(ocr.OCRError, 'page 2'):
                ocr.extract_page_texts_with_ocr(self.pdf)

    def test_tagging_reuses_ocr_text_without_extraction(self):
        tagger = Mock()
        tagger.top_tags_many.return_value = [[{'tag': 'Finance', 'score': .9}]]
        with patch.object(tags, 'DocumentTagger', return_value=tagger), patch.object(tags, 'extract_text') as extract:
            result = tags.tag_files([self.pdf], self.store / database.DATABASE_NAME,
                                    texts_by_path={str(self.pdf): 'invoice paid'})
        extract.assert_not_called()
        tagger.top_tags_many.assert_called_once_with(['invoice paid'], top_k=3)
        self.assertEqual(len(result['tagged']), 1)
        self.assertEqual(tags.load_tag_rows(self.store / database.DATABASE_NAME)[0]['tag_1'], 'Finance')


if __name__ == '__main__':
    unittest.main()
