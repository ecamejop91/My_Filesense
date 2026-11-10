"""Actual Office/RTF files through extraction, indexing, tagging and reload."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches

import analysis_progress as analysis
import document_database as database
import document_formats as formats
import document_search_engine as engine
import document_tag_store as tags
import model_runtime


class Tokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return ''.join(tokens)


class Model:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        return np.array([[1., 0., 0.] if 'invoice' in text.lower() else [0., 1., 0.]
                         for text in texts], dtype=np.float32)


def make_documents(root):
    doc = Document()
    doc.sections[0].header.paragraphs[0].text = 'Company header'
    doc.add_paragraph('Invoice for office maintenance.')
    cell = doc.add_table(rows=1, cols=2).cell(0, 0)
    cell.text = 'invoice table'
    cell.add_table(rows=1, cols=1).cell(0, 0).text = 'Nested details'
    doc.add_paragraph('Payment due within thirty days.')
    doc.save(root / 'letter.docx')
    deck = Presentation()
    deck.slides.add_slide(deck.slide_layouts[6])  # Retain blank slide numbering.
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    group = slide.shapes.add_group_shape()
    group.shapes.add_textbox(0, 0, Inches(4), Inches(1)).text = 'Invoice presentation'
    slide.shapes.add_table(1, 1, 0, Inches(2), Inches(4), Inches(1)).table.cell(0, 0).text = 'Table amount 125.50'
    slide.notes_slide.notes_text_frame.text = 'Speaker note: payment schedule'
    deck.save(root / 'slides.pptx')
    book = Workbook()
    book.active.title = 'Empty'
    sheet = book.create_sheet('Invoices')
    sheet.append(['Invoice', 'Amount'])
    sheet.append(['invoice 4729', 125.5])
    sheet.append(['zero balance', 0])
    book.save(root / 'book.xlsx')
    book.close()
    shutil.copyfile(Path(__file__).parent / 'fixtures/invoice.xls', root / 'old-book.xls')
    (root / 'notes.rtf').write_bytes(
        b"{\\rtf1\\ansi\\ansicpg1252 Invoice \\b payment\\b0\\par Caf\\'e9 \\u8364? \x96 receipt}")
    return [root / name for name in ('letter.docx', 'slides.pptx', 'book.xlsx', 'old-book.xls', 'notes.rtf')]


class DocumentFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.paths = make_documents(self.root)

    def test_actual_readers_preserve_tables_notes_values_and_unicode(self):
        expected = [('Company header', 'invoice table', 'Nested details', 'Payment due'),
                    ('Invoice presentation', 'Table amount', 'Speaker note'),
                    ('Invoices', 'invoice 4729', '125.5', 'zero balance\t0'),
                    ('Invoices', 'invoice 4729', '125.5', 'zero balance\t0'),
                    ('Invoice payment', 'Caf\u00e9', '\u20ac', '\u2013 receipt')]
        with patch.object(engine, 'extract_page_texts_with_ocr', side_effect=AssertionError('Office file sent to OCR')):
            for path, needles in zip(self.paths, expected):
                with self.subTest(path=path.name):
                    method, sections, chunks = engine.extract_document_chunks(path, Tokenizer())
                    self.assertEqual(method, 'text')
                    text = '\n'.join(t for _, t in sections)
                    for needle in needles:
                        self.assertIn(needle, text)
                    self.assertTrue(chunks)
                    if path.suffix in ('.pptx', '.xlsx', '.xls'):
                        self.assertEqual(sections[0][0], 2)
        word = formats.extract_office_texts(self.paths[0])[1][0][1]
        self.assertLess(word.index('invoice table'), word.index('Payment due'))

    def test_blank_office_files_do_not_trigger_pdf_ocr(self):
        for name, doc in [('blank.docx', Document()), ('blank.pptx', Presentation())]:
            path = self.root / name
            doc.save(path)
            with patch.object(engine, 'extract_page_texts_with_ocr') as ocr:
                method, sections, chunks = engine.extract_document_chunks(path, Tokenizer(), progress=Mock())
            self.assertEqual((method, sections, chunks), ('none', [], []))
            ocr.assert_not_called()

    def test_bad_office_file_raises_instead_of_saving_an_empty_success(self):
        for suffix in ('.docx', '.pptx', '.xlsx', '.xls', '.rtf'):
            path = self.root / ('broken' + suffix)
            path.write_bytes(b'not a document')
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'Could not read'):
                formats.extract_office_texts(path)

    def test_progress_cancellation_is_not_reclassified_as_a_reader_error(self):
        stop = analysis.AnalysisCancelled()
        with self.assertRaises(analysis.AnalysisCancelled), \
                patch.object(engine, 'extract_page_texts_with_ocr', side_effect=AssertionError('Used PDF OCR')):
            engine.extract_document_chunks(self.paths[1], Tokenizer(), progress=Mock(side_effect=stop))

    def test_directory_import_handles_case_and_ignores_office_lock_files(self):
        capitalized = self.root / 'CAPITAL.DOCX'
        shutil.copyfile(self.paths[0], capitalized)
        shutil.copyfile(self.paths[0], self.root / '~$letter.docx')
        found = set(engine.iter_document_files(self.root))
        self.assertEqual(found, set(self.paths + [capitalized]))

    def test_mixed_files_index_tag_reload_search_and_skip_without_reextraction(self):
        model_runtime._models.clear()
        model_runtime._categories.clear()
        self.addCleanup(model_runtime._models.clear)
        self.addCleanup(model_runtime._categories.clear)
        store = self.root / 'store'
        with patch.object(engine, 'SentenceTransformer', return_value=Model()), \
                patch.object(engine.tiktoken, 'get_encoding', return_value=Tokenizer()):
            worker = analysis.AnalysisWorker(self.paths, store)
            failed, completed = Mock(), Mock()
            worker.failed.connect(failed)
            worker.completed.connect(completed)
            worker.run()
            failed.assert_not_called()
            completed.assert_called_once()
            self.assertEqual(len(completed.call_args.args[1]['tagged']), 5)
            loaded = engine.load_vector_store(store)
            self.assertEqual(len(loaded['doc_records']), 5)
            hits = engine.search_index('invoice', loaded['model'], loaded['chunk_records'],
                                       loaded['chunk_vectors'], loaded['docs_by_id'], top_k=len(loaded['chunk_records']))
            self.assertEqual({h['path'] for h in hits}, {p.as_posix() for p in self.paths})
            self.assertEqual(len(tags.load_tag_rows(store / database.DATABASE_NAME)), 5)
            with patch.object(engine, 'extract_document_chunks', side_effect=AssertionError('Unchanged file re-extracted')):
                again = engine.update_vector_store(self.paths, store)
            self.assertEqual(again['added_doc_records'], [])
            wrong_extension = self.root / 'not-a-presentation.pptx'
            shutil.copyfile(self.paths[0], wrong_extension)
            with self.assertRaises(ValueError):
                engine.update_vector_store([wrong_extension], store)
            self.assertEqual(len(database.load_index(store)[0]), 5)

    def test_progress_and_result_locations_use_document_type(self):
        for path, label in [(self.paths[1], 'Slide 2'), (self.paths[2], 'Sheet 2'), (self.paths[0], 'Document')]:
            self.assertEqual(formats.location_label(path, 2), label)
            events = []
            engine.extract_document_chunks(path, Tokenizer(), progress=events.append)
            self.assertTrue(all(e.get('unit') == formats.progress_unit(path) for e in events))
