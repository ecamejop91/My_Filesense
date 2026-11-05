"""Real Qt worker/event-loop integration and real PDF progress without downloads."""
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import threading
from time import monotonic
import unittest
from unittest.mock import Mock, patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from qt_runtime import prepare_qt_plugins
prepare_qt_plugins()
import numpy as np
import pymupdf
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QListWidgetItem

import analysis_progress as analysis
import document_query_visuals as gui
import document_search_engine as engine
import document_database as database
import document_tag_store as tags
import ocr
from evaluation.performance import PerformanceController


class Tokenizer:
    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


class Model:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        return np.array([[1., 0., 0.] for text in texts], dtype=np.float32)


class AnalysisProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pdf = self.root / 'book.pdf'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.windows = []
        self.gates = []
        self.addCleanup(self.close_windows)

    def close_windows(self):
        for gate in self.gates:
            gate.set()
        for window in self.windows:
            worker = window._analysis_worker
            if worker is not None:
                worker.requestInterruption()
                worker.wait(5000)
                self.app.processEvents()
            window.close()
            window.deleteLater()
        self.app.processEvents()

    def wait_until(self, condition):
        deadline = monotonic() + 5
        while not condition() and monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(condition(), 'Timed out waiting for analysis')

    def window(self):
        for name, value in [('tag_names', ['Finance']), ('load_tag_rows', []),
                            ('load_tag_definitions', {'Finance': 'invoice'})]:
            self.stack.enter_context(patch.object(gui, name, return_value=value))
        self.stack.enter_context(patch('evaluation.performance.PerformanceController',
            side_effect=lambda parent, **kwargs: PerformanceController(
                parent, source_base=self.root, database=self.root / 'index.sqlite3',
                state_dir=self.root / 'state', run_dir=self.root / 'runs')))
        window = gui.DocumentQueryWindow()
        window.vector_store_dir = self.root / 'store'
        self.windows.append(window)
        item = QListWidgetItem(self.pdf.name)
        item.setData(Qt.ItemDataRole.UserRole, str(self.pdf))
        window.file_list.addItem(item)
        window.show()
        return window

    def store(self):
        return dict(model=Model(), doc_records=[], chunk_records=[], added_doc_records=[], skipped_paths=[])

    def blocked_index(self, paths, folder, progress, **options):
        self.worker_thread = threading.get_ident()
        progress(dict(stage='extracting', path=str(self.pdf), current=4, total=100,
                      detail='Reading page 5 of 100'))
        if not self.gates[-1].wait(5):
            raise RuntimeError('Test gate timed out')
        progress(dict(stage='embedding', path=str(self.pdf), current=32, total=96,
                      detail='Embedded 32 of 96 chunks'))
        return self.store()

    def test_worker_keeps_ui_alive_and_reports_pages_chunks_and_completion(self):
        window = self.window()
        gate = threading.Event()
        self.gates.append(gate)
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(10)
        self.addCleanup(timer.stop)
        with patch.object(analysis, 'update_vector_store', side_effect=self.blocked_index), \
                patch.object(analysis, 'tag_files', return_value={'tagged': []}):
            window.analyze_files()
            self.wait_until(lambda: len(ticks) >= 3 and window._analysis_dialog.stage_bar.maximum() == 100)
            self.assertNotEqual(self.worker_thread, threading.get_ident())
            self.assertEqual(window._analysis_dialog.file_label.text(), 'book.pdf')
            self.assertEqual(window._analysis_dialog.stage_bar.value(), 4)
            self.assertFalse(window.analyze_btn.isEnabled())
            worker = window._analysis_worker
            self.assertEqual(worker.ocr_model, 'indra')
            window.analyze_files()
            self.assertIs(window._analysis_worker, worker)
            gate.set()
            self.wait_until(lambda: window._analysis_worker is None)
        self.assertTrue(window.analyze_btn.isEnabled())
        self.assertEqual(window._analysis_dialog.stage_label.text(), 'Analysis complete')
        self.assertEqual(window._analysis_dialog.items[str(self.pdf)].text(2), 'Embedded 32 of 96 chunks')
        self.assertEqual(window.file_list.count(), 1)

    def test_model_selector_is_removed(self):
        window = self.window()
        self.assertFalse(hasattr(window, 'ocr_model_combo'))
        self.assertFalse(hasattr(window, 'model_settings'))

    def test_cancel_waits_for_current_call_then_restores_controls(self):
        window = self.window()
        gate = threading.Event()
        self.gates.append(gate)
        with patch.object(analysis, 'update_vector_store', side_effect=self.blocked_index), \
                patch.object(analysis, 'tag_files') as tag:
            window.analyze_files()
            self.wait_until(lambda: window._analysis_dialog.stage_bar.maximum() == 100)
            window._analysis_dialog.cancel_button.click()
            self.assertTrue(window._analysis_worker.isInterruptionRequested())
            gate.set()
            self.wait_until(lambda: window._analysis_worker is None)
        tag.assert_not_called()
        self.assertTrue(window.analyze_btn.isEnabled())
        self.assertIsNone(window.index_data)
        self.assertEqual(window._analysis_dialog.stage_label.text(), 'Analysis cancelled')

    def test_worker_failure_is_visible_and_allows_retry(self):
        window = self.window()
        with patch.object(analysis, 'update_vector_store', side_effect=RuntimeError('bad PDF')):
            window.analyze_files()
            self.wait_until(lambda: window._analysis_worker is None)
        self.assertIn('bad PDF', window._analysis_dialog.detail_label.text())
        self.assertTrue(window.analyze_btn.isEnabled())
        self.assertIsNone(window.index_data)

    def test_close_requests_safe_stop_and_waits_without_blocking_event_loop(self):
        window = self.window()
        self.gates.append(threading.Event())
        with patch.object(analysis, 'update_vector_store', side_effect=self.blocked_index):
            window.analyze_files()
            self.wait_until(lambda: window._analysis_dialog.stage_bar.maximum() == 100)
            window.close()
            self.assertTrue(window.isVisible())
            self.assertTrue(window._analysis_worker.isInterruptionRequested())
            self.gates[-1].set()
            self.wait_until(lambda: window._analysis_worker is None)
            self.wait_until(lambda: not window.isVisible())

    def test_real_pdf_progress_batches_align_vectors_and_reuses_model_for_tags(self):
        with pymupdf.open() as pdf:
            for _ in range(5):
                pdf.new_page().insert_textbox((40, 40, 550, 750), 'invoice payment ' * 100)
            pdf.save(self.pdf)
        events = []
        with patch.object(engine, 'SentenceTransformer', return_value=Model()) as constructor, \
                patch.object(engine.tiktoken, 'get_encoding', return_value=Tokenizer()), \
                patch('sentence_transformers.SentenceTransformer', side_effect=AssertionError('Duplicate model load')):
            store = engine.update_vector_store([self.pdf], self.root / 'store', progress=events.append)
            result = tags.tag_files([self.pdf], self.root / 'store' / database.DATABASE_NAME,
                texts_by_path={str(self.pdf): 'invoice payment'}, model=store['model'], progress=events.append)
        constructor.assert_called_once()
        self.assertEqual(len(result['tagged']), 1)
        embeddings = [e for e in events if e['stage'] == 'embedding']
        self.assertGreater(embeddings[-1]['total'], 32)
        self.assertEqual(embeddings[-1]['current'], len(store['chunk_records']))
        self.assertEqual(len(store['chunk_vectors']), len(store['chunk_records']))
        self.assertEqual({e['stage'] for e in events},
            {'loading', 'checking', 'extracting', 'chunking', 'embedding', 'saving', 'indexed', 'tagging', 'done'})
        loaded = database.load_index(self.root / 'store')
        np.testing.assert_array_equal(loaded[2], store['chunk_vectors'])

    def test_ocr_reports_page_before_and_after_prediction(self):
        with pymupdf.open() as pdf:
            pdf.new_page()
            pdf.new_page()
            pdf.save(self.pdf)
        events = []
        predictor = Mock()
        predictor.predict.return_value = [{'rec_texts': ['invoice']}]
        with patch.object(ocr, '_get_engine', return_value=predictor):
            pages = ocr.extract_page_texts_with_ocr(self.pdf, progress=events.append)
        self.assertEqual(len(pages), 2)
        self.assertEqual([e['current'] for e in events if 'current' in e], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
