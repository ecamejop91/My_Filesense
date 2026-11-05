"""Offscreen Qt integration with synthetic search rows and a lightweight child worker."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from time import monotonic
import unittest
from unittest.mock import Mock, patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from qt_runtime import prepare_qt_plugins
prepare_qt_plugins()

from PyQt6.QtCore import QProcess
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel

import document_query_visuals as gui
from evaluation.performance import PerformanceController
from evaluation.performance_dialog import PerformanceDialog


def fixture_report():
    return dict(
        metadata=dict(cutoffs=[1, 3], depth=3, synthetic_demo=True, documents=2),
        aggregate=[dict(system='tfidf', queries=2, eligible_recall_rr_ndcg_queries=1,
                        zero_relevant_queries=1, **{'macro_precision@3': 1 / 3,
                        'macro_recall@3': 1.0, 'mrr_at_depth': 1.0,
                        'macro_ndcg@3': 1.0, 'mean_latency_ms': 2.0})],
        per_query=[dict(system='tfidf', query_id='q-no-answer', **{'precision@3': 0.0,
                        'recall@3': None, 'reciprocal_rank': None,
                        'ndcg@3': None, 'latency_ms': 2.0})],
        rankings=[dict(system='tfidf', query_id='q-answer', rank=1,
                       document_id='invoice.pdf', score=0.75, relevance=3)])


FAKE_WORKER = '''
import json
import os
from pathlib import Path
import sys
import time

prefix = 'FILESENSE_EVAL_EVENT:'
def emit(event, **values):
    print(prefix + json.dumps(dict(event=event, **values)), flush=True)

base = Path(__file__).resolve().parents[1]
job = Path(sys.argv[sys.argv.index('--job-dir') + 1])
job.mkdir(parents=True)
(job / 'environment.json').write_text(json.dumps({key: os.environ.get(key) for key in (
    'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY')}))
behavior = (base / 'behavior.txt').read_text()
print('ordinary diagnostic output', flush=True)
print(prefix + 'not-json', flush=True)
emit('status', message='Synthetic worker is ready')
if behavior == 'wait':
    time.sleep(30)
elif behavior == 'error':
    emit('error', message='Synthetic answer key is invalid; supply complete labels.')
    sys.exit(2)
else:
    time.sleep(0.05)
    report = job / 'report'
    report.mkdir()
    (report / 'results.json').write_text((base / 'fixture-report.json').read_text())
    if behavior != 'no-complete':
        emit('complete', report_dir='/untrusted/printed/path', result_path='/untrusted/results.json')
'''


class PerformanceGUIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.resources = []
        self.addCleanup(self.close_resources)

    def close_resources(self):
        for resource in reversed(self.resources):
            if isinstance(resource, PerformanceController):
                resource.shutdown()
            else:
                resource.close()
            resource.deleteLater()
        self.app.processEvents()

    def controller(self, source=None):
        controller = PerformanceController(source_base=source or self.root,
                                           database=self.root / 'synthetic.sqlite3',
                                           state_dir=self.root / 'state', run_dir=self.root / 'runs')
        self.resources.append(controller)
        return controller

    def wait_until(self, condition, timeout=5):
        deadline = monotonic() + timeout
        while not condition() and monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(condition(), 'Timed out waiting for Qt process state')

    def fake_source(self, behavior='complete'):
        source = self.root / 'worker-source'
        package = source / 'evaluation'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        (package / 'gui_worker.py').write_text(FAKE_WORKER)
        (source / 'behavior.txt').write_text(behavior)
        (source / 'fixture-report.json').write_text(json.dumps(fixture_report()))
        return source

    def window(self):
        invoice = str(self.root / 'invoice.pdf')
        hotel = str(self.root / 'hotel.pdf')
        rows = [dict(file_path=path, file_name=Path(path).name, tag_1=tag, tag_2='', tag_3='')
                for path, tag in ((invoice, 'Finance'), (hotel, 'Travel'))]
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(gui, 'tag_names', return_value=['Finance', 'Travel']))
        stack.enter_context(patch.object(gui, 'tag_color', return_value='#457563'))
        stack.enter_context(patch.object(gui, 'load_tag_definitions', return_value={'Finance': 'invoice'}))
        stack.enter_context(patch.object(gui, 'load_tag_rows', return_value=rows))
        tag_search = stack.enter_context(patch.object(gui, 'exact_tag_matches', return_value=[]))

        def isolated_controller(parent, **kwargs):
            return PerformanceController(parent, source_base=self.root,
                                         database=self.root / 'synthetic.sqlite3',
                                         state_dir=self.root / 'state', run_dir=self.root / 'runs')

        stack.enter_context(patch('evaluation.performance.PerformanceController', side_effect=isolated_controller))
        window = gui.DocumentQueryWindow()
        self.resources.append(window)
        window._show_warning = Mock()
        for checkbox in window.tag_filters.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        window.index_data = dict(model=object(), doc_records=[dict(doc_id='fixture')],
                                 chunk_records=[], chunk_vectors=[], docs_by_id={})
        semantic_rows = [dict(path=path, document=Path(path).name, page=page, score=score, text='Invented text')
                         for path, page, score in [(invoice, 1, .9), (hotel, 2, .8), (invoice, 3, .7)]]
        return window, semantic_rows, tag_search

    def test_tracking_toggle_persists_but_query_history_is_session_only(self):
        controller = self.controller()
        self.assertFalse(controller.history.enabled)
        controller.record_search('disabled query', 'semantic', [], 2)
        self.assertEqual(controller.history.records, [])
        controller.set_tracking(True)
        controller.record_search('invented secret query', 'semantic', [], 3)
        self.assertEqual(len(controller.history.records), 1)
        reloaded = self.controller()
        self.assertTrue(reloaded.history.enabled)
        self.assertEqual(reloaded.history.records, [])
        files = [path for path in self.root.rglob('*') if path.is_file()]
        self.assertEqual([path.name for path in files], ['performance.ini'])
        self.assertNotIn('invented secret query', files[0].read_text())
        controller.set_tracking(False)
        controller.record_search('paused query', 'semantic', [], 2)
        self.assertEqual(len(controller.history.records), 1)
        controller.clear_history()
        self.assertFalse(controller.history.enabled)
        self.assertEqual(controller.history.records, [])

    def test_semantic_search_records_only_displayed_rows_without_extra_retrieval(self):
        window, hits, _ = self.window()
        window.query_input.setText('  invented invoice question  ')
        with patch.object(gui, 'search_index', return_value=hits) as search:
            window.search_documents()
            self.assertEqual(search.call_count, 1)
            self.assertEqual(window.performance.history.records, [])
            window.performance.set_tracking(True)
            # Filtering the previous display must not become a new search record.
            window.tag_filters['Travel'].setChecked(False)
            self.assertEqual(search.call_count, 1)
            self.assertEqual(window.performance.history.records, [])
            window.search_documents()
            self.assertEqual(search.call_count, 2)
        record = window.performance.history.records[0]
        self.assertEqual(record['query'], 'invented invoice question')
        self.assertEqual(record['kind'], 'semantic')
        self.assertEqual(record['result_count'], 2)
        self.assertEqual([hit['page'] for hit in record['results']], [1, 3])
        self.assertEqual([hit['rank'] for hit in record['results']], [1, 2])
        self.assertTrue(all(hit['document'] == 'invoice.pdf' for hit in record['results']))
        self.assertNotIn('hotel.pdf', window.results_view.toPlainText())
        self.assertEqual(record['metrics']['quality_status'], 'not_evaluated')
        self.assertGreaterEqual(record['elapsed_ms'], 0)

    def test_tag_empty_blank_and_failed_search_hooks(self):
        window, hits, tags = self.window()
        window.performance.set_tracking(True)
        tags.return_value = [dict(file_path=hits[0]['path'], file_name='invoice.pdf', score=.8)]
        with patch.object(gui, 'search_index') as search:
            window.query_input.setText('Finance')
            window.search_documents()
            search.assert_not_called()
            self.assertEqual(window.performance.history.records[0]['kind'], 'tag')
            self.assertIsNone(window.performance.history.records[0]['results'][0]['page'])
            tags.return_value = []
            window.search_documents()
            self.assertEqual(window.performance.history.records[0]['result_count'], 0)
            window.query_input.setText('  ')
            window.search_documents()
            self.assertEqual(len(window.performance.history.records), 2)
            window._show_warning.assert_called_with('Enter a query before searching.')
            window.query_input.setText('invented unmatched question')
            search.return_value = []
            window.search_documents()
            self.assertEqual(search.call_count, 1)
            self.assertEqual(window.performance.history.records[0]['kind'], 'semantic')
            self.assertEqual(window.performance.history.records[0]['result_count'], 0)
            search.side_effect = RuntimeError('synthetic failure')
            window.search_documents()
            self.assertEqual(len(window.performance.history.records), 3)
            window._show_warning.assert_called_with('Search failed: synthetic failure')

    def test_dialog_toggle_history_quality_labels_and_report_rendering(self):
        controller = self.controller()
        dialog = PerformanceDialog(controller)
        self.resources.append(dialog)
        self.assertFalse(dialog.tracking_toggle.isChecked())
        dialog.tracking_toggle.setChecked(True)
        controller.record_search('invented query', 'semantic', [dict(
            path=str(self.root / 'invoice.pdf'), page=2, score=.75)], 2.5)
        self.assertEqual(dialog.history_table.rowCount(), 1)
        self.assertEqual(dialog.history_table.item(0, 1).text(), 'invented query')
        self.assertEqual(dialog.search_table.item(0, 3).text(), '0.750')
        labels = ' '.join(label.text() for label in dialog.findChildren(QLabel))
        self.assertIn('Quality not evaluated', labels)
        self.assertIn('No extra search or model run', labels)
        self.assertIn('invented PDFs', dialog.mode_note.text())
        controller.result = fixture_report()
        controller.state = 'complete'
        controller.report_dir = self.root / 'synthetic-report'
        controller.job_changed.emit()
        self.assertEqual(dialog.summary_table.rowCount(), 1)
        self.assertEqual(dialog.summary_table.horizontalHeaderItem(1).text(), 'Precision@3')
        self.assertEqual(dialog.summary_table.item(0, 0).text(), 'TF-IDF')
        self.assertEqual(dialog.query_table.item(0, 3).text(), 'N/A')
        self.assertIn('SAMPLE DATA', dialog.result_note.text())
        self.assertIn('undefined', dialog.result_note.text())
        self.assertTrue(dialog.export_report_button.isEnabled())
        dialog.clear_button.click()
        self.assertEqual(dialog.history_table.rowCount(), 0)
        self.assertTrue(controller.history.enabled)

    def test_qprocess_status_completion_and_offline_environment(self):
        controller = self.controller(self.fake_source())
        messages = []
        controller.job_changed.connect(lambda: messages.append(controller.message))
        controller.start(depth=3, systems='tfidf')
        self.assertTrue(controller.busy)
        with self.assertRaisesRegex(ValueError, 'already running'):
            controller.start()
        self.wait_until(lambda: not controller.busy)
        self.assertEqual(controller.state, 'complete', controller.message)
        self.assertIn('Synthetic worker is ready', messages)
        self.assertEqual(controller.result, fixture_report())
        self.assertEqual(controller.report_dir, controller._job_dir / 'report')
        self.assertNotEqual(str(controller.report_dir), '/untrusted/printed/path')
        self.assertIsNone(controller.process)
        environment = json.loads((controller._job_dir / 'environment.json').read_text())
        self.assertEqual(set(environment.values()), {'1'})

    def test_qprocess_error_and_missing_completion_are_not_success(self):
        source = self.fake_source('error')
        controller = self.controller(source)
        controller.start()
        self.wait_until(lambda: not controller.busy)
        self.assertEqual(controller.state, 'failed')
        self.assertIn('supply complete labels', controller.message)
        self.assertIsNone(controller.result)
        (source / 'behavior.txt').write_text('no-complete')
        controller.start()
        self.wait_until(lambda: not controller.busy)
        self.assertEqual(controller.state, 'failed')
        self.assertIsNone(controller.report_dir)

    def test_closing_panel_keeps_worker_and_shutdown_cancels_it(self):
        controller = self.controller(self.fake_source('wait'))
        dialog = PerformanceDialog(controller)
        self.resources.append(dialog)
        dialog.show()
        controller.start()
        self.wait_until(lambda: controller.message == 'Synthetic worker is ready')
        self.assertFalse(dialog.run_button.isEnabled())
        self.assertTrue(dialog.cancel_button.isEnabled())
        dialog.close()
        self.assertTrue(controller.busy)
        self.assertIsNotNone(controller.process)
        self.assertEqual(controller.process.state(), QProcess.ProcessState.Running)
        controller.shutdown()
        self.wait_until(lambda: not controller.busy)
        self.assertEqual(controller.state, 'idle')
        self.assertIn('cancelled', controller.message)
        self.assertIsNone(controller.result)
        self.assertIsNone(controller.process)

    def test_advanced_menu_opens_one_modeless_panel(self):
        window, _, _ = self.window()
        actions = window.advanced_button.menu().actions()
        self.assertEqual([action.text() for action in actions if not action.isSeparator()],
                         ['Performance…', 'Import Existing Library…', 'Clear Database'])
        actions[0].trigger()
        first = window.performance_dialog
        self.assertIsInstance(first, PerformanceDialog)
        self.assertFalse(first.isModal())
        first.close()
        actions[0].trigger()
        self.assertIs(window.performance_dialog, first)
        with patch.object(window.performance, 'shutdown') as shutdown:
            window.close()
            shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
