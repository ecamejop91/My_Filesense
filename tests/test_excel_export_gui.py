"""Exercise the save-file workflow without opening Excel or the user's data."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from qt_runtime import prepare_qt_plugins
prepare_qt_plugins()

from PyQt6.QtWidgets import QApplication
from openpyxl import load_workbook

from evaluation.performance import PerformanceController
from evaluation.performance_dialog import PerformanceDialog


def report_fixture():
    return dict(schema_version=1, metadata=dict(cutoffs=[1, 5], depth=5, synthetic_demo=True,
                documents=3, benchmark_id='synthetic-export-test'),
                aggregate=[dict(system='tfidf', queries=1, eligible_recall_rr_ndcg_queries=1,
                    zero_relevant_queries=0, **{'macro_precision@5': .2, 'macro_recall@5': 1,
                    'mrr_at_depth': 1, 'macro_ndcg@5': 1, 'mean_latency_ms': 2})],
                per_query=[dict(system='tfidf', query_id='q-invoice', latency_ms=2,
                    **{'precision@5': .2, 'recall@5': 1, 'reciprocal_rank': 1, 'ndcg@5': 1})],
                rankings=[dict(system='tfidf', query_id='q-invoice', rank=1,
                    document_id='invoice.pdf', score=.8, relevance=3, judged=True)])


class ExcelExportGUIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.controller = PerformanceController(source_base=self.root,
            state_dir=self.root / 'state', run_dir=self.root / 'runs')
        self.dialog = PerformanceDialog(self.controller)
        self.controller.history.set_enabled(True)
        self.controller.record_search('invented invoice query', 'semantic', [dict(
            path='/synthetic/invoice.pdf', page=1, score=.8)], 12.5)
        self.addCleanup(self.close_widgets)

    def close_widgets(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.controller.shutdown()
        self.controller.deleteLater()
        self.app.processEvents()

    def test_history_save_creates_only_one_workbook_and_opens_it(self):
        existing = set(self.root.iterdir())
        output = self.root / 'history.xlsx'
        with patch('evaluation.performance_dialog.QFileDialog.getSaveFileName',
                   return_value=(str(output), 'Excel workbook (*.xlsx)')) as save, \
             patch('evaluation.performance_dialog.QDesktopServices.openUrl', return_value=True) as opened, \
             patch('evaluation.performance_dialog.QMessageBox.warning') as warning:
            self.dialog.export_history()
        save.assert_called_once()
        warning.assert_not_called()
        self.assertEqual(Path(opened.call_args.args[0].toLocalFile()), output)
        self.assertEqual(set(self.root.iterdir()) - existing, {output})
        book = load_workbook(output)
        try:
            self.assertEqual(book.sheetnames, ['Summary', 'Searches', 'Results'])
        finally:
            book.close()
        self.assertEqual(len(self.controller.history.records), 1)

    def test_evaluation_export_uses_snapshot_and_keeps_raw_reports_unchanged(self):
        output = self.root / 'evaluation.xlsx'
        raw = self.root / 'raw'
        raw.mkdir()
        source = raw / 'results.json'
        source.write_text('Raw output remains untouched.')
        self.controller.result = report_fixture()
        self.controller.report_dir = raw
        with patch('evaluation.performance_dialog.QFileDialog.getSaveFileName', return_value=(str(output), '')), \
             patch('evaluation.performance_dialog.QDesktopServices.openUrl', return_value=True), \
             patch('evaluation.performance_dialog.QMessageBox.warning') as warning:
            self.dialog.export_report()
        warning.assert_not_called()
        self.assertTrue(output.is_file())
        self.assertEqual(source.read_text(), 'Raw output remains untouched.')
        book = load_workbook(output)
        try:
            self.assertIn('Summary', book.sheetnames)
            self.assertIn('Rankings', book.sheetnames)
            self.assertIn('Metadata', book.sheetnames)
        finally:
            book.close()
        self.assertEqual([p.name for p in self.root.glob('*.xlsx')], ['evaluation.xlsx'])

    def test_cancel_does_not_write_and_missing_extension_is_added(self):
        with patch('evaluation.performance_dialog.QFileDialog.getSaveFileName', return_value=('', '')), \
             patch('evaluation.performance_dialog.export_history_workbook') as export:
            self.dialog.export_history()
        export.assert_not_called()
        with patch('evaluation.performance_dialog.QFileDialog.getSaveFileName',
                   return_value=(str(self.root / 'history'), '')):
            self.assertEqual(self.dialog._choose_excel_path('Export', 'history'), self.root / 'history.xlsx')

    def test_existing_file_and_missing_dependency_are_actionable(self):
        output = self.root / 'history.xlsx'
        output.write_bytes(b'existing report')
        with patch('evaluation.performance_dialog.QFileDialog.getSaveFileName', return_value=(str(output), '')), \
             patch('evaluation.performance_dialog.QDesktopServices.openUrl') as opened, \
             patch('evaluation.performance_dialog.QMessageBox.warning') as warning:
            self.dialog.export_history()
        self.assertEqual(output.read_bytes(), b'existing report')
        warning.assert_called_once()
        opened.assert_not_called()
        with patch('evaluation.performance_dialog.QFileDialog.getSaveFileName', return_value=(str(output), '')), \
             patch('evaluation.performance_dialog.export_history_workbook',
                   side_effect=ImportError('Install openpyxl in the FileSense environment.')), \
             patch('evaluation.performance_dialog.QMessageBox.warning') as warning:
            self.dialog.export_history()
        self.assertIn('Install openpyxl', warning.call_args.args[2])


if __name__ == '__main__':
    unittest.main()
