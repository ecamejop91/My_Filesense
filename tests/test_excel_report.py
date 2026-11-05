"""Read exported workbooks back to verify exact values, types and boundaries."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import load_workbook
from evaluation import excel_report as excel


def history():
    return [dict(search_id=2, timestamp='2026-10-27T08:00:00-05:00', query='=1+1',
                 kind='semantic', elapsed_ms=12.25, result_count=2,
                 metrics={'quality_status': 'not_evaluated'}, results=[
                     dict(rank=1, document='invoice.pdf', path='/synthetic/invoice.pdf', page=1, score=.8),
                     dict(rank=2, document='invoice.pdf', path='/synthetic/invoice.pdf', page=2, score=-.2)]),
            dict(search_id=1, timestamp='2026-10-27T12:00:00+00:00', query='empty',
                 kind='tag', elapsed_ms=0, result_count=0, results=[])]


def report():
    return dict(schema_version=1, metadata=dict(cutoffs=[1, 5], depth=5, synthetic_demo=True,
                documents=3, benchmark_id='fixture', nested={'a/b': {'~key': [None, False, '=text']}}),
                aggregate=[dict(system='tfidf', queries=2, **{'macro_precision@5': .1,
                    'macro_recall@5': 1, 'mrr_at_depth': 1, 'macro_ndcg@5': 1,
                    'pooled_precision@5': .1, 'pooled_recall@5': 1, 'mean_latency_ms': 2,
                    'macro_precision@1': .5})],
                per_query=[dict(system='tfidf', query_id='=1+1', **{'precision@5': 0,
                    'recall@5': None, 'reciprocal_rank': None, 'ndcg@5': None})],
                rankings=[dict(system='tfidf', query_id='=1+1', rank=1,
                    document_id='@document.pdf', score=0, relevance=0, judged=True)])


def rows(sheet):
    header = int(''.join(filter(str.isdigit, sheet.auto_filter.ref.split(':')[0])))
    fields = [cell.value for cell in sheet[header]]
    return [dict(zip(fields, values)) for values in sheet.iter_rows(min_row=header + 1, values_only=True)]


class ExcelReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def book(self, path):
        book = load_workbook(path)
        self.addCleanup(book.close)
        return book

    def test_history_preserves_empty_search_duplicates_dates_and_numeric_values(self):
        output = excel.export_history_workbook(history(), self.root / 'history.xlsx')
        book = self.book(output)
        self.assertEqual(book.sheetnames, ['Summary', 'Searches', 'Results'])
        searches = rows(book['Searches'])
        results = rows(book['Results'])
        self.assertEqual([row['Search ID'] for row in searches], [2, 1])
        self.assertEqual(searches[0]['Time (UTC)'], datetime(2026, 10, 27, 13))
        self.assertEqual(searches[0]['Response (ms)'], 12.25)
        self.assertEqual(searches[1]['Displayed results'], 0)
        self.assertEqual([row['PDF'] for row in results], ['invoice.pdf', 'invoice.pdf'])
        self.assertEqual([row['Score'] for row in results], [.8, -.2])
        self.assertEqual(results[0]['Query'], '=1+1')
        summary = {row['Measure']: row['Value'] for row in rows(book['Summary'])}
        self.assertEqual(summary['Searches'], 2)
        self.assertEqual(summary['Searches without results'], 1)
        self.assertIn('Not evaluated', summary['Quality'])
        self.assertEqual(list(self.root.iterdir()), [output])

    def test_evaluation_preserves_all_fields_nulls_provenance_and_formats(self):
        source = report()
        book = self.book(excel.export_evaluation_workbook(source, self.root / 'evaluation.xlsx'))
        self.assertEqual(book.sheetnames, ['Summary', 'Aggregate', 'Per query', 'Rankings', 'Metadata'])
        self.assertEqual(rows(book['Aggregate']), source['aggregate'])
        self.assertEqual(rows(book['Rankings']), source['rankings'])
        query = rows(book['Per query'])[0]
        self.assertEqual(query['precision@5'], 0)
        self.assertEqual(query['recall@5'], 'N/A')
        metadata = {row['Path']: row for row in rows(book['Metadata'])}
        self.assertEqual(metadata['/metadata/nested/a~1b/~0key/0']['Type'], 'null')
        self.assertEqual(metadata['/metadata/nested/a~1b/~0key/1']['Value'], False)
        self.assertEqual(metadata['/metadata/nested/a~1b/~0key/2']['Value'], '=text')
        self.assertIn('SAMPLE DATA', book['Summary']['A2'].value)
        for sheet in book:
            self.assertTrue(sheet.freeze_panes)
            self.assertTrue(sheet.auto_filter.ref)
            self.assertIn('A1:', str(sheet.merged_cells))
        for row in book['Aggregate']:
            for cell in row:
                if cell.value == .1:
                    self.assertEqual(cell.number_format, '0.0%')

    def test_source_text_never_becomes_a_formula_or_link(self):
        records = history()
        records[0]['query'] = '#N/A'
        records[0]['results'][0]['document'] = '=HYPERLINK("https://example.com")'
        records[0]['results'][0]['path'] = 'https://example.com'
        book = self.book(excel.export_history_workbook(records, self.root / 'literal.xlsx'))
        for sheet in book:
            for row in sheet:
                for cell in row:
                    self.assertNotEqual(cell.data_type, 'f')
                    self.assertNotEqual(cell.data_type, 'e')
                    self.assertIsNone(cell.hyperlink)

    def test_existing_file_invalid_extension_and_failure_leave_no_partial_file(self):
        output = self.root / 'existing.xlsx'
        output.write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            excel.export_history_workbook(history(), output)
        self.assertEqual(output.read_bytes(), b'original')
        with self.assertRaises(ValueError):
            excel.export_history_workbook(history(), self.root / 'invalid.csv')
        with patch('openpyxl.workbook.workbook.Workbook.save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                excel.export_history_workbook(history(), self.root / 'failed.xlsx')
        self.assertEqual(list(self.root.iterdir()), [output])

    def test_limits_and_bad_numbers_reject_before_creating_output(self):
        output = self.root / 'bad.xlsx'
        for mutate in ('long', 'control', 'nan', 'count', 'date'):
            records = history()
            if mutate == 'long': records[0]['query'] = 'x' * 32768
            if mutate == 'control': records[0]['query'] = 'bad\x00text'
            if mutate == 'nan': records[0]['elapsed_ms'] = float('nan')
            if mutate == 'count': records[0]['result_count'] = 9
            if mutate == 'date': records[0]['timestamp'] = '2026-01-01'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                excel.export_history_workbook(records, output)
            self.assertFalse(output.exists())
        with patch.object(excel, 'EXCEL_MAX_ROWS', 6), self.assertRaises(ValueError):
            excel.export_history_workbook(history(), output)
        with patch.object(excel, 'EXCEL_MAX_COLUMNS', 2), self.assertRaises(ValueError):
            excel.export_history_workbook(history(), output)

    def test_empty_history_and_missing_dependency(self):
        book = self.book(excel.export_history_workbook([], self.root / 'empty.xlsx'))
        self.assertEqual(rows(book['Searches']), [])
        self.assertEqual(rows(book['Results']), [])
        with patch.dict('sys.modules', {'openpyxl': None}), self.assertRaisesRegex(ImportError, 'evaluation/requirements.txt'):
            excel.export_history_workbook(history(), self.root / 'missing.xlsx')
        self.assertFalse((self.root / 'missing.xlsx').exists())


if __name__ == '__main__':
    unittest.main()
