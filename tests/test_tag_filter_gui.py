"""Tag-filter behavior against a tiny ranked corpus, without model downloads."""
from contextlib import ExitStack, closing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from qt_runtime import prepare_qt_plugins
prepare_qt_plugins()

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

import document_query_visuals as gui
import document_database as database
import document_search_engine as engine
import document_tag_store as tags


class TagFilterGUIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.windows = []
        self.addCleanup(self.close_windows)
        # The first Finance match is sixth globally. The highest score belongs
        # to an untagged document, so selecting every tag differs from clearing.
        self.names = ['untagged.pdf', *[f'travel-{i}.pdf' for i in range(1, 5)],
                      'finance-sixth.pdf', 'finance-seventh.pdf', 'both-tags.pdf']
        self.paths = [str(self.root / name) for name in self.names]
        assignments = [[], *[['Travel'] for _ in range(4)], ['Finance'], ['Finance'],
                       ['Finance', 'Travel']]
        self.rows = [dict(file_path=path, file_name=Path(path).name,
                          **{f'tag_{i + 1}': (tags[i] if i < len(tags) else '') for i in range(3)})
                     for path, tags in zip(self.paths, assignments)]
        self.stack.enter_context(patch.object(gui, 'tag_color', return_value='#457563'))
        self.stack.enter_context(patch.object(gui, 'load_tag_definitions', return_value={'Finance': 'invoice'}))
        self.stack.enter_context(patch.object(gui, 'load_tag_rows', side_effect=lambda: list(self.rows)))
        self.tag_search = self.stack.enter_context(patch.object(
            gui, 'exact_tag_matches', side_effect=self.tag_matches))
        self.search = self.stack.enter_context(patch.object(gui, 'search_index', wraps=engine.search_index))

    def close_windows(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        self.app.processEvents()

    def tag_matches(self, tag):
        return [dict(file_path=row['file_path'], file_name=row['file_name'], score=.8)
                for row in self.rows if tag in [row[f'tag_{i}'] for i in range(1, 4)]]

    def window(self, tags=('Finance', 'Travel')):
        self.stack.enter_context(patch.object(gui, 'tag_names', return_value=list(tags)))
        window = gui.DocumentQueryWindow()
        self.windows.append(window)
        window._show_warning = Mock()
        self.model = Mock()
        self.model.encode.return_value = np.array([[1., 0.]], dtype=np.float32)
        docs = [dict(doc_id=f'doc-{i}', pdf_path=path) for i, path in enumerate(self.paths)]
        chunks = [dict(doc_id=doc['doc_id'], pdf_path=doc['pdf_path'], page=1,
                       chunk_id=i, text='Invented fixture text') for i, doc in enumerate(docs)]
        scores = np.linspace(.95, .25, len(docs))
        vectors = np.array([[score, np.sqrt(1 - score ** 2)] for score in scores], dtype=np.float32)
        window.index_data = dict(model=self.model, doc_records=docs, chunk_records=chunks,
                                 chunk_vectors=vectors, docs_by_id={d['doc_id']: d for d in docs})
        return window

    def submit(self, window, query='find the relevant passage'):
        window.query_input.setText(query)
        window.search_documents()
        window._show_warning.assert_not_called()

    def displayed_names(self, window):
        text = window.results_view.toPlainText()
        return [name for name in self.names if name in text]

    def indexed_rename_fixture(self):
        window = self.window()
        path = self.root / 'letter.docx'
        path.write_bytes(b'rename fixture')
        window.vector_store_dir = self.root / 'store'
        db = window.vector_store_dir / database.DATABASE_NAME
        docs = [dict(doc_id='doc', pdf_path=str(path), method='text', char_count=14)]
        chunks = [dict(doc_id='doc', pdf_path=str(path), page=1, chunk_id=0, text='rename fixture')]
        vectors = np.array([[1., 0.]], dtype=np.float32)
        database.save_index(docs, chunks, vectors, window.vector_store_dir, engine.MODEL_NAME)
        with closing(tags.connect_tag_db(db)) as connection:
            tags.save_file_tags(path, [{'tag': 'Finance', 'score': .8}], 14, connection)
        window.index_data.update(doc_records=docs, chunk_records=chunks, chunk_vectors=vectors,
                                 docs_by_id={'doc': docs[0]})
        return window, path, db

    def test_default_unchecked_search_includes_untagged_and_keeps_five_results(self):
        window = self.window()
        self.assertFalse(any(check.isChecked() for check in window.tag_filters.values()))
        self.submit(window)
        self.assertEqual(self.displayed_names(window), self.names[:5])
        self.assertIn('All documents', window.statusBar().currentMessage())
        self.assertEqual(self.search.call_args.kwargs['top_k'], 8)
        self.model.encode.assert_called_once()

    def test_visible_results_show_query_relevant_passage_and_distinct_scores(self):
        from test_search_snippets import ScoredModel
        window = self.window()
        target = 'The furnace requires repair before occupancy.'
        source = 'The garden has several trees. ' + target + ' Travel bookings are archived separately.'
        window.index_data['chunk_records'][0]['text'] = source
        model = ScoredModel({'repair': 1., target: .91})
        window.index_data['model'] = model
        self.submit(window, 'repair')
        displayed = window.results_view.toPlainText()
        self.assertIn(target, displayed)
        self.assertNotIn('garden has several trees', displayed)
        self.assertIn('95.0% section similarity', displayed)
        self.assertIn('Closest short passage', displayed)
        self.assertIn('91.0% similarity', displayed)
        calls = len(model.calls)
        window._display_query_results()
        self.assertEqual(len(model.calls), calls)

    def test_passage_text_is_escaped_when_rendered(self):
        window = self.window()
        result = dict(path=self.paths[0], document=self.names[0], page=1,
                      score=.7, snippet_score=.9, text='<script>alert(1)</script> & details')
        rendered = window._format_search_results_html([result])
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)

    def test_word_passage_shows_actual_page_in_place_of_passage_similarity(self):
        window = self.window()
        result = dict(path=str(self.root / 'essay.docx'), document='essay.docx', page=1,
                      match_pages=(4, 4), score=.7, snippet_score=.525, text='Found passage.')
        rendered = window._format_search_results_html([result])
        self.assertIn('Match found in Page 4', rendered)
        self.assertIn('Found passage.', rendered)
        self.assertNotIn('Closest short passage', rendered)
        self.assertNotIn('52.5% similarity', rendered)
        self.assertLess(rendered.index('Match found in Page 4'), rendered.index('Found passage.'))
        result['match_pages'] = (4, 5)
        self.assertIn('Match found in Pages 4–5', window._format_search_results_html([result]))
        result.pop('match_pages')
        rendered = window._format_search_results_html([result])
        self.assertIn('Page unavailable', rendered)
        self.assertNotIn('Match found in Page 1', rendered)

    def test_office_file_picker_and_location_labels(self):
        window = self.window()
        path = self.root / 'fixture.docx'
        path.write_bytes(b'picker fixture')
        with patch.object(gui.QFileDialog, 'getOpenFileNames', return_value=([str(path)], '')) as picker:
            window.add_files()
        self.assertIn('*.docx', picker.call_args.args[3])
        self.assertIn('*.pptx', picker.call_args.args[3])
        self.assertIn('*.xls', picker.call_args.args[3])
        self.assertIn('*.rtf', picker.call_args.args[3])
        self.assertEqual(window._selected_file_paths(), [path])
        for name, location in [('slides.pptx', 'Slide 2'), ('book.xlsx', 'Sheet 2'), ('letter.docx', 'Document')]:
            result = dict(path=str(self.root / name), document=name, page=2, score=.7, text='fixture')
            self.assertIn(location + ' · Tags:', window._format_search_results_html([result]))

    def test_rename_preserves_office_extension(self):
        window, path, db = self.indexed_rename_fixture()
        with patch.object(gui.QInputDialog, 'getText', return_value=('renamed', True)), \
                patch.object(gui.QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            window._rename_file(path)
        destination = self.root / 'renamed.docx'
        self.assertTrue(destination.is_file())
        self.assertFalse(path.exists())
        self.assertFalse((self.root / 'renamed.pdf').exists())
        docs, chunks, vectors = database.load_index(window.vector_store_dir)
        self.assertEqual(docs[0]['pdf_path'], destination.as_posix())
        self.assertEqual(chunks[0]['pdf_path'], destination.as_posix())
        self.assertEqual(window.index_data['doc_records'][0]['pdf_path'], destination.as_posix())
        np.testing.assert_array_equal(vectors, [[1., 0.]])
        row = tags.load_tag_rows(db)[0]
        self.assertEqual((row['file_path'], row['file_name'], row['tag_1']),
                         (str(destination), destination.name, 'Finance'))
        window._show_warning.assert_not_called()

    def test_rename_database_failure_restores_file_and_all_database_paths(self):
        window, path, db = self.indexed_rename_fixture()
        with closing(database.connect_database(db)) as connection, connection:
            connection.execute("CREATE TRIGGER prevent_tag_rename BEFORE UPDATE ON file_tags "
                               "BEGIN SELECT RAISE(ABORT, 'tag rename failed'); END")
        with patch.object(gui.QInputDialog, 'getText', return_value=('renamed', True)), \
                patch.object(gui.QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            window._rename_file(path)
        self.assertTrue(path.is_file())
        self.assertFalse((self.root / 'renamed.docx').exists())
        docs, chunks, _ = database.load_index(window.vector_store_dir)
        self.assertEqual(docs[0]['pdf_path'], str(path))
        self.assertEqual(chunks[0]['pdf_path'], str(path))
        self.assertEqual(tags.load_tag_rows(db)[0]['file_path'], str(path))
        self.assertIn('tag rename failed', window._show_warning.call_args.args[0])

    def test_rename_display_failure_keeps_committed_file_and_database_paths(self):
        window, path, db = self.indexed_rename_fixture()
        with patch.object(gui.QInputDialog, 'getText', return_value=('renamed', True)), \
                patch.object(gui.QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                patch.object(window, '_format_categories_html', side_effect=OSError('display failed')):
            window._rename_file(path)
        destination = self.root / 'renamed.docx'
        self.assertTrue(destination.is_file())
        self.assertFalse(path.exists())
        docs, chunks, _ = database.load_index(window.vector_store_dir)
        self.assertEqual(docs[0]['pdf_path'], destination.as_posix())
        self.assertEqual(chunks[0]['pdf_path'], destination.as_posix())
        self.assertEqual(tags.load_tag_rows(db)[0]['file_path'], str(destination))
        self.assertIsNone(window.index_data)
        self.assertIn('display could not refresh', window._show_warning.call_args.args[0])

    def test_rename_rejects_directory_separators_with_a_warning(self):
        window, path, _ = self.indexed_rename_fixture()
        for name in ('nested/renamed.docx', 'nested\\renamed.docx'):
            with self.subTest(name=name), \
                    patch.object(gui.QInputDialog, 'getText', return_value=(name, True)), \
                    patch.object(gui.QMessageBox, 'question') as confirm:
                window._show_warning.reset_mock()
                window._rename_file(path)
                confirm.assert_not_called()
                window._show_warning.assert_called_once()
                self.assertTrue(path.is_file())

    def test_clear_database_cancellation_preserves_loaded_state(self):
        window = self.window()
        original = window.index_data
        with patch.object(gui.QMessageBox, 'question', return_value=QMessageBox.StandardButton.No), \
                patch('document_database.clear_database') as clear:
            window.settings_button.menu().actions()[0].trigger()
        clear.assert_not_called()
        self.assertIs(window.index_data, original)



    def test_clear_database_resets_filters_and_cached_results_keeps_queue(self):
        window = self.window()
        self.submit(window)
        window.tag_filters['Finance'].setChecked(True)
        window.file_list.addItem('queued.pdf')
        actions = window.settings_button.menu().actions()
        action = next(action for action in actions if action.text() == 'Clear Database')
        with patch.object(gui.QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                patch('document_database.clear_database') as clear:
            action.trigger()
        clear.assert_called_once_with(window.vector_store_dir)
        self.assertIsNone(window.index_data)
        self.assertIsNone(window._last_query)
        self.assertIsNone(window._last_results)
        self.assertFalse(any(check.isChecked() for check in window.tag_filters.values()))
        self.assertEqual(window.file_list.count(), 1)
        self.assertEqual(window.query_input.text(), '')
        self.assertFalse(window.analysis_progress.isVisible())

    def test_clear_database_failure_preserves_loaded_state(self):
        window = self.window()
        original = window.index_data
        with patch.object(gui.QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                patch('document_database.clear_database', side_effect=OSError('locked')):
            window.clear_database()
        self.assertIs(window.index_data, original)
        window._show_warning.assert_called_once()

    def test_filter_before_top_five_and_or_semantics_without_model_reruns(self):
        window = self.window()
        window.tag_filters['Finance'].setChecked(True)
        self.submit(window)
        self.assertEqual(self.displayed_names(window), self.names[5:])
        window.query_input.setText('a draft that has not been submitted')
        window.tag_filters['Travel'].setChecked(True)
        # ANY selection includes the union, but does not include the untagged
        # PDF even when every available tag is checked.
        self.assertEqual(self.displayed_names(window), self.names[1:6])
        self.assertIn('Matching any of: Finance, Travel', window.tag_filter_scope.text())
        self.assertIn('find the relevant passage', window.statusBar().currentMessage())
        self.assertNotIn('a draft', window.statusBar().currentMessage())
        window.tag_filters['Finance'].setChecked(False)
        self.assertEqual(self.displayed_names(window), self.names[1:5] + ['both-tags.pdf'])
        window.tag_filters['Travel'].setChecked(False)
        self.assertEqual(self.displayed_names(window), self.names[:5])
        self.assertEqual(window.tag_filter_scope.text(), 'All documents')
        self.search.assert_called_once()
        self.model.encode.assert_called_once()

    def test_exact_tag_query_keeps_its_intent_when_filters_and_input_change(self):
        window = self.window()
        self.submit(window, 'Finance')
        self.assertEqual(self.displayed_names(window), self.names[5:])
        window.query_input.setText('Travel')
        window.tag_filters['Travel'].setChecked(True)
        self.assertEqual(self.displayed_names(window), ['both-tags.pdf'])
        self.assertIn('for "Finance"', window.statusBar().currentMessage())
        window.tag_filters['Travel'].setChecked(False)
        self.assertEqual(self.displayed_names(window), self.names[5:])
        self.assertEqual(window._last_query, 'Finance')
        self.tag_search.assert_called_once_with('Finance')
        self.search.assert_not_called()
        self.model.encode.assert_not_called()

    def test_filter_browsing_before_query_does_not_become_a_submitted_query(self):
        window = self.window()
        window.tag_filters['Finance'].setChecked(True)
        self.assertEqual(self.displayed_names(window), self.names[5:])
        window.tag_filters['Travel'].setChecked(True)
        self.assertEqual(self.displayed_names(window), self.names[1:])
        window.tag_filters['Finance'].setChecked(False)
        window.tag_filters['Travel'].setChecked(False)
        self.assertIn('Ready when you are', window.results_view.toPlainText())
        self.assertIn('All documents', window.statusBar().currentMessage())
        self.assertIsNone(window._last_query)
        self.assertIsNone(window._last_results)
        self.search.assert_not_called()

    def test_empty_new_tag_query_cannot_restore_previous_semantic_results(self):
        window = self.window(tags=('Finance', 'Travel', 'Empty'))
        self.submit(window)
        self.submit(window, 'Empty')
        self.assertEqual(self.displayed_names(window), [])
        window.tag_filters['Travel'].setChecked(True)
        window.tag_filters['Travel'].setChecked(False)
        self.assertEqual(self.displayed_names(window), [])
        self.assertEqual(window._last_query, 'Empty')
        self.assertEqual(window._last_results, [])
        self.search.assert_called_once()

    def test_removing_document_invalidates_cached_ranking_before_filter_change(self):
        window = self.window()
        self.submit(window)
        removed = Path(self.paths[0])
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
             patch.object(engine, 'remove_document_from_vector_store', return_value=True), \
             patch.object(gui, 'delete_file_tags'):
            window._remove_file_from_database(removed)
        self.rows = [row for row in self.rows if row['file_path'] != str(removed)]
        window.tag_filters['Finance'].setChecked(True)
        window.tag_filters['Finance'].setChecked(False)
        self.assertNotIn('untagged.pdf', window.results_view.toPlainText())
        self.assertIsNone(window._last_query)
        self.assertIsNone(window._last_results)
        self.assertEqual(len(window.index_data['doc_records']), 7)
        self.search.assert_called_once()

    def test_horizontal_overflow_keeps_full_labels_and_font_size(self):
        tags = ('Finance', 'Travel', *[f'Custom category number {i}' for i in range(12)],
                'A longer custom tag that must stay readable')
        window = self.window(tags)
        window.resize(1416, 1014)
        window.show()
        self.app.processEvents()
        wide_sizes = [(check.width(), check.font().pointSizeF()) for check in window.tag_filters.values()]
        window.resize(1080, 806)
        self.app.processEvents()
        scroll = window.tag_filter_scroll
        self.assertLessEqual(window.width(), 1080)
        self.assertGreater(scroll.horizontalScrollBar().maximum(), 0)
        self.assertLess(scroll.viewport().width(), scroll.widget().width())
        self.assertEqual(scroll.verticalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for check, (wide_width, font_size) in zip(window.tag_filters.values(), wide_sizes):
            self.assertEqual(check.width(), wide_width)
            self.assertGreaterEqual(check.width(), check.sizeHint().width())
            self.assertEqual(check.font().pointSizeF(), font_size)
            self.assertLessEqual(check.height(), scroll.viewport().height())
        scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().maximum())
        self.app.processEvents()
        last = window.tag_filters[tags[-1]]
        last_right = last.mapTo(scroll.viewport(), last.rect().topRight()).x()
        self.assertGreater(last_right, 0)
        self.assertLessEqual(last_right, scroll.viewport().width())


if __name__ == '__main__':
    unittest.main()
