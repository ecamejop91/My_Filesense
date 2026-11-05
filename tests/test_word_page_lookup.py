"""Actual page locations for Word passages, including older whole-document chunks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from word_page_lookup import PageMap, WordPageLocator


class WordPageTests(unittest.TestCase):
    def test_passage_in_old_whole_document_chunk_uses_actual_page(self):
        pages = ['Introduction.\r', 'The strange fits were influenced by stress.\r', 'Conclusion.\r']
        mapping = PageMap(pages)
        self.assertEqual(mapping.locate('The strange fits were influenced by stress.',
                                       '\n'.join(pages)), (2, 2))

    def test_passage_spanning_pages_preserves_both_page_numbers(self):
        mapping = PageMap(['The strange fits were ', 'influenced by stress.'])
        self.assertEqual(mapping.locate('The strange fits were influenced by stress.'), (1, 2))

    def test_whitespace_table_boundaries_and_unicode_are_matched(self):
        mapping = PageMap(['Cover\r', 'Café\tinvoice\x07due\r\nwithin thirty days.'])
        self.assertEqual(mapping.locate('Café invoice due within thirty days.'), (2, 2))

    def test_ambiguous_sentence_needs_source_context(self):
        mapping = PageMap(['First context. Shared sentence.\r', 'Second context. Shared sentence.'])
        self.assertIsNone(mapping.locate('Shared sentence.'))
        self.assertEqual(mapping.locate('Shared sentence.', 'Second context. Shared sentence.'), (2, 2))
        self.assertIsNone(mapping.locate('Text that is no longer in the document.'))

    def test_blank_pages_count_and_page_number_is_never_inferred_from_chunk(self):
        mapping = PageMap(['', '', 'Found passage.'])
        self.assertEqual(mapping.locate('Found passage.'), (3, 3))

    def test_pages_are_cached_and_reloaded_after_file_changes(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'fixture.docx'
            path.write_bytes(b'first version')
            reader = Mock(side_effect=[{path: ['Cover.\r', 'Found passage.']},
                                       {path: ['Found passage.']}])
            locator = WordPageLocator(reader=reader)
            row = dict(path=str(path), page=1, text='Found passage.')
            locator.enrich([row])
            locator.enrich([row])
            reader.assert_called_once_with([path])
            self.assertEqual(row['match_pages'], (2, 2))
            path.write_bytes(b'longer revised document')
            locator.enrich([row])
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(row['match_pages'], (1, 1))

    def test_unavailable_pages_leave_results_usable_without_a_guessed_page(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'fixture.docx'
            path.touch()
            reader = Mock(return_value={path: None})
            locator = WordPageLocator(reader=reader)
            row = dict(path=str(path), page=1, text='Found passage.')
            locator.enrich([row])
            locator.enrich([row])
            reader.assert_called_once()
            self.assertNotIn('match_pages', row)

    def test_other_document_types_do_not_open_word(self):
        reader = Mock()
        locator = WordPageLocator(reader=reader)
        locator.enrich([dict(path='fixture.pdf', text='Found passage.')])
        reader.assert_not_called()
