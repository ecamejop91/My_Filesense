"""Query-aware passages: relevance, brevity, source fidelity and lazy caching."""
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

import document_search_engine as engine
from search_snippets import MAX_CANDIDATES, MAX_PASSAGE_CHARS, PassageSelector, passage_candidates


class ScoredModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.array([[score, np.sqrt(1 - score ** 2)]
                         for score in (self.scores.get(text, .1) for text in texts)], dtype=np.float32)


def row(text, **extra):
    return dict(text=text[:700], _source_text=text, score=.65, rank=1,
                path='fixture.pdf', page=2, chunk_id=3,
                _query_vector=np.array([1., 0.], dtype=np.float32), **extra)


class PassageSelectionTests(unittest.TestCase):
    def test_selects_relevant_sentence_after_long_unrelated_intro(self):
        intro = 'The archive contains travel itineraries and hotel reservations. '
        target = 'The furnace must be repaired before the property is occupied.'
        source = intro * 14 + target + ' The garden contains several mature trees.'
        result = row(source)
        model = ScoredModel({target: .93})
        PassageSelector().select('furnace repairs', [result], model)
        self.assertEqual(result['text'], target)
        self.assertAlmostEqual(result['snippet_score'], .93, places=5)
        self.assertEqual((result['rank'], result['score'], result['page'], result['chunk_id']), (1, .65, 2, 3))
        self.assertEqual(result['_source_text'], source)
        self.assertEqual(len(model.calls), 1)

    def test_prefers_shortest_passage_near_best_score(self):
        short = 'The furnace requires repair.'
        context = 'An inspection must be completed before occupancy.'
        model = ScoredModel({short: .925, short + ' ' + context: .93})
        result = row(short + ' ' + context)
        PassageSelector().select('furnace repair', [result], model)
        self.assertEqual(result['text'], short)
        self.assertAlmostEqual(result['snippet_score'], .925, places=5)

    def test_keeps_adjacent_context_when_it_scores_materially_better(self):
        one = 'The alarm was inspected.'
        two = 'Its batteries must not be removed.'
        both = one + ' ' + two
        model = ScoredModel({one: .70, two: .75, both: .91})
        result = row(both)
        PassageSelector().select('can alarm batteries be removed', [result], model)
        self.assertEqual(result['text'], both)

    def test_already_short_chunk_needs_no_extra_model_call(self):
        result = row('Invoice 4729 remains unpaid.')
        model = Mock()
        PassageSelector().select('unpaid invoices', [result], model)
        model.encode.assert_not_called()
        self.assertEqual(result['snippet_score'], result['score'])

    def test_filter_redisplay_and_new_query_reuse_passage_embeddings(self):
        source = 'The furnace requires repair. The garden needs watering.'
        selector = PassageSelector()
        model = ScoredModel({'The furnace requires repair.': .9})
        first = row(source)
        selector.select('repair', [first], model)
        selector.select('repair', [first], model)
        selector.select('heating maintenance', [row(source)], model)
        self.assertEqual(len(model.calls), 1)

    def test_unshown_rows_are_not_embedded_and_cache_is_bounded(self):
        selector = PassageSelector(cache_size=1)
        model = ScoredModel({})
        first = row('An unpaid invoice is overdue. A hotel booking is cancelled.')
        hidden = row('Hidden document sentence one. Hidden document sentence two.')
        selector.select('invoice', [first], model)
        self.assertFalse(any('Hidden' in text for batch in model.calls for text in batch))
        self.assertNotIn('snippet_score', hidden)
        selector.select('query', [row('Another document has two sentences. This is its second sentence.')], model)
        self.assertEqual(len(selector._cache), 1)

    def test_model_change_does_not_reuse_other_model_vectors(self):
        source = 'The furnace requires repair. The garden needs watering.'
        selector = PassageSelector()
        first_model = ScoredModel({'The furnace requires repair.': .9})
        second_model = ScoredModel({'The garden needs watering.': .95})
        result = row(source)
        selector.select('query', [result], first_model)
        selector.select('query', [result], second_model)
        self.assertEqual(result['text'], 'The garden needs watering.')
        self.assertEqual(len(second_model.calls), 1)

    def test_long_unpunctuated_ocr_has_bounded_verbatim_candidates_at_end(self):
        source = ' '.join(f'word{i}' for i in range(180))
        candidates = passage_candidates(source)
        self.assertLessEqual(len(candidates), MAX_CANDIDATES)
        self.assertTrue(all(len(part) <= MAX_PASSAGE_CHARS and part in source for part in candidates))
        self.assertTrue(any(part.endswith('word179') for part in candidates))

    def test_decimal_amounts_wrapped_lines_and_unspaced_text(self):
        source = 'Invoice total is 125.50 dollars.\nPayment is due\nwithin thirty days.'
        candidates = passage_candidates(source)
        self.assertIn('Invoice total is 125.50 dollars.', candidates)
        self.assertIn('Payment is due within thirty days.', candidates)
        long_unspaced = '住房维修检查记录' * 120
        parts = passage_candidates(long_unspaced)
        self.assertTrue(all(part in long_unspaced and len(part) <= MAX_PASSAGE_CHARS for part in parts))

    def test_retrieval_retains_full_source_and_shared_query_for_later_selection(self):
        model = ScoredModel({'repair': 1.})
        text = 'intro ' * 180 + 'The furnace requires repair.'
        docs = {'doc': dict(pdf_path=str(Path('fixture.pdf').resolve()))}
        chunks = [dict(doc_id='doc', text=text, page=2, chunk_id=0)]
        results = engine.search_index('repair', model, chunks, np.array([[1., 0.]]), docs)
        self.assertEqual(results[0]['_source_text'], text)
        self.assertEqual(len(results[0]['text']), 700)
        self.assertEqual(len(model.calls), 1)
        np.testing.assert_array_equal(results[0]['_query_vector'], [1., 0.])
