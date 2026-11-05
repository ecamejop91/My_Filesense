"""Hand-calculated fixtures; no application or third-party imports required."""
from math import log2
import unittest

from evaluation.retrieval import precision, recall, precision_at_k, recall_at_k
from evaluation.ranking import reciprocal_rank, mean_reciprocal_rank, dcg_at_k, ndcg_at_k


class RetrievalMetricTests(unittest.TestCase):
    def test_hand_calculated_unranked(self):
        self.assertEqual(precision(['a', 'b', 'x'], {'a', 'b', 'c', 'd'}), 2 / 3)
        self.assertEqual(recall(['a', 'b', 'x'], {'a', 'b', 'c', 'd'}), 1 / 2)

    def test_cutoffs_and_duplicates(self):
        ranking = ['x', 'a', 'a', 'b', 'c']
        relevant = {'a', 'b', 'c', 'd'}
        self.assertEqual(precision_at_k(ranking, relevant, 3), 2 / 3)
        self.assertEqual(recall_at_k(ranking, relevant, 3), 1 / 2)
        self.assertEqual(precision(['a', 'a', 'x'], {'a'}), 1 / 2)

    def test_short_results_always_divide_by_k(self):
        self.assertEqual(precision_at_k(['a'], {'a'}, 5), 1 / 5)
        self.assertEqual(recall_at_k(['a'], {'a'}, 5), 1)

    def test_empty_and_no_relevance(self):
        self.assertEqual(precision([], {'a'}), 0)
        self.assertEqual(recall([], {'a'}), 0)
        self.assertEqual(precision(['a'], set()), 0)
        self.assertIsNone(recall(['a'], set()))
        self.assertIsNone(recall_at_k([], set(), 3))

    def test_invalid_k(self):
        for k in (0, -1, True, 1.5, '3'):
            for fn in (precision_at_k, recall_at_k, dcg_at_k, ndcg_at_k):
                with self.subTest(fn=fn.__name__, k=k), self.assertRaises(ValueError):
                    fn([], {}, k)


class RankingMetricTests(unittest.TestCase):
    def test_rr_and_mrr_hand_calculated(self):
        self.assertEqual(reciprocal_rank(['x', 'a'], {'a'}), .5)
        self.assertEqual(reciprocal_rank(['x'], {'a'}), 0)
        self.assertIsNone(reciprocal_rank([], set()))
        # (1/2 + 1/3 + 0) / 3 = 5/18; the fourth query has no relevant docs.
        self.assertAlmostEqual(mean_reciprocal_rank(
            [['x', 'a'], ['x', 'y', 'a'], [], ['x']], [{'a'}, {'a'}, {'a'}, set()]), 5 / 18)
        self.assertIsNone(mean_reciprocal_rank([], []))
        with self.assertRaises(ValueError):
            mean_reciprocal_rank([[]], [])

    def test_graded_dcg_ndcg_hand_calculated(self):
        grades = {'a': 3, 'b': 2, 'c': 1, 'x': 0}
        # [b, x, a] gains [3, 0, 7]; discounts [1, log2(3), 2].
        self.assertEqual(dcg_at_k(['b', 'x', 'a'], grades, 3), 6.5)
        ideal = 7 + 3 / log2(3) + 1 / 2
        self.assertAlmostEqual(ndcg_at_k(['b', 'x', 'a'], grades, 3), 6.5 / ideal)
        self.assertEqual(ndcg_at_k(['a', 'b', 'c'], grades, 3), 1)
        self.assertEqual(ndcg_at_k(['x'], grades, 3), 0)

    def test_idcg_uses_all_judgments_not_only_retrieved(self):
        self.assertEqual(ndcg_at_k(['c'], {'a': 3, 'c': 1}, 1), 1 / 7)
        self.assertEqual(dcg_at_k(['a', 'a', 'c'], {'a': 3, 'c': 1}, 2), 7 + 1 / log2(3))

    def test_zero_idcg_and_empty_ranking(self):
        self.assertIsNone(ndcg_at_k(['x'], {'x': 0}, 5))
        self.assertIsNone(ndcg_at_k([], {}, 5))
        self.assertEqual(ndcg_at_k([], {'a': 1}, 5), 0)
        self.assertEqual(dcg_at_k([], {}, 5), 0)

    def test_invalid_grades(self):
        for grade in (-1, 4, 1.5, True, float('nan'), '1'):
            with self.subTest(grade=grade), self.assertRaises(ValueError):
                ndcg_at_k(['a'], {'a': grade}, 1)


if __name__ == '__main__':
    unittest.main()
