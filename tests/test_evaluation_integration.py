"""Synthetic PDFs, real SQLite/search API/TF-IDF; deterministic encoder, no downloads."""
import csv
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import document_database as database
import document_search_engine as engine
from evaluation.__main__ import main
from evaluation.backends import FileSenseSearch, TfidfSearch
from evaluation.benchmark import Benchmark, Query, bind_queries, load_benchmark
from evaluation.corpus import Corpus, deduplicate_hits, document_id, load_corpus, rank_scores
from evaluation.demo import create_demo
from evaluation.runner import evaluate, export_results


class Model:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            text = text.lower()
            category = 0 if 'invoice' in text or 'pay' in text else 1 if 'hotel' in text or 'accommodation' in text else 2
            vectors.append(np.eye(3)[category])
        return np.asarray(vectors, dtype=np.float32)


class Tokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return ' '.join(tokens)


class EvaluationIntegrationTests(unittest.TestCase):
    def test_report_csv_treats_formula_like_identifiers_as_text(self):
        rows = [dict(system='filesense', query_id=value, document_id=value + '.pdf', score=-0.5)
                for value in ('=1+1', '+SUM(1,2)', '-1+1', '@SUM(1,2)', '  =1+1', '\tname', 'ordinary')]
        result = dict(metadata={}, aggregate=rows, per_query=rows, rankings=rows)
        output = self.root / 'csv-safety'
        export_results(result, output)
        self.assertEqual(json.loads((output / 'results.json').read_text()), result)
        for name in ('aggregate', 'per_query', 'rankings'):
            with (output / f'{name}.csv').open(newline='') as handle:
                exported = list(csv.DictReader(handle))
            for original, row in zip(rows, exported):
                prefix = '' if original['query_id'] == 'ordinary' else "'"
                self.assertEqual(row['query_id'], prefix + original['query_id'])
                self.assertEqual(row['document_id'], prefix + original['document_id'])
                self.assertEqual(float(row['score']), -0.5)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pdfs = self.root / 'pdfs'
        self.pdfs.mkdir()
        self.docs = [dict(doc_id=f'doc_{i}', pdf_path=str(self.pdfs / name), method='text', char_count=20)
                     for i, name in enumerate(['a/report.pdf', 'b/report.pdf', 'empty.pdf'])]
        self.docs[2].update(method='none', char_count=0)
        self.chunks = [dict(doc_id=self.docs[i]['doc_id'], pdf_path=self.docs[i]['pdf_path'],
                            page=1, chunk_id=j, text=text)
                       for i, j, text in [(0, 0, 'invoice payment'), (0, 1, 'invoice chairs'), (1, 0, 'hotel accommodation')]]
        self.vectors = np.array([[1, 0, 0], [.8, .6, 0], [0, 1, 0]], dtype=np.float32)
        self.folder = self.root / 'index'
        database.save_index(self.docs, self.chunks, self.vectors, self.folder, 'fixture-model')
        self.db = self.folder / database.DATABASE_NAME
        self.corpus = load_corpus(self.db, self.pdfs, self.root)
        self.benchmark = Benchmark('fixture-v1', (
            Query('q1', 'invoice', 'test', {'a/report.pdf': 3, 'b/report.pdf': 0, 'empty.pdf': 0}),
            Query('q2', 'hotel', 'test', {'a/report.pdf': 1, 'b/report.pdf': 3, 'empty.pdf': 1}),
            Query('q3', 'cake', 'test', {'a/report.pdf': 0, 'b/report.pdf': 0, 'empty.pdf': 0})))

    def test_real_search_requests_all_chunks_then_collapses_max(self):
        with patch.object(engine, 'search_index', wraps=engine.search_index) as search:
            results = FileSenseSearch(self.corpus, model=Model()).search('invoice')
        self.assertEqual(search.call_args.kwargs['top_k'], 3)
        self.assertEqual([r['document_id'] for r in results], ['a/report.pdf', 'b/report.pdf'])
        self.assertEqual(results[0]['score'], 1)

    def test_dedup_max_negative_scores_and_ties(self):
        hits = [dict(path=self.docs[0]['pdf_path'], score=-.9),
                dict(path=self.docs[1]['pdf_path'], score=-.2),
                dict(path=self.docs[0]['pdf_path'], score=-.2)]
        ranked = deduplicate_hits(hits, self.corpus)
        self.assertEqual([x['document_id'] for x in ranked], ['a/report.pdf', 'b/report.pdf'])
        self.assertEqual(ranked[0]['score'], -.2)
        with self.assertRaises(ValueError):
            deduplicate_hits([dict(path=self.docs[0]['pdf_path'], score=float('nan'))], self.corpus)

    def test_path_aliases_normalization_and_same_basenames(self):
        self.assertEqual(self.corpus.resolve_id('doc_0'), 'a/report.pdf')
        self.assertEqual(self.corpus.resolve_id('a\\report.pdf'), 'a/report.pdf')
        self.assertEqual(self.corpus.resolve_id('./a/../b/report.pdf'), 'b/report.pdf')
        self.assertEqual(self.corpus.resolve_id(self.docs[0]['pdf_path']), 'a/report.pdf')
        for raw in ('report.pdf', '../outside.pdf', 'missing.pdf'):
            with self.assertRaises(ValueError):
                self.corpus.resolve_id(raw)
        self.assertEqual(document_id('Case.PDF', self.pdfs), 'Case.PDF')

    def test_read_only_snapshot_and_reject_missing_database(self):
        before = self.db.read_bytes()
        load_corpus(self.db, self.pdfs, self.root)
        self.assertEqual(before, self.db.read_bytes())
        missing = self.root / 'missing.sqlite3'
        with self.assertRaises(FileNotFoundError):
            load_corpus(missing, self.pdfs, self.root)
        self.assertFalse(missing.exists())

    def test_corrupt_vectors_and_orphans_rejected(self):
        for vectors in (np.ones((1, 3)), np.full((3, 3), np.nan), np.ones((3, 3))):
            with self.assertRaises(ValueError):
                Corpus.from_records(self.docs, self.chunks, vectors, 'fixture', self.pdfs, self.root)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('UPDATE chunks SET embedding=? WHERE position=0', (b'bad',))
        with self.assertRaises(ValueError):
            load_corpus(self.db, self.pdfs, self.root)

    def test_corpus_fingerprint_is_portable_and_content_sensitive(self):
        moved_docs = [{**d, 'pdf_path': d['pdf_path'].replace(str(self.pdfs), '/relocated')} for d in self.docs]
        moved_chunks = [{**c, 'pdf_path': c['pdf_path'].replace(str(self.pdfs), '/relocated')} for c in self.chunks]
        moved = Corpus.from_records(moved_docs, moved_chunks, self.vectors, 'fixture-model', '/relocated', '/')
        self.assertEqual(moved.fingerprint, self.corpus.fingerprint)
        moved_chunks[0]['text'] = 'changed'
        changed = Corpus.from_records(moved_docs, moved_chunks, self.vectors, 'fixture-model', '/relocated', '/')
        self.assertNotEqual(changed.fingerprint, moved.fingerprint)

    def test_tfidf_corpus_only_and_oov_query(self):
        backend = TfidfSearch(self.corpus)
        self.assertEqual(backend.search('hotel')[0]['document_id'], 'b/report.pdf')
        before = backend.vectorizer.idf_.copy()
        results = backend.search('neverincorpus')
        self.assertNotIn('neverincorpus', backend.vectorizer.vocabulary_)
        np.testing.assert_array_equal(before, backend.vectorizer.idf_)
        self.assertEqual([r['document_id'] for r in results], ['a/report.pdf', 'b/report.pdf'])
        self.assertTrue(all(r['score'] == 0 for r in results))
        # idf for hotel: ln((1+3)/(1+1))+1; empty PDF participates in N.
        self.assertAlmostEqual(backend.vectorizer.idf_[backend.vectorizer.vocabulary_['hotel']], np.log(2) + 1)

    def test_empty_vocabulary_returns_tied_searchable_documents(self):
        corpus = Corpus.from_records(self.docs, [{**c, 'text': 'a !'} for c in self.chunks],
                                     self.vectors, 'fixture', self.pdfs, self.root)
        self.assertEqual([r['score'] for r in TfidfSearch(corpus).search('a')], [0, 0])

    def test_zero_chunks_work_for_both_backends(self):
        corpus = Corpus.from_records(self.docs, [], np.empty((0, 3)), 'fixture', self.pdfs, self.root)
        self.assertEqual(FileSenseSearch(corpus, model=Model()).search('invoice'), [])
        self.assertEqual(TfidfSearch(corpus).search('invoice'), [])

    def test_partial_judgments_and_alias_collisions_fail(self):
        benchmark = Benchmark('partial', (Query('q', 'x', 'test', {'doc_0': 1}),))
        with self.assertRaises(ValueError):
            bind_queries(benchmark, self.corpus, 'test')
        self.assertEqual(bind_queries(benchmark, self.corpus, 'test', True)[0].judgments, {'a/report.pdf': 1})
        bad = Benchmark('aliases', (Query('q', 'x', 'test', {'doc_0': 1, 'a/report.pdf': 2}),))
        with self.assertRaises(ValueError):
            bind_queries(bad, self.corpus, 'test', True)
        with self.assertRaises(ValueError):
            bind_queries(self.benchmark, self.corpus, 'dev')

    def test_macro_pooled_and_no_relevance_counts(self):
        result = evaluate(self.corpus, self.benchmark,
                          {'filesense': FileSenseSearch(self.corpus, model=Model())}, depth=1, cutoffs=[1])
        summary = result['aggregate'][0]
        # q1: 1/1, q2: 1/3, q3: undefined; pooled recall=(1+1+0)/(1+3+0).
        self.assertAlmostEqual(summary['macro_recall'], 2 / 3)
        self.assertEqual(summary['pooled_recall'], .5)
        self.assertAlmostEqual(summary['macro_precision'], 2 / 3)
        self.assertEqual(summary['mrr_at_depth'], 1)
        self.assertEqual(summary['zero_relevant_queries'], 1)
        self.assertEqual(summary['eligible_recall_rr_ndcg_queries'], 2)
        self.assertIsNone(result['per_query'][2]['ndcg@1'])
        with self.assertRaises(ValueError):
            evaluate(self.corpus, self.benchmark, {'x': TfidfSearch(self.corpus)}, depth=1, cutoffs=[2])

    def test_pooled_precision_weights_result_counts(self):
        class ShortResults:
            def search(self, query):
                return [{'document_id': 'a/report.pdf', 'score': 1.0}] if query == 'invoice' else []

        summary = evaluate(self.corpus, self.benchmark, {'short': ShortResults()},
                           depth=2, cutoffs=[2])['aggregate'][0]
        self.assertAlmostEqual(summary['macro_precision'], 1 / 3)
        self.assertEqual(summary['pooled_precision'], 1)
        self.assertEqual(summary['macro_recall'], .5)
        self.assertEqual(summary['pooled_recall'], .25)
        self.assertEqual(summary['macro_precision@2'], summary['pooled_precision@2'])

    def test_all_negative_empty_results_export_nulls_and_headers(self):
        corpus = Corpus.from_records(self.docs, [], np.empty((0, 3)), 'fixture', self.pdfs, self.root)
        benchmark = Benchmark('negative-only', (self.benchmark.queries[2],))
        result = evaluate(corpus, benchmark, {'tfidf': TfidfSearch(corpus)}, depth=1, cutoffs=[1])
        summary = result['aggregate'][0]
        self.assertIsNone(summary['mrr_at_depth'])
        self.assertIsNone(summary['macro_ndcg@1'])
        self.assertIsNone(summary['pooled_recall'])
        self.assertEqual(summary['pooled_precision'], 0)
        output = self.root / 'empty-results'
        export_results(result, output)
        with (output / 'rankings.csv').open() as handle:
            reader = csv.DictReader(handle)
            self.assertIn('document_id', reader.fieldnames)
            self.assertEqual(list(reader), [])

    def test_duplicate_corpus_aliases_and_unknown_results_fail(self):
        with self.assertRaises(ValueError):
            Corpus.from_records(self.docs + [self.docs[0]], self.chunks, self.vectors,
                                'fixture', self.pdfs, self.root)
        with self.assertRaises(ValueError):
            deduplicate_hits([{'path': str(self.pdfs / 'unindexed.pdf'), 'score': 1}], self.corpus)
        with self.assertRaises(ValueError):
            FileSenseSearch(self.corpus, model=Model(), search_fn=lambda *a, **kw: []).search('invoice')

    def test_exports_and_cli_tfidf_do_not_modify_database(self):
        path = self.root / 'benchmark.json'
        data = dict(schema_version=1, benchmark_id=self.benchmark.benchmark_id, queries=[
            dict(query_id=q.query_id, text=q.text, split=q.split, judgments=q.judgments) for q in self.benchmark.queries])
        path.write_text(json.dumps(data))
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        output = self.root / 'results'
        main(['run', '--database', str(self.db), '--corpus-root', str(self.pdfs), '--benchmark', str(path),
              '--systems', 'tfidf', '--depth', '2', '--k', '1', '2', '--output', str(output)])
        result = json.loads((output / 'results.json').read_text())
        self.assertEqual(result['metadata']['packages']['scikit-learn'], __import__('sklearn').__version__)
        self.assertEqual(len(result['per_query']), 3)
        self.assertEqual(before, hashlib.sha256(self.db.read_bytes()).hexdigest())
        self.assertNotIn(str(self.pdfs), (output / 'results.json').read_text())
        self.assertNotIn('invoice payment', (output / 'results.json').read_text())
        with (output / 'per_query.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[2]['recall'], '')
        with self.assertRaises(FileExistsError):
            export_results(result, output)

    def test_pdf_build_sqlite_benchmark_to_both_systems(self):
        demo = self.root / 'demo'
        create_demo(demo)
        with patch.object(engine, 'SentenceTransformer', return_value=Model()), \
             patch.object(engine.tiktoken, 'get_encoding', return_value=Tokenizer()):
            main(['index', '--corpus-root', str(demo / 'pdfs'), '--output', str(demo / 'index'),
                  '--model', 'fixture-model'])
            main(['run', '--database', str(demo / 'index/file_tags.sqlite3'), '--corpus-root', str(demo / 'pdfs'),
                  '--benchmark', str(demo / 'benchmark.json'), '--depth', '3', '--k', '1', '3',
                  '--output', str(demo / 'results')])
        result = json.loads((demo / 'results/results.json').read_text())
        self.assertEqual({r['system'] for r in result['aggregate']}, {'filesense', 'tfidf'})
        self.assertEqual(result['metadata']['index_build']['chunk_size'], engine.CHUNK_SIZE)
        self.assertEqual(len(result['per_query']), 8)


class BenchmarkValidationTests(unittest.TestCase):
    def test_examples_json_and_csv_equivalent(self):
        folder = Path(__file__).resolve().parents[1] / 'evaluation/examples'
        self.assertEqual(load_benchmark(folder / 'benchmark.json'), load_benchmark(folder / 'benchmark.csv'))

    def test_invalid_json_schemas(self):
        valid = dict(schema_version=1, benchmark_id='test', queries=[
            dict(query_id='q', text='query', split='test', judgments={'a.pdf': 1})])
        invalid = [[], dict(valid, queries=[None]), dict(valid, schema_version=True), dict(valid, schema_version=2), dict(valid, queries=[]),
                   dict(valid, queries=valid['queries'] * 2)]
        for alteration in ({'text': ''}, {'split': 'train'}, {'judgments': {'a.pdf': -1}}, {'judgments': []}):
            invalid.append(dict(valid, queries=[dict(valid['queries'][0], **alteration)]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'benchmark.json'
            for data in invalid:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    load_benchmark(path)
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(ValueError):
                load_benchmark(path)

    def test_duplicate_conflicting_csv_rejected(self):
        header = 'schema_version,benchmark_id,query_id,query,split,document_id,relevance\n'
        row = '1,test,q,invoice,test,a.pdf,1\n'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'benchmark.csv'
            for text in (header + row * 2, header + row + row.replace('invoice', 'hotel').replace('a.pdf', 'b.pdf'),
                         header + row.replace(',1\n', ',1.5\n'), header + row.replace(',1\n', ',1,extra\n')):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_benchmark(path)


if __name__ == '__main__':
    unittest.main()
