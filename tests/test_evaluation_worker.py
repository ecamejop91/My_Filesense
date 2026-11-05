"""Background worker contract using invented data and a deterministic local encoder."""
from contextlib import redirect_stdout, closing
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import document_database as database
import document_search_engine as engine
from evaluation import gui_worker as worker
from evaluation.corpus import load_corpus


class FixtureModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            text = text.lower()
            category = 0 if 'invoice' in text or 'pay' in text else 1 if 'hotel' in text or 'accommodation' in text else 2
            vectors.append(np.eye(3)[category])
        return np.asarray(vectors, dtype=np.float32)


class FixtureTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return ' '.join(tokens)


def events(output):
    return [json.loads(line[len(worker.EVENT_PREFIX):])
            for line in output.splitlines() if line.startswith(worker.EVENT_PREFIX)]


class EvaluationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pdfs = self.root / 'pdfs'
        self.pdfs.mkdir()
        self.docs = [dict(doc_id=f'doc_{i}', pdf_path=str(self.pdfs / name),
                          method='text', char_count=20)
                     for i, name in enumerate(('invoice.pdf', 'hotel.pdf'))]
        self.chunks = [dict(doc_id=d['doc_id'], pdf_path=d['pdf_path'], page=1,
                            chunk_id=0, text=text)
                       for d, text in zip(self.docs, ('invoice payment', 'hotel accommodation'))]
        self.index_dir = self.root / 'source-index'
        database.save_index(self.docs, self.chunks, np.eye(3, dtype=np.float32)[:2],
                            self.index_dir, 'fixture-model')
        self.db = self.index_dir / database.DATABASE_NAME
        self.benchmark = self.root / 'labels.json'
        self.data = dict(schema_version=1, benchmark_id='fixture-v1', queries=[
            dict(query_id='test-payment', text='invoice', split='test',
                 judgments={'invoice.pdf': 3, 'hotel.pdf': 0}),
            dict(query_id='dev-hotel', text='hotel', split='dev',
                 judgments={'invoice.pdf': 0, 'hotel.pdf': 3})])
        self.write_labels()

    def write_labels(self):
        self.benchmark.write_text(json.dumps(self.data), encoding='utf-8')

    def arguments(self, name='job', *extra):
        return ['--mode', 'benchmark', '--job-dir', str(self.root / name),
                '--benchmark', str(self.benchmark), '--database', str(self.db),
                '--corpus-root', str(self.pdfs), '--systems', 'tfidf', '--depth', '2', *extra]

    def invoke(self, args):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = worker.main(args)
        return code, events(stream.getvalue())

    def test_custom_report_snapshot_and_inputs_remain_unchanged(self):
        before = {p: p.read_bytes() for p in (self.db, self.benchmark)}
        with patch.object(engine, 'SentenceTransformer', side_effect=AssertionError('No model needed')):
            code, messages = self.invoke(self.arguments())
        self.assertEqual(code, 0, messages)
        self.assertTrue(any(e['event'] == 'status' for e in messages))
        complete = messages[-1]
        self.assertEqual(complete['event'], 'complete')
        self.assertEqual(Path(complete['report_dir']), self.root / 'job/report')
        report_path = Path(complete['result_path'])
        result = json.loads(report_path.read_text())
        metadata = result['metadata']
        self.assertFalse(metadata['synthetic_demo'])
        self.assertEqual(metadata['mode'], 'benchmark')
        self.assertEqual(metadata['cutoffs'], [1, 2])
        self.assertEqual(metadata['unjudged_policy'], 'error')
        self.assertEqual(metadata['setup_seconds'].keys(), {'tfidf'})
        self.assertIn('not definitive', metadata['timing_note'])
        self.assertIn('unknown', metadata['index_build']['chunk_size'])
        self.assertEqual(result['aggregate'][0]['macro_ndcg@1'], 1)
        self.assertEqual((self.root / 'job/benchmark.json').read_bytes(), before[self.benchmark])
        snapshot = self.root / 'job/index/file_tags.sqlite3'
        self.assertEqual(metadata['index_snapshot_sha256'], hashlib.sha256(snapshot.read_bytes()).hexdigest())
        self.assertNotIn(str(self.pdfs), report_path.read_text())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        for filename in ('aggregate.csv', 'per_query.csv', 'rankings.csv'):
            self.assertTrue((report_path.parent / filename).is_file())
        # Later source changes do not alter the run's frozen collection.
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE chunks SET text='changed' WHERE position=0")
        frozen = load_corpus(snapshot, self.pdfs, worker.SOURCE_BASE)
        self.assertEqual(frozen.documents['invoice.pdf']['text'], 'invoice payment')

    def test_readonly_snapshot_includes_committed_wal(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute("UPDATE chunks SET text='invoice wal text' WHERE position=0")
            connection.commit()
            code, messages = self.invoke(self.arguments())
            self.assertEqual(code, 0, messages)
            frozen = load_corpus(self.root / 'job/index/file_tags.sqlite3', self.pdfs, worker.SOURCE_BASE)
            self.assertEqual(frozen.documents['invoice.pdf']['text'], 'invoice wal text')

    def test_demo_uses_real_engine_cached_model_metrics_and_no_ocr(self):
        with patch.object(engine, 'SentenceTransformer', return_value=FixtureModel()) as model, \
             patch.object(engine.tiktoken, 'get_encoding', return_value=FixtureTokenizer()), \
             patch.object(engine, 'build_index', wraps=engine.build_index) as build, \
             patch.object(engine, 'search_index', wraps=engine.search_index) as search, \
             patch.object(engine, 'extract_page_texts_with_ocr', side_effect=AssertionError('No OCR')):
            code, messages = self.invoke(['--mode', 'demo', '--job-dir', str(self.root / 'demo-job'),
                                          '--depth', '3'])
        self.assertEqual(code, 0, messages)
        build.assert_called_once()
        self.assertEqual(model.call_count, 1)
        self.assertTrue(model.call_args.kwargs['local_files_only'])
        self.assertEqual(search.call_count, 4)
        result = json.loads(Path(messages[-1]['result_path']).read_text())
        metadata = result['metadata']
        self.assertTrue(metadata['synthetic_demo'])
        self.assertTrue(metadata['offline'])
        self.assertEqual(metadata['documents'], 3)
        self.assertEqual(metadata['cutoffs'], [1, 3])
        self.assertIn('do not establish', metadata['interpretation_note'])
        self.assertIn('OCR disabled', metadata['index_build']['extraction'])
        self.assertEqual(metadata['index_build']['chunk_size'], engine.CHUNK_SIZE)
        self.assertEqual({r['system'] for r in result['aggregate']}, {'filesense', 'tfidf'})
        self.assertEqual(len(result['per_query']), 8)
        self.assertTrue((self.root / 'demo-job/benchmark.json').exists())

    def test_custom_both_systems_and_dev_split(self):
        with patch.object(engine, 'SentenceTransformer', return_value=FixtureModel()) as model:
            code, messages = self.invoke(self.arguments('dev-job', '--systems', 'both', '--split', 'dev'))
        self.assertEqual(code, 0, messages)
        self.assertTrue(model.call_args.kwargs['local_files_only'])
        result = json.loads(Path(messages[-1]['result_path']).read_text())
        self.assertEqual({row['query_id'] for row in result['per_query']}, {'dev-hotel'})
        self.assertEqual(len(result['aggregate']), 2)

    def test_source_sidecar_preserved_only_when_fingerprint_matches(self):
        corpus = load_corpus(self.db, self.pdfs, worker.SOURCE_BASE)
        settings = dict(corpus_fingerprint=corpus.fingerprint, chunk_size=17, model_revision='fixture')
        sidecar = self.index_dir / 'evaluation-index.json'
        sidecar.write_text(json.dumps(settings))
        code, messages = self.invoke(self.arguments())
        self.assertEqual(code, 0, messages)
        result = json.loads(Path(messages[-1]['result_path']).read_text())
        self.assertEqual(result['metadata']['index_build'], settings)
        self.assertEqual(json.loads((self.root / 'job/index/evaluation-index.json').read_text()), settings)
        sidecar.write_text(json.dumps(dict(settings, corpus_fingerprint='wrong')))
        code, messages = self.invoke(self.arguments('bad-sidecar'))
        self.assertEqual(code, 1)
        self.assertIn('do not match', messages[-1]['message'])
        self.assertFalse((self.root / 'bad-sidecar/report').exists())

    def test_invalid_labels_fail_before_backend_creation(self):
        for label, judgments in [('missing', {'invoice.pdf': 3}),
                                  ('unknown', {'invoice.pdf': 3, 'hotel.pdf': 0, 'absent.pdf': 1}),
                                  ('grade', {'invoice.pdf': 4, 'hotel.pdf': 0})]:
            self.data['queries'][0]['judgments'] = judgments
            self.write_labels()
            with self.subTest(label=label), patch('evaluation.backends.TfidfSearch',
                                                side_effect=AssertionError('Validation must precede setup')):
                code, messages = self.invoke(self.arguments(label))
                self.assertEqual(code, 1)
                self.assertEqual(messages[-1]['event'], 'error')
                self.assertNotIn('--allow-unjudged', messages[-1]['message'])
                self.assertFalse((self.root / label / 'report').exists())

    def test_existing_job_is_refused_and_untouched(self):
        job = self.root / 'job'
        job.mkdir()
        marker = job / 'keep.txt'
        marker.write_text('keep')
        code, messages = self.invoke(self.arguments())
        self.assertEqual(code, 1)
        self.assertIn('already exists', messages[-1]['message'])
        self.assertEqual(marker.read_text(), 'keep')
        self.assertEqual(list(job.iterdir()), [marker])

    def test_invalid_request_produces_structured_error_without_creating_job(self):
        for arguments in [self.arguments('missing-db', '--database', str(self.root / 'missing.sqlite3')),
                          self.arguments('bad-root', '--corpus-root', str(self.root / 'missing')),
                          self.arguments('zero-depth', '--depth', '0'),
                          ['--mode', 'benchmark', '--job-dir', str(self.root / 'no-inputs')],
                          ['--mode', 'demo', '--job-dir', 'relative'],
                          ['--mode', 'demo', '--job-dir', str(self.root / 'demo-dev'), '--split', 'dev']]:
            with self.subTest(arguments=arguments):
                code, messages = self.invoke(arguments)
                self.assertEqual(code, 1)
                self.assertEqual([m['event'] for m in messages], ['error'])
        self.assertFalse((self.root / 'missing.sqlite3').exists())
        self.assertFalse((self.root / 'zero-depth').exists())

    def test_non_database_error_is_actionable(self):
        invalid = self.root / 'invalid.sqlite3'
        invalid.write_text('not sqlite')
        code, messages = self.invoke(self.arguments('invalid', '--database', str(invalid)))
        self.assertEqual(code, 1)
        self.assertIn('readable FileSense index', messages[-1]['message'])
        self.assertEqual(invalid.read_text(), 'not sqlite')

    def test_offline_guards_block_tokenizer_download_and_ocr(self):
        import tiktoken.load
        original = tiktoken.load.read_file
        with worker.offline_engine() as guarded_engine:
            with self.assertRaises(worker.OfflineCacheError):
                tiktoken.load.read_file('https://example.invalid/tokenizer')
            with self.assertRaisesRegex(ValueError, 'OCR is disabled'):
                guarded_engine.extract_page_texts_with_ocr(self.pdfs / 'invented.pdf')
        self.assertIs(tiktoken.load.read_file, original)

    def test_missing_dependencies_and_model_cache_messages(self):
        self.assertIn('evaluation/requirements.txt', worker.friendly_error(ImportError('sklearn')))
        message = worker.friendly_error(OSError('Could not find cached files in huggingface cache'))
        self.assertIn('never downloads models', message)
        self.assertIn('TF-IDF', message)

    def test_missing_model_cache_emits_actionable_error_and_no_completion(self):
        with patch.object(engine, 'SentenceTransformer', side_effect=OSError("Can't load model")) as model:
            code, messages = self.invoke(self.arguments('uncached', '--systems', 'filesense'))
        self.assertEqual(code, 1)
        self.assertTrue(model.call_args.kwargs['local_files_only'])
        self.assertEqual(messages[-1]['event'], 'error')
        self.assertIn('never downloads models', messages[-1]['message'])
        self.assertNotIn('complete', [message['event'] for message in messages])
        self.assertFalse((self.root / 'uncached/report').exists())

    def test_worker_module_runs_as_subprocess_without_gui(self):
        process = subprocess.run([sys.executable, '-m', 'evaluation.gui_worker', *self.arguments()],
                                 cwd=worker.SOURCE_BASE, capture_output=True, text=True, timeout=40,
                                 env=dict(os.environ, QT_QPA_PLATFORM='invalid-headless-test',
                                          HF_HUB_OFFLINE='0'))
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        messages = events(process.stdout)
        self.assertEqual(messages[-1]['event'], 'complete')
        self.assertTrue(Path(messages[-1]['result_path']).is_file())

    def test_offline_flags_override_online_parent_before_ml_imports(self):
        script = ('import os, sys; import evaluation.gui_worker; '
                  'print(os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"], '
                  '"sentence_transformers" in sys.modules)')
        process = subprocess.run([sys.executable, '-c', script], cwd=worker.SOURCE_BASE,
                                 capture_output=True, text=True, timeout=10,
                                 env=dict(os.environ, HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0'))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), '1 1 False')


if __name__ == '__main__':
    unittest.main()
