"""Headless, offline evaluation jobs launched by the desktop GUI with QProcess.

The existing ``python -m evaluation`` interface remains independent. Every job
owns a new output directory, a benchmark copy, and a frozen SQLite index.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from time import perf_counter

# Set these before importing the engine, transformers, or huggingface_hub. The
# GUI may supply HF_HOME to reuse its existing cache; never change that path.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

EVENT_PREFIX = 'FILESENSE_EVAL_EVENT:'
SOURCE_BASE = Path(__file__).resolve().parents[1]
TIMING_NOTE = ('Timings are local observations from one run. They include query encoding, '
               'scoring, sorting, and document collapse, but exclude setup and metrics. '
               'Cache state, hardware, and system load affect them; they are not definitive '
               'performance comparisons.')
DEMO_NOTE = ('Synthetic smoke test using three invented PDFs and four authored queries. '
             'It demonstrates the evaluation workflow; its scores do not establish '
             'retrieval quality on real documents.')


class OfflineCacheError(RuntimeError):
    """A required local asset was unavailable; no download was attempted."""


class WorkerParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parser():
    result = WorkerParser(description=__doc__)
    result.add_argument('--mode', choices=('demo', 'benchmark'), required=True)
    result.add_argument('--job-dir', type=Path, required=True)
    result.add_argument('--benchmark', type=Path)
    result.add_argument('--database', type=Path)
    result.add_argument('--corpus-root', type=Path)
    result.add_argument('--source-base', type=Path, default=SOURCE_BASE)
    result.add_argument('--split', choices=('dev', 'test'), default='test')
    result.add_argument('--depth', type=int, default=10)
    result.add_argument('--systems', choices=('both', 'filesense', 'tfidf'), default='both')
    return result


def emit(event, **fields):
    print(EVENT_PREFIX + json.dumps(dict(event=event, **fields), allow_nan=False), flush=True)


def _validate_request(args):
    args.job_dir = args.job_dir.expanduser()
    if not args.job_dir.is_absolute():
        raise ValueError('Job directory must be an absolute path to a new directory.')
    if args.job_dir.exists() or args.job_dir.is_symlink():
        raise ValueError('Job directory already exists. Start a new run with a new job directory.')
    args.job_dir = args.job_dir.resolve()
    if args.depth <= 0:
        raise ValueError('Retrieval depth must be a positive integer.')
    if args.mode == 'demo':
        if args.split != 'test':
            raise ValueError('The synthetic demo contains test queries. Select the test split.')
        return
    for option in ('benchmark', 'database', 'corpus_root'):
        if getattr(args, option) is None:
            raise ValueError('Benchmark mode requires --' + option.replace('_', '-') + '.')
        setattr(args, option, getattr(args, option).expanduser().resolve())
    if not args.benchmark.is_file():
        raise ValueError('Benchmark file was not found. Select an existing labeled JSON or CSV file.')
    if args.benchmark.suffix.lower() not in ('.json', '.csv'):
        raise ValueError('Benchmark must be a labeled .json or .csv file.')
    if not args.database.is_file():
        raise ValueError('SQLite index was not found. Select an existing file_tags.sqlite3 index.')
    if not args.corpus_root.is_dir():
        raise ValueError('Corpus root must be the existing directory containing the indexed PDFs.')


def snapshot_database(source, destination):
    """SQLite backup includes committed WAL content without opening a write connection."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as reader:
        reader.execute('PRAGMA query_only = ON')
        with closing(sqlite3.connect(destination)) as writer:
            reader.backup(writer)


@contextmanager
def offline_engine():
    """Use the actual engine with cached models and fail before any tokenizer download."""
    import document_search_engine as engine
    import tiktoken.load

    original_model = engine.SentenceTransformer
    original_read = tiktoken.load.read_file
    original_ocr = engine.extract_page_texts_with_ocr

    def cached_model(*args, **kwargs):
        kwargs['local_files_only'] = True
        try:
            return original_model(*args, **kwargs)
        except OSError as exc:
            raise OfflineCacheError(
                'The FileSense embedding model could not load from the local cache. Prepare '
                'the matching model in the FileSense environment and retry, or use TF-IDF '
                'with an existing index. Evaluation never downloads models. Details: ' + str(exc)) from exc

    def cached_tokenizer_asset(path):
        if str(path).startswith(('https://', 'http://')):
            raise OfflineCacheError(
                'The tiktoken tokenizer is not cached. Prepare the cl100k_base tokenizer '
                'cache in the FileSense environment, then retry. Evaluation never downloads assets.')
        return original_read(path)

    def no_ocr(*args, **kwargs):
        raise ValueError('The synthetic demo PDF has no extractable text. Start a new demo run; '
                         'OCR is disabled for evaluation demos.')

    engine.SentenceTransformer = cached_model
    tiktoken.load.read_file = cached_tokenizer_asset
    engine.extract_page_texts_with_ocr = no_ocr
    try:
        yield engine
    finally:
        engine.SentenceTransformer = original_model
        tiktoken.load.read_file = original_read
        engine.extract_page_texts_with_ocr = original_ocr


def _create_demo_index(args, engine):
    from .corpus import load_corpus
    from .demo import create_demo

    emit('status', message='Creating three synthetic PDFs and a fully labeled benchmark…')
    demo_dir = args.job_dir / 'demo'
    create_demo(demo_dir)
    args.corpus_root = demo_dir / 'pdfs'
    args.benchmark = demo_dir / 'benchmark.json'
    args.source_base = SOURCE_BASE
    emit('status', message='Building an isolated demo index with the cached FileSense model…')
    start = perf_counter()
    store = engine.build_index(files_dir=args.corpus_root, model_name=engine.MODEL_NAME,
                               encoding_name=engine.ENCODING_NAME, chunk_size=engine.CHUNK_SIZE,
                               include_checkpoints=False)
    index_dir = args.job_dir / 'index'
    engine.save_vector_store(store['doc_records'], store['chunk_records'], store['chunk_vectors'],
                             index_dir, engine.MODEL_NAME)
    args.database = index_dir / 'file_tags.sqlite3'
    corpus = load_corpus(args.database, args.corpus_root, args.source_base)
    settings = dict(schema_version=1, model=engine.MODEL_NAME, chunk_size=engine.CHUNK_SIZE,
                    encoding=engine.ENCODING_NAME, include_checkpoints=False,
                    corpus_fingerprint=corpus.fingerprint, build_seconds=perf_counter() - start,
                    extraction='FileSense embedded text; OCR disabled for synthetic demo',
                    model_revision='not recorded by FileSense; preserve local weights/environment')
    (index_dir / 'evaluation-index.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    return corpus, settings, store['model']


def _load_benchmark_index(args):
    from .corpus import load_corpus

    emit('status', message='Taking a read-only SQLite snapshot of the selected index…')
    destination = args.job_dir / 'index' / 'file_tags.sqlite3'
    snapshot_database(args.database, destination)
    corpus = load_corpus(destination, args.corpus_root, args.source_base)
    sidecar = args.database.parent / 'evaluation-index.json'
    if sidecar.exists():
        settings = json.loads(sidecar.read_text(encoding='utf-8'))
        if not isinstance(settings, dict) or settings.get('corpus_fingerprint') != corpus.fingerprint:
            raise ValueError('The evaluation-index.json settings do not match the selected index. '
                             'Select the matching index or rebuild a separate evaluation index.')
        (destination.parent / sidecar.name).write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    else:
        settings = dict(chunk_size='unknown; not stored in native SQLite metadata',
                        model_revision='unknown; not stored in native SQLite metadata')
    return corpus, settings


def run_job(args):
    _validate_request(args)
    # Imports are delayed so missing packages still produce structured errors.
    from .backends import FileSenseSearch, TfidfSearch, TFIDF_SETTINGS
    from .benchmark import bind_queries, load_benchmark
    from .runner import evaluate, export_results, provenance

    args.job_dir.mkdir(parents=True, exist_ok=False)
    from app_storage import private_directory
    private_directory(args.job_dir)
    with ExitStack() as stack:
        demo_model = None
        if args.mode == 'demo':
            engine = stack.enter_context(offline_engine())
            corpus, settings, demo_model = _create_demo_index(args, engine)
        else:
            corpus, settings = _load_benchmark_index(args)

        benchmark_copy = args.job_dir / ('benchmark' + args.benchmark.suffix.lower())
        benchmark_copy.write_bytes(args.benchmark.read_bytes())
        emit('status', message='Validating document IDs, relevance labels, and the selected split…')
        benchmark = load_benchmark(benchmark_copy)
        bind_queries(benchmark, corpus, args.split, allow_unjudged=False)

        backends, setup = {}, {}
        if args.systems in ('both', 'filesense'):
            if args.mode != 'demo':
                stack.enter_context(offline_engine())
            emit('status', message='Preparing FileSense semantic search from the cached model…')
            start = perf_counter()
            backends['filesense'] = FileSenseSearch(corpus, model=demo_model)
            setup['filesense'] = perf_counter() - start
        if args.systems in ('both', 'tfidf'):
            emit('status', message='Fitting the TF-IDF baseline on indexed document text…')
            start = perf_counter()
            backends['tfidf'] = TfidfSearch(corpus)
            setup['tfidf'] = perf_counter() - start

        cutoffs = sorted({1, args.depth, *[k for k in (5, 10) if k <= args.depth]})
        emit('status', message='Running labeled queries and computing retrieval and ranking metrics…')
        result = evaluate(corpus, benchmark, backends, split=args.split, depth=args.depth,
                          cutoffs=cutoffs, allow_unjudged=False)
        result['metadata'].update(provenance(benchmark_copy))
        result['metadata'].update(
            mode=args.mode, synthetic_demo=args.mode == 'demo',
            interpretation_note=DEMO_NOTE if args.mode == 'demo' else
            'Scores describe this labeled benchmark, selected split, and frozen indexed corpus.',
            timing_note=TIMING_NOTE, offline=True, setup_seconds=setup,
            setup_timing_note=('The demo reuses the model loaded during its index build; '
                               'that loading time is included in index_build.build_seconds.'
                               if args.mode == 'demo' else
                               'Backend setup includes model loading or TF-IDF fitting and is '
                               'reported separately from query latency.'),
            tfidf_settings=TFIDF_SETTINGS, index_build=settings,
            benchmark_copy=benchmark_copy.name, index_snapshot='index/file_tags.sqlite3',
            index_snapshot_sha256=hashlib.sha256(
                (args.job_dir / 'index' / 'file_tags.sqlite3').read_bytes()).hexdigest())
        emit('status', message='Writing the JSON report and per-query, aggregate, and ranking CSV files…')
        report_dir = args.job_dir / 'report'
        export_results(result, report_dir)
    emit('complete', report_dir=str(report_dir), result_path=str(report_dir / 'results.json'))
    return report_dir


def friendly_error(exc):
    if isinstance(exc, ImportError):
        return ('An evaluation dependency is missing or could not load. In the same Python '
                'environment as FileSense, install requirements.txt and evaluation/requirements.txt, '
                f'then retry. Details: {exc}')
    if isinstance(exc, sqlite3.DatabaseError):
        return ('The selected SQLite file is not a readable FileSense index. Select an index '
                'containing documents, chunks, and index_metadata, or build a separate evaluation '
                f'index. Details: {exc}')
    message = str(exc)
    if isinstance(exc, OfflineCacheError):
        return message
    if any(word in message.lower() for word in ('huggingface', 'hugging face', 'offline',
                                                'local_files_only', 'cached files', 'couldn\'t connect')):
        return ('The FileSense embedding model is unavailable in the local cache. Prepare the '
                'matching model in the FileSense environment and retry, or use TF-IDF with an '
                'existing index. Evaluation runs offline and never downloads models. Details: ' + message)
    if 'unjudged documents' in message:
        return (message.split(';', 1)[0] + '. Add an explicit 0–3 relevance label for every '
                'indexed PDF in each query; in-app evaluation requires complete judgments.')
    return message or type(exc).__name__


def main(argv=None):
    try:
        run_job(parser().parse_args(argv))
    except Exception as exc:
        emit('error', message=friendly_error(exc))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
