"""Run with python -m evaluation; no GUI imports or normal-index writes."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from .backends import FileSenseSearch, TfidfSearch, TFIDF_SETTINGS
from .benchmark import bind_queries, load_benchmark
from .corpus import load_corpus
from .runner import evaluate, export_results, provenance


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Generate invented PDFs and a labeled smoke benchmark')
    demo.add_argument('--output', type=Path, required=True)
    index = commands.add_parser('index', help='Build a separate evaluation index from explicit PDFs')
    index.add_argument('--corpus-root', type=Path, required=True)
    index.add_argument('--output', type=Path, required=True)
    index.add_argument('--chunk-size', type=int, default=200)  # keep equal to document_search_engine.CHUNK_SIZE
    index.add_argument('--model', default='all-MiniLM-L6-v2')
    index.add_argument('--encoding', default='cl100k_base')
    for name in ('inspect', 'run'):
        sub = commands.add_parser(name)
        sub.add_argument('--database', type=Path, required=True)
        sub.add_argument('--corpus-root', type=Path, required=True)
        sub.add_argument('--source-base', type=Path, default=Path(__file__).resolve().parents[1],
                         help='Base for legacy relative paths stored in SQLite (default: repository)')
        sub.add_argument('--output', type=Path, required=True)
        if name == 'run':
            sub.add_argument('--benchmark', type=Path, required=True)
            sub.add_argument('--systems', choices=('both', 'filesense', 'tfidf'), default='both')
            sub.add_argument('--split', choices=('dev', 'test'), default='test')
            sub.add_argument('--depth', type=int, default=10)
            sub.add_argument('--k', type=int, nargs='+', default=[1, 5, 10])
            sub.add_argument('--allow-unjudged', action='store_true')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('Output already exists; choose a new output path')
        if args.command == 'demo':
            from .demo import create_demo
            create_demo(args.output)
        elif args.command == 'index':
            if args.chunk_size <= 0:
                raise ValueError('chunk-size must be positive')
            if not args.corpus_root.is_dir():
                raise ValueError('corpus-root must be an existing directory')
            import document_search_engine as engine
            if args.output.resolve() == engine.VECTOR_STORE_DIR.resolve():
                raise ValueError('Use a separate evaluation output, never the normal GUI index')
            if not list(engine.iter_document_files(args.corpus_root)):
                raise ValueError('Corpus contains no supported documents recognized by FileSense')
            start = perf_counter()
            store = engine.build_index(files_dir=args.corpus_root, model_name=args.model,
                                       encoding_name=args.encoding, chunk_size=args.chunk_size)
            args.output.mkdir(parents=True, exist_ok=False)
            engine.save_vector_store(store['doc_records'], store['chunk_records'], store['chunk_vectors'],
                                     args.output, args.model)
            corpus = load_corpus(args.output / 'file_tags.sqlite3', args.corpus_root, engine.PROJECT_DIR)
            (args.output / 'evaluation-index.json').write_text(json.dumps(dict(
                schema_version=1, model=args.model, chunk_size=args.chunk_size,
                encoding=args.encoding, include_checkpoints=False,
                corpus_fingerprint=corpus.fingerprint, build_seconds=perf_counter() - start,
                extraction='FileSense embedded-text-first, whole-document OCR fallback',
                model_revision='not recorded by FileSense; preserve local weights/environment'), indent=2) + '\n')
        else:
            corpus = load_corpus(args.database, args.corpus_root, args.source_base)
            if args.command == 'inspect':
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('x', encoding='utf-8') as handle:
                    json.dump(dict(corpus_fingerprint=corpus.fingerprint, documents=[
                        dict(document_id=key, method=doc['method'], searchable=bool(doc['text']))
                        for key, doc in sorted(corpus.documents.items())]), handle, indent=2)
            else:
                from .retrieval import validate_k
                validate_k(args.depth)
                for k in args.k:
                    validate_k(k)
                    if k > args.depth:
                        raise ValueError('Every cutoff must be <= retrieval depth')
                benchmark = load_benchmark(args.benchmark)
                bind_queries(benchmark, corpus, args.split, args.allow_unjudged)
                backends, setup = {}, {}
                for name, factory in [('filesense', FileSenseSearch), ('tfidf', TfidfSearch)]:
                    if args.systems in ('both', name):
                        start = perf_counter()
                        backends[name] = factory(corpus)
                        setup[name] = perf_counter() - start
                result = evaluate(corpus, benchmark, backends, split=args.split, depth=args.depth,
                                  cutoffs=args.k, allow_unjudged=args.allow_unjudged)
                result['metadata'].update(provenance(args.benchmark))
                result['metadata'].update(setup_seconds=setup, tfidf_settings=TFIDF_SETTINGS)
                sidecar = args.database.parent / 'evaluation-index.json'
                if sidecar.exists():
                    settings = json.loads(sidecar.read_text())
                    if settings.get('corpus_fingerprint') != corpus.fingerprint:
                        raise ValueError('Evaluation index sidecar does not match the database snapshot')
                    result['metadata']['index_build'] = settings
                else:
                    result['metadata']['index_build'] = {
                        'chunk_size': 'unknown; not stored in native SQLite metadata',
                        'model_revision': 'unknown; not stored in native SQLite metadata'}
                export_results(result, args.output)
        print(f'Evaluation {args.command} complete: {args.output}')
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
