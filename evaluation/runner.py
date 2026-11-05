"""Per-query/aggregate evaluation and portable exports (no query or PDF text)."""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter

from . import __version__
from .benchmark import bind_queries
from .ranking import dcg_at_k, ndcg_at_k, reciprocal_rank
from .retrieval import precision, precision_at_k, recall, recall_at_k, validate_k


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def evaluate(corpus, benchmark, backends, *, split='test', depth=10,
             cutoffs=(1, 5, 10), allow_unjudged=False):
    validate_k(depth)
    cutoffs = sorted(set(cutoffs))
    if not cutoffs:
        raise ValueError("At least one cutoff is required")
    for k in cutoffs:
        validate_k(k)
        if k > depth:
            raise ValueError("Every cutoff must be <= retrieval depth")
    if not backends:
        raise ValueError("At least one backend is required")
    queries = bind_queries(benchmark, corpus, split, allow_unjudged)
    rows, rankings, summaries = [], [], []
    for name, backend in backends.items():
        system_rows = []
        for query in queries:
            start = perf_counter()
            results = backend.search(query.text)[:depth]
            latency = (perf_counter() - start) * 1000
            ranking = [hit['document_id'] for hit in results]
            if len(set(ranking)) != len(ranking) or not set(ranking) <= set(corpus.documents):
                raise ValueError("Backend must return unique, known document IDs")
            relevant = {doc for doc, grade in query.judgments.items() if grade >= 1}
            tp = len(set(ranking) & relevant)
            row = dict(system=name, query_id=query.query_id, split=split,
                       retrieved=len(ranking), relevant=len(relevant), true_positive=tp,
                       judged=len(query.judgments), unjudged=len(corpus.documents) - len(query.judgments),
                       precision=precision(ranking, relevant), recall=recall(ranking, relevant),
                       reciprocal_rank=reciprocal_rank(ranking, relevant), latency_ms=latency)
            for k in cutoffs:
                row.update({f'precision@{k}': precision_at_k(ranking, relevant, k),
                            f'recall@{k}': recall_at_k(ranking, relevant, k),
                            f'dcg@{k}': dcg_at_k(ranking, query.judgments, k),
                            f'ndcg@{k}': ndcg_at_k(ranking, query.judgments, k),
                            f'true_positive@{k}': len(set(ranking[:k]) & relevant)})
            rows.append(row)
            system_rows.append(row)
            rankings.extend(dict(system=name, query_id=query.query_id, rank=i,
                                 document_id=hit['document_id'], score=hit['score'],
                                 relevance=query.judgments.get(hit['document_id'], 0),
                                 judged=hit['document_id'] in query.judgments)
                            for i, hit in enumerate(results, 1))
        eligible = sum(r['relevant'] > 0 for r in system_rows)
        summary = dict(system=name, split=split, queries=len(queries),
                       eligible_recall_rr_ndcg_queries=eligible,
                       zero_relevant_queries=len(queries) - eligible)
        metrics = ['precision', 'recall', 'reciprocal_rank'] + [
            f'{metric}@{k}' for k in cutoffs for metric in ('precision', 'recall', 'dcg', 'ndcg')]
        for metric in metrics:
            label = 'mrr_at_depth' if metric == 'reciprocal_rank' else f'macro_{metric}'
            summary[label] = _mean(r[metric] for r in system_rows)
        tp = sum(r['true_positive'] for r in system_rows)
        retrieved = sum(r['retrieved'] for r in system_rows)
        relevant = sum(r['relevant'] for r in system_rows)
        summary['pooled_precision'] = tp / retrieved if retrieved else 0.0
        summary['pooled_recall'] = tp / relevant if relevant else None
        for k in cutoffs:
            hits = sum(r[f'true_positive@{k}'] for r in system_rows)
            summary[f'pooled_precision@{k}'] = hits / (k * len(queries))
            summary[f'pooled_recall@{k}'] = hits / relevant if relevant else None
        summary['mean_latency_ms'] = _mean(r['latency_ms'] for r in system_rows)
        summary['median_latency_ms'] = statistics.median(r['latency_ms'] for r in system_rows)
        summaries.append(summary)
    return dict(schema_version=1, metadata=dict(
        evaluator_version=__version__, created_at=datetime.now(timezone.utc).isoformat(),
        benchmark_id=benchmark.benchmark_id, benchmark_schema_version=benchmark.schema_version,
        corpus_fingerprint=corpus.fingerprint, model_name=corpus.model_name,
        documents=len(corpus.documents), searchable_documents=len(corpus.searchable_ids),
        chunks=len(corpus.chunks), split=split, depth=depth, cutoffs=cutoffs,
        unjudged_policy='nonrelevant' if allow_unjudged else 'error',
        zero_relevant_policy='recall/RR/nDCG=null; excluded from their macro means',
        zero_idcg_policy='null; excluded from macro nDCG',
        precision_empty_policy=0, precision_at_k_denominator='k',
        binary_relevance='grade >= 1', gains='2**grade - 1', discount='log2(rank + 1)',
        ties='score descending, case-sensitive NFC relative PDF ID ascending',
        latency_scope='query encoding + scoring + sorting + document collapse; excludes setup and metrics',
        result_text_policy='query text, PDF text, and absolute PDF paths omitted'),
        per_query=rows, aggregate=summaries, rankings=rankings)


def provenance(benchmark_path):
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / 'evaluation').glob('*.py')) + [root / 'document_search_engine.py', root / 'document_database.py']
    packages = {}
    for name in ('numpy', 'scikit-learn', 'sentence-transformers', 'torch', 'transformers', 'PyMuPDF', 'tiktoken'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                benchmark_sha256=hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest(),
                source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def _csv_cell(value):
    # Query IDs and PDF names are user text, never spreadsheet formulas.
    if isinstance(value, str) and (value.startswith(('\t', '\r', '\n'))
                                  or value.lstrip().startswith(('=', '+', '-', '@'))):
        return "'" + value
    return value


def export_results(result, output):
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'results.json').write_text(serialized, encoding='utf-8')
    for key in ('per_query', 'aggregate', 'rankings'):
        rows = result[key]
        with (output / f'{key}.csv').open('w', newline='', encoding='utf-8') as handle:
            fields = list(rows[0]) if rows else ['system', 'query_id', 'rank', 'document_id', 'score', 'relevance', 'judged']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: _csv_cell(value) for key, value in row.items()} for row in rows)
    # JSON remains authoritative: CSV empty cells represent null; formula-like
    # identifiers are escaped with an apostrophe only in the CSV representation.
