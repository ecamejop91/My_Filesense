"""Version 1 JSON and long-form CSV benchmarks with strict validation."""
import csv
from dataclasses import dataclass
import json
from pathlib import Path

from .ranking import validate_judgments


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    split: str
    judgments: dict


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    queries: tuple
    schema_version: int = 1


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_benchmark(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'schema_version', 'benchmark_id', 'query_id', 'query',
                        'split', 'document_id', 'relevance'}
            if set(reader.fieldnames or []) != required:
                raise ValueError(f"CSV headers must be exactly {sorted(required)}")
            rows = list(reader)
        if not rows:
            raise ValueError("Benchmark must contain queries")
        groups = {}
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed CSV row: wrong number of fields')
            if (row['schema_version'], row['benchmark_id']) != ('1', rows[0]['benchmark_id']):
                raise ValueError("CSV version and benchmark_id must be consistent (version 1)")
            qid = row['query_id']
            query = groups.setdefault(qid, dict(query_id=qid, text=row['query'],
                                               split=row['split'], judgments={}))
            if (query['text'], query['split']) != (row['query'], row['split']):
                raise ValueError("Conflicting metadata for query")
            doc = row['document_id']
            if doc in query['judgments']:
                raise ValueError("Duplicate query/document judgment")
            if row['relevance'] not in ('0', '1', '2', '3'):
                raise ValueError("CSV relevance must be 0, 1, 2, or 3")
            query['judgments'][doc] = int(row['relevance'])
        data = dict(schema_version=1, benchmark_id=rows[0]['benchmark_id'], queries=list(groups.values()))
    elif path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_no_duplicate_keys)
    else:
        raise ValueError("Benchmark must be .json or .csv")
    if not isinstance(data, dict):
        raise ValueError('Benchmark JSON must be an object')
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError("Unsupported benchmark schema_version; expected 1")
    benchmark_id = _string(data.get('benchmark_id'), 'benchmark_id')
    queries, seen = [], set()
    if not isinstance(data.get('queries'), list) or not data['queries']:
        raise ValueError("Benchmark must contain queries")
    for row in data['queries']:
        if not isinstance(row, dict):
            raise ValueError('Each query must be an object')
        qid = _string(row.get('query_id'), 'query_id')
        if qid in seen:
            raise ValueError("Duplicate query_id")
        seen.add(qid)
        text = _string(row.get('text'), 'text')
        split = row.get('split')
        if split not in ('dev', 'test'):
            raise ValueError("Each query split must be dev or test")
        judgments = row.get('judgments')
        if not isinstance(judgments, dict):
            raise ValueError("judgments must map document IDs to grades")
        for doc in judgments:
            _string(doc, 'document_id')
        validate_judgments(judgments)
        queries.append(Query(qid, text, split, judgments))
    return Benchmark(benchmark_id, tuple(queries))


def bind_queries(benchmark, corpus, split, allow_unjudged=False):
    """Resolve aliases and require exhaustive judgments unless explicitly waived."""
    bound = []
    for query in benchmark.queries:
        judgments = {}
        for raw, grade in query.judgments.items():
            doc = corpus.resolve_id(raw)
            if doc in judgments:
                raise ValueError("Multiple judgment aliases refer to the same document")
            judgments[doc] = grade
        missing = set(corpus.documents) - set(judgments)
        if missing and not allow_unjudged:
            raise ValueError(f"Query {query.query_id}: {len(missing)} unjudged documents; "
                             "label all PDFs or explicitly use --allow-unjudged")
        if query.split == split:
            bound.append(Query(query.query_id, query.text, query.split, judgments))
    if not bound:
        raise ValueError(f"No queries in split {split}")
    return bound
