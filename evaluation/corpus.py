"""Read-only snapshot of FileSense's inspected SQLite schema and ID mapping."""
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import sqlite3
import unicodedata

import numpy as np
from document_formats import is_supported


def normalized_path(raw, base):
    raw = str(raw)
    if PureWindowsPath(raw).is_absolute() and not Path(raw).is_absolute():
        raise ValueError("Windows absolute paths require running on Windows or rebuilding the index")
    path = Path(raw.replace('\\', '/')).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def document_id(raw, corpus_root, relative_base=None):
    root = Path(corpus_root).expanduser().resolve()
    path = normalized_path(raw, relative_base or root)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Document lies outside --corpus-root") from exc
    if not is_supported(path):
        raise ValueError('Corpus contains an unsupported document type')
    return unicodedata.normalize('NFC', relative.as_posix())


@dataclass
class Corpus:
    documents: dict
    chunks: list
    vectors: np.ndarray
    model_name: str
    root: Path
    source_base: Path
    aliases: dict
    fingerprint: str

    @classmethod
    def from_records(cls, docs, chunks, vectors, model_name, root, source_base):
        root, source_base = Path(root).resolve(), Path(source_base).resolve()
        documents, aliases, ids = {}, {}, {}
        for doc in docs:
            key = document_id(doc['pdf_path'], root, source_base)
            if key in documents or doc['doc_id'] in ids:
                raise ValueError("Duplicate document ID or normalized PDF path")
            documents[key] = {**doc, 'text': ''}
            ids[doc['doc_id']] = key
            aliases[doc['doc_id']] = key
        if not documents:
            raise ValueError("Corpus is empty")
        # An index-generated ID cannot silently shadow a relative PDF ID.
        for key in documents:
            if key in aliases and aliases[key] != key:
                raise ValueError("Ambiguous document alias")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError("Invalid chunk/vector alignment or non-finite vectors")
        if len(chunks) and (vectors.shape[1] == 0 or not np.allclose(
                np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)):
            raise ValueError("Expected unit-normalized FileSense chunk embeddings")
        seen, texts = set(), {key: [] for key in documents}
        for chunk in chunks:
            if chunk['doc_id'] not in ids:
                raise ValueError("Orphan chunk")
            key = ids[chunk['doc_id']]
            if document_id(chunk['pdf_path'], root, source_base) != key:
                raise ValueError("Chunk PDF path disagrees with document")
            pair = (key, chunk['chunk_id'])
            if pair in seen:
                raise ValueError("Duplicate chunk ID")
            seen.add(pair)
            if (type(chunk['page']) is not int or chunk['page'] < 1 or
                    type(chunk['chunk_id']) is not int or chunk['chunk_id'] < 0 or
                    not isinstance(chunk['text'], str) or not chunk['text'].strip()):
                raise ValueError("Invalid chunk page, ID, or text")
            texts[key].append((chunk['page'], chunk['chunk_id'], chunk['text']))
        for key, parts in texts.items():
            documents[key]['text'] = '\n'.join(part[2] for part in sorted(parts))
        # Portable hash: excludes absolute paths, includes exact extracted chunks,
        # aligned embeddings, and model identity. It is NOT a raw-PDF checksum.
        digest = hashlib.sha256()
        payload = dict(model_name=model_name, vector_shape=list(vectors.shape), documents=[
            (key, documents[key]['method'], documents[key]['text']) for key in sorted(documents)],
            chunks=[(ids[c['doc_id']], c['page'], c['chunk_id'], c['text']) for c in chunks])
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
        digest.update(vectors.astype('<f4').tobytes())
        return cls(documents, chunks, vectors, model_name, root, source_base, aliases, digest.hexdigest())

    def resolve_id(self, raw):
        if raw in self.aliases:
            return self.aliases[raw]
        key = document_id(raw, self.root)
        if key not in self.documents:
            raise ValueError(f"Unknown benchmark document ID: {raw}")
        return key

    def resolve_hit(self, raw):
        key = document_id(raw, self.root, self.source_base)
        if key not in self.documents:
            raise ValueError("Search returned a document outside the evaluated corpus")
        return key

    @property
    def searchable_ids(self):
        return sorted(key for key, doc in self.documents.items() if doc['text'])


def load_corpus(database, corpus_root, source_base):
    """Never call connect_database/load_index: they can migrate or create tables."""
    path = Path(database).expanduser().resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN')  # One consistent snapshot, including WAL state.
        metadata = connection.execute('SELECT * FROM index_metadata WHERE id=1').fetchone()
        if metadata is None or metadata['vector_dimension'] < 0:
            raise ValueError("Missing/invalid index metadata; build an isolated evaluation index first")
        docs = [dict(row) for row in connection.execute('SELECT * FROM documents ORDER BY doc_id')]
        paths = {d['doc_id']: d['pdf_path'] for d in docs}
        rows = connection.execute('SELECT * FROM chunks ORDER BY position').fetchall()
        vectors = np.empty((len(rows), metadata['vector_dimension']), dtype=np.float32)
        chunks = []
        for i, row in enumerate(rows):
            if row['doc_id'] not in paths:
                raise ValueError("Orphan chunk in database")
            vector = np.frombuffer(row['embedding'], dtype='<f4')
            if len(vector) != metadata['vector_dimension']:
                raise ValueError("Invalid stored embedding dimension")
            vectors[i] = vector
            chunks.append({**{k: row[k] for k in ('doc_id', 'page', 'chunk_id', 'text')},
                           'pdf_path': paths[row['doc_id']]})
    return Corpus.from_records(docs, chunks, vectors, metadata['model_name'], corpus_root, source_base)


def rank_scores(scores):
    """Shared deterministic tie policy: score descending, canonical ID ascending."""
    if any(not math.isfinite(float(score)) for score in scores.values()):
        raise ValueError("Search produced non-finite scores")
    return [{'document_id': key, 'score': float(score)}
            for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def deduplicate_hits(hits, corpus):
    scores = {}
    for hit in hits:
        key = corpus.resolve_hit(hit['path'])
        score = float(hit['score'])
        if not math.isfinite(score):
            raise ValueError("Search produced non-finite scores")
        scores[key] = max(scores.get(key, -math.inf), score)
    return rank_scores(scores)
