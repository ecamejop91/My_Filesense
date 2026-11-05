"""SQLite persistence for the PDF search index, alongside existing tag tables."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
from app_storage import PATHS, prepare_managed_file

PROJECT_DIR = Path(__file__).resolve().parent
FILES_DIR = PROJECT_DIR / "Files"
VECTOR_STORE_DIR = PATHS.library
# Keep the filename for explicit imports; runtime libraries are per account.
DATABASE_NAME = "file_tags.sqlite3"
DEFAULT_DB_PATH = VECTOR_STORE_DIR / DATABASE_NAME


def connect_database(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    prepare_managed_file(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                pdf_path TEXT NOT NULL UNIQUE,
                method TEXT NOT NULL,
                char_count INTEGER NOT NULL CHECK (char_count >= 0)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                position INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                page INTEGER NOT NULL CHECK (page > 0),
                chunk_id INTEGER NOT NULL CHECK (chunk_id >= 0),
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                UNIQUE (doc_id, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS index_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model_name TEXT NOT NULL,
                vector_dimension INTEGER NOT NULL CHECK (vector_dimension >= 0)
            );
            CREATE TABLE IF NOT EXISTS document_processing (
                doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
                content_sha256 TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                indexing_signature TEXT NOT NULL,
                tag_signature TEXT
            );
        """)
    except Exception:
        connection.close()
        raise
    return connection


def _write_index(connection, doc_records, chunk_records, chunk_vectors, model_name):
    vectors = np.asarray(chunk_vectors, dtype="<f4")
    if not chunk_records and vectors.size == 0 and vectors.ndim == 1:
        vectors = vectors.reshape(0, 0)
    if vectors.ndim != 2 or len(vectors) != len(chunk_records):
        raise ValueError("The number of embeddings must match the number of text chunks.")
    if len(vectors) and (vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
        raise ValueError("Embeddings must contain finite values and at least one dimension.")
    paths = {doc["doc_id"]: doc["pdf_path"] for doc in doc_records}
    for chunk in chunk_records:
        if chunk["doc_id"] not in paths or chunk["pdf_path"] != paths[chunk["doc_id"]]:
            raise ValueError("Every text chunk must belong to its matching document.")

    # The caller owns the transaction: a failed replacement restores the old index.
    connection.execute("DELETE FROM chunks")
    connection.execute("DELETE FROM documents")
    connection.executemany(
        "INSERT INTO documents (doc_id, pdf_path, method, char_count) VALUES (?, ?, ?, ?)",
        [(d["doc_id"], d["pdf_path"], d["method"], d["char_count"]) for d in doc_records],
    )
    connection.executemany(
        "INSERT INTO chunks (position, doc_id, page, chunk_id, text, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(i, c["doc_id"], c["page"], c["chunk_id"], c["text"], vectors[i].tobytes())
         for i, c in enumerate(chunk_records)],
    )
    connection.execute(
        "INSERT INTO index_metadata (id, model_name, vector_dimension) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET model_name=excluded.model_name, "
        "vector_dimension=excluded.vector_dimension",
        (model_name, vectors.shape[1]),
    )


def save_index(doc_records, chunk_records, chunk_vectors, vector_store_dir, model_name):
    with closing(connect_database(Path(vector_store_dir) / DATABASE_NAME)) as connection:
        with connection:
            _write_index(connection, doc_records, chunk_records, chunk_vectors, model_name)


def load_processing_states(vector_store_dir):
    with closing(connect_database(Path(vector_store_dir) / DATABASE_NAME)) as connection:
        return {row['doc_id']: dict(row) for row in connection.execute(
            'SELECT p.*, d.pdf_path FROM document_processing p JOIN documents d USING(doc_id)')}


def apply_index_updates(documents, chunks, vectors, states, vector_store_dir, model_name):
    """Replace only changed PDFs and their processing state in one transaction."""
    vectors = np.asarray(vectors, dtype='<f4')
    if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
        raise ValueError('Invalid changed chunk/embedding alignment')
    paths = {doc['doc_id']: doc['pdf_path'] for doc in documents}
    if any(c['doc_id'] not in paths or c['pdf_path'] != paths[c['doc_id']] for c in chunks):
        raise ValueError('Changed chunks must match their documents')
    with closing(connect_database(Path(vector_store_dir) / DATABASE_NAME)) as connection:
        with connection:
            connection.execute('BEGIN IMMEDIATE')
            for doc in documents:
                connection.execute('DELETE FROM chunks WHERE doc_id=?', (doc['doc_id'],))
                if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='file_tags'").fetchone():
                    connection.execute('DELETE FROM file_tags WHERE file_path=?',
                                       (str(Path(doc['pdf_path']).resolve()),))
                connection.execute(
                    'INSERT INTO documents VALUES (?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET '
                    'pdf_path=excluded.pdf_path, method=excluded.method, char_count=excluded.char_count',
                    (doc['doc_id'], doc['pdf_path'], doc['method'], doc['char_count']))
            position = connection.execute('SELECT COALESCE(MAX(position), -1) + 1 FROM chunks').fetchone()[0]
            connection.executemany('INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)', [
                (position + i, c['doc_id'], c['page'], c['chunk_id'], c['text'], vectors[i].tobytes())
                for i, c in enumerate(chunks)])
            if documents:
                connection.execute('INSERT INTO index_metadata VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE SET '
                                   'model_name=excluded.model_name, vector_dimension=excluded.vector_dimension',
                                   (model_name, vectors.shape[1]))
            connection.executemany(
                'INSERT INTO document_processing VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET '
                'content_sha256=excluded.content_sha256, file_size=excluded.file_size, mtime_ns=excluded.mtime_ns, '
                'indexing_signature=excluded.indexing_signature, tag_signature=excluded.tag_signature',
                [(s['doc_id'], s['content_sha256'], s['file_size'], s['mtime_ns'], s['indexing_signature'],
                  s.get('tag_signature')) for s in states])


def edit_indexed_document(old_path, vector_store_dir, model_name, new_path=None):
    """Rename/remove one document without loading models or rewriting embeddings."""
    target = Path(old_path).expanduser().resolve()
    with closing(connect_database(Path(vector_store_dir) / DATABASE_NAME)) as connection:
        with connection:
            connection.execute('BEGIN IMMEDIATE')
            metadata = connection.execute('SELECT model_name FROM index_metadata WHERE id=1').fetchone()
            if metadata and metadata['model_name'] != model_name:
                raise ValueError('Use the model recorded by this index.')
            found = next((row for row in connection.execute('SELECT doc_id, pdf_path FROM documents')
                          if (Path(row['pdf_path']) if Path(row['pdf_path']).is_absolute()
                              else PROJECT_DIR / row['pdf_path']).resolve() == target), None)
            if found is None:
                return False
            has_tags = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='file_tags'").fetchone()
            if new_path is None:
                connection.execute('DELETE FROM documents WHERE doc_id=?', (found['doc_id'],))
                if has_tags:
                    connection.execute('DELETE FROM file_tags WHERE file_path=?', (str(target),))
            else:
                destination = Path(new_path).expanduser().resolve()
                connection.execute('UPDATE documents SET pdf_path=? WHERE doc_id=?',
                                   (destination.as_posix(), found['doc_id']))
                if has_tags:
                    connection.execute('UPDATE file_tags SET file_path=?, file_name=? WHERE file_path=?',
                                       (str(destination), destination.name, str(target)))
    return True


def clear_database(vector_store_dir=VECTOR_STORE_DIR):
    """Atomically clear indexed content and assignments, retaining tag definitions.

    Keep an empty index metadata row so old JSONL backups are not imported again.
    """
    with closing(connect_database(Path(vector_store_dir) / DATABASE_NAME)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='file_tags'"
            ).fetchone():
                connection.execute("DELETE FROM file_tags")
            connection.execute(
                "INSERT OR IGNORE INTO index_metadata (id, model_name, vector_dimension) "
                "VALUES (1, ?, 0)", ("all-MiniLM-L6-v2",)
            )


def _import_legacy_index(connection, folder: Path, model_name: str):
    paths = [folder / name for name in ("docs.jsonl", "chunks.jsonl", "vectors.npy")]
    if not any(path.exists() for path in paths):
        return
    if not all(path.exists() for path in paths):
        raise ValueError(
            "The legacy index is incomplete. Restore docs.jsonl, chunks.jsonl, and "
            "vectors.npy together before importing it into SQLite."
        )
    records = []
    for path in paths[:2]:
        with path.open(encoding="utf-8") as handle:
            records.append([json.loads(line) for line in handle if line.strip()])
    for group in records:
        for record in group:
            pdf_path = Path(record["pdf_path"]).expanduser()
            if not pdf_path.is_absolute():
                pdf_path = folder.resolve().parent / pdf_path
            record["pdf_path"] = str(pdf_path.resolve())
    vectors = np.load(paths[2], allow_pickle=False)
    # Original files remain untouched as a migration backup.
    _write_index(connection, records[0], records[1], vectors, model_name)


def _checked_metadata(connection, folder: Path, model_name: str):
    """Return index metadata, importing a legacy JSONL index first if needed.

    Call inside a write transaction so the one-time migration is atomic.
    """
    metadata = connection.execute("SELECT * FROM index_metadata WHERE id = 1").fetchone()
    if metadata is None:
        _import_legacy_index(connection, folder, model_name)
        metadata = connection.execute("SELECT * FROM index_metadata WHERE id = 1").fetchone()
    if metadata is not None and metadata["model_name"] != model_name:
        raise ValueError(
            f"This index uses {metadata['model_name']!r}, not {model_name!r}. "
            "Use the original model or rebuild the index with the requested model."
        )
    return metadata


def load_index_summary(vector_store_dir=VECTOR_STORE_DIR, model_name="all-MiniLM-L6-v2"):
    """Document rows and index facts without reading any chunk text or embeddings.

    Analysis uses this instead of load_index so its cost depends on the files
    being analyzed, not on the size of the whole library.
    """
    folder = Path(vector_store_dir)
    with closing(connect_database(folder / DATABASE_NAME)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            metadata = _checked_metadata(connection, folder, model_name)
            documents = [dict(row) for row in connection.execute(
                "SELECT doc_id, pdf_path, method, char_count FROM documents ORDER BY rowid"
            )]
            has_chunks = connection.execute("SELECT EXISTS (SELECT 1 FROM chunks)").fetchone()[0]
    return dict(
        documents=documents,
        dimension=metadata["vector_dimension"] if metadata is not None else 0,
        has_chunks=bool(has_chunks),
    )


def load_document_chunks(vector_store_dir, doc_ids):
    """Chunks and embeddings for only the given documents, in chunk order.

    Returns {doc_id: (chunk dicts with page/text, float32 matrix)}; documents
    without chunks map to an empty list and a (0, dimension) matrix.
    """
    doc_ids = list(dict.fromkeys(doc_ids))
    with closing(connect_database(Path(vector_store_dir) / DATABASE_NAME)) as connection:
        metadata = connection.execute("SELECT vector_dimension FROM index_metadata WHERE id = 1").fetchone()
        dimension = metadata["vector_dimension"] if metadata is not None else 0
        found = {doc_id: ([], []) for doc_id in doc_ids}
        # Stay well below SQLite's bound-parameter limit.
        for start in range(0, len(doc_ids), 500):
            batch = doc_ids[start:start + 500]
            marks = ", ".join("?" * len(batch))
            for row in connection.execute(
                f"SELECT doc_id, page, chunk_id, text, embedding FROM chunks "
                f"WHERE doc_id IN ({marks}) ORDER BY doc_id, chunk_id", batch
            ):
                chunks, vectors = found[row["doc_id"]]
                chunks.append({"page": row["page"], "text": row["text"]})
                vectors.append(row["embedding"])
    result = {}
    for doc_id, (chunks, blobs) in found.items():
        matrix = (np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(blobs), -1).astype(np.float32)
                  if blobs else np.empty((0, dimension), dtype=np.float32))
        if len(matrix) and (matrix.shape[1] != dimension or not np.isfinite(matrix).all()):
            raise ValueError("The SQLite index contains an invalid embedding. Rebuild the index.")
        result[doc_id] = (chunks, matrix)
    return result


def load_index(vector_store_dir=VECTOR_STORE_DIR, model_name="all-MiniLM-L6-v2"):
    folder = Path(vector_store_dir)
    with closing(connect_database(folder / DATABASE_NAME)) as connection:
        # One transaction gives a consistent snapshot and makes migration atomic.
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            metadata = _checked_metadata(connection, folder, model_name)
            documents = [dict(row) for row in connection.execute(
                "SELECT doc_id, pdf_path, method, char_count FROM documents ORDER BY rowid"
            )]
            rows = connection.execute(
                "SELECT c.doc_id, d.pdf_path, c.page, c.chunk_id, c.text, c.embedding "
                "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id ORDER BY c.position"
            ).fetchall()
            dimension = metadata["vector_dimension"] if metadata is not None else 0
            vectors = np.empty((len(rows), dimension), dtype=np.float32)
            chunks = []
            for i, row in enumerate(rows):
                vector = np.frombuffer(row["embedding"], dtype="<f4")
                if len(vector) != dimension or not np.isfinite(vector).all():
                    raise ValueError("The SQLite index contains an invalid embedding. Rebuild the index.")
                vectors[i] = vector
                chunks.append({key: row[key] for key in ("doc_id", "pdf_path", "page", "chunk_id", "text")})
    return documents, chunks, vectors
