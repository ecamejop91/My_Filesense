"""Explicit, non-destructive import of a library into an empty account library."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile

from app_storage import PATHS, private_directory, protect_path
from document_database import DATABASE_NAME, DEFAULT_DB_PATH, load_index


TABLES = ('documents', 'chunks', 'index_metadata', 'document_processing', 'file_tags', 'tag_definitions')
EXPECTED_COLUMNS = {
    'documents': ['doc_id', 'pdf_path', 'method', 'char_count'],
    'chunks': ['position', 'doc_id', 'page', 'chunk_id', 'text', 'embedding'],
    'index_metadata': ['id', 'model_name', 'vector_dimension'],
    'document_processing': ['doc_id', 'content_sha256', 'file_size', 'mtime_ns', 'indexing_signature', 'tag_signature'],
    'file_tags': ['file_name', 'file_path', 'char_count', 'tag_1', 'score_1', 'tag_2', 'score_2', 'tag_3', 'score_3'],
    'tag_definitions': ['tag_name', 'keywords'],
}


def import_library(source, destination=DEFAULT_DB_PATH):
    from document_tag_store import connect_tag_db

    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).absolute()
    if source == destination.resolve():
        raise ValueError('Choose a different existing library to import.')
    with tempfile.TemporaryDirectory(prefix='library-import-', dir=private_directory(PATHS.temporary)) as scratch:
        snapshot = Path(scratch) / DATABASE_NAME
        with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as reader:
            reader.execute('PRAGMA query_only=ON')
            with closing(sqlite3.connect(snapshot)) as copy:
                reader.backup(copy)
        protect_path(snapshot)
        with closing(sqlite3.connect(snapshot)) as copy:
            if copy.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('The selected library is damaged.')
            if copy.execute("SELECT 1 FROM sqlite_master WHERE type IN ('trigger', 'view')").fetchone():
                raise ValueError('The selected library has unsupported triggers or views.')
            tables = {r[0] for r in copy.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'documents', 'chunks', 'index_metadata'} <= tables or not tables <= set(TABLES):
                raise ValueError('Choose a FileSense library database (file_tags.sqlite3).')
            # Earlier portable libraries could store paths relative to the old
            # application folder. Resolve these before moving the index.
            with copy:
                for table, column in (('documents', 'pdf_path'), ('file_tags', 'file_path')):
                    if table not in tables:
                        continue
                    for rowid, raw in copy.execute(f'SELECT rowid, {column} FROM {table}').fetchall():
                        path = Path(raw).expanduser()
                        if not path.is_absolute():
                            path = source.parent.parent / path
                        copy.execute(f'UPDATE {table} SET {column}=? WHERE rowid=?', (str(path.resolve()), rowid))
        # Validate model metadata and every stored vector, and migrate only the
        # private snapshot's known older schema. The original is never rewritten.
        documents, chunks, _ = load_index(Path(scratch))
        with closing(connect_tag_db(snapshot)) as copy:
            records = {table: copy.execute(f'SELECT * FROM {table}').fetchall() for table in TABLES}
            columns = {table: [r[1] for r in copy.execute(f'PRAGMA table_info({table})')] for table in TABLES}
            if columns != EXPECTED_COLUMNS:
                raise ValueError('The selected library has an unsupported database schema.')
        with closing(connect_tag_db(destination)) as target:
            with target:
                target.execute('BEGIN IMMEDIATE')
                for table in ('documents', 'chunks', 'file_tags', 'tag_definitions'):
                    if target.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone():
                        raise ValueError('Your account library already contains data. Import requires an empty library.')
                target.execute('DELETE FROM document_processing')
                target.execute('DELETE FROM index_metadata')
                for table in TABLES:
                    names = ', '.join(columns[table])
                    marks = ', '.join('?' for _ in columns[table])
                    target.executemany(f'INSERT INTO {table} ({names}) VALUES ({marks})', records[table])
        return dict(documents=len(documents), chunks=len(chunks))
