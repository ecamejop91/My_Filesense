from __future__ import annotations

import sqlite3
import hashlib
import json
from pathlib import Path

from document_database import DEFAULT_DB_PATH, PROJECT_DIR, connect_database, load_document_chunks
from document_search_engine import extract_document_chunks, ENCODING_NAME
from model_runtime import get_model, category_vectors, canonical_model_name
import tiktoken
from document_formats import is_supported


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TAG_PROTOTYPES = {
    "Travel": "airline ticket, boarding pass, hotel reservation, itinerary, rental car, trip confirmation",
    "Finance": "bank statement, transaction, invoice, payment, tax form, receipt, account balance",
    "Legal": "contract, agreement, terms, legal notice, court, liability, signature",
    "Government": "government form, benefits, immigration, DMV, official notice, ID document",
    "Health": "medical report, lab results, prescription, clinic, diagnosis, patient, appointment",
    "Education": "transcript, course syllabus, certificate, assignment, grades, school document",
    "Work": "work report, proposal, meeting notes, company document, policy, performance",
    "Shopping": "order confirmation, shipping, delivery, return, warranty, purchase details",
    "Bills": "utility bill, rent, phone bill, internet bill, due date, amount due",
    "Insurance": "policy, coverage, premium, deductible, claim, adjuster, quote",
    "Personal": "personal letter, resume, ID copy, family document, miscellaneous",
    "Property": "lease, mortgage, HOA, property tax, title, deed, landlord",
    "Events": "event ticket, registration, RSVP, conference pass, admission",
    "Technical": "manual, specification, diagram, instructions, troubleshooting, engineering",
    "Marketing": "promotion, brochure, offer, newsletter, advertisement, campaign",
    "Professional": "resume, cover letter, recommendation, portfolio, certification, contract proposal, business correspondence, employment verification, offer letter, NDA, invoice professional, statement of work, project report, meeting minutes",
}

TAG_COLOR_PALETTE = [
    "#0f766e",
    "#dc2626",
    "#2563eb",
    "#9333ea",
    "#ca8a04",
    "#16a34a",
    "#db2777",
    "#ea580c",
    "#4f46e5",
    "#0891b2",
    "#65a30d",
    "#be123c",
    "#7c3aed",
    "#047857",
    "#b45309",
    "#1d4ed8",
]


class TaggingDependencyError(RuntimeError):
    """Raised when optional tagging dependencies are not installed."""


class DocumentTagger:
    """Assign semantic categories to extracted document text."""

    def __init__(self, model_name: str = DEFAULT_MODEL, db_path: Path = DEFAULT_DB_PATH, model=None, definitions=None) -> None:
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise TaggingDependencyError(
                "NumPy is required for tagging. Install it with 'pip install numpy'."
            ) from exc

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise TaggingDependencyError(
                "sentence-transformers is required for tagging. "
                "Install it with 'pip install sentence-transformers'."
            ) from exc

        self._np = np
        self._model = model if model is not None else get_model(model_name, SentenceTransformer)
        self._tag_definitions = definitions if definitions is not None else load_tag_definitions(db_path)
        self._tags = list(self._tag_definitions.keys())
        self._tag_vectors = category_vectors(self._model, self._tag_definitions)

    def top_tags_many(self, document_texts, top_k=3):
        if top_k <= 0:
            return [[] for _ in document_texts]
        vectors = self._model.encode([text.strip()[:12000] for text in document_texts], normalize_embeddings=True)
        return [[dict(tag=self._tags[i], score=float(scores[i]))
                 for i in self._np.argsort(-scores)[:top_k]]
                for scores in vectors @ self._tag_vectors.T]

    def top_tags_from_vectors(self, chunk_matrices, top_k=3):
        """Categorize documents from their stored chunk embeddings (no model call).

        The document vector is the normalized mean of its normalized chunk
        vectors, so it reflects the whole document rather than only the first
        ~256 tokens a direct encode would see.
        """
        np = self._np
        results = []
        for matrix in chunk_matrices:
            matrix = np.asarray(matrix, dtype=np.float32)
            if top_k <= 0 or matrix.ndim != 2 or not len(matrix):
                results.append([])
                continue
            mean = matrix.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if not np.isfinite(norm) or norm == 0.0:
                results.append([])
                continue
            scores = self._tag_vectors @ (mean / norm)
            results.append([dict(tag=self._tags[i], score=float(scores[i])) for i in np.argsort(-scores)[:top_k]])
        return results


def extract_text(pdf_path: Path) -> str:
    _, page_texts, _ = extract_document_chunks(pdf_path, tiktoken.get_encoding(ENCODING_NAME))
    return "\n".join(text for _, text in page_texts)


def connect_tag_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    connection = connect_database(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS file_tags (
            file_name TEXT NOT NULL,
            file_path TEXT PRIMARY KEY,
            char_count INTEGER NOT NULL,
            tag_1 TEXT,
            score_1 REAL,
            tag_2 TEXT,
            score_2 REAL,
            tag_3 TEXT,
            score_3 REAL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tag_definitions (
            tag_name TEXT PRIMARY KEY,
            keywords TEXT NOT NULL
        )
        """
    )
    # Older versions keyed assignments by filename, which merged files from different folders.
    columns = connection.execute("PRAGMA table_info(file_tags)").fetchall()
    if any(column["name"] == "file_name" and column["pk"] for column in columns):
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("ALTER TABLE file_tags RENAME TO legacy_file_tags")
                connection.execute("""
                    CREATE TABLE file_tags (
                        file_name TEXT NOT NULL, file_path TEXT PRIMARY KEY,
                        char_count INTEGER NOT NULL, tag_1 TEXT, score_1 REAL,
                        tag_2 TEXT, score_2 REAL, tag_3 TEXT, score_3 REAL
                    )
                """)
                for row in connection.execute("SELECT * FROM legacy_file_tags").fetchall():
                    path = Path(row["file_path"]).expanduser()
                    if not path.is_absolute():
                        path = PROJECT_DIR / path
                    connection.execute(
                        "INSERT INTO file_tags VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (row["file_name"], str(path.resolve()), row["char_count"],
                         row["tag_1"], row["score_1"], row["tag_2"], row["score_2"],
                         row["tag_3"], row["score_3"]),
                    )
                connection.execute("DROP TABLE legacy_file_tags")
        except Exception:
            connection.close()
            raise
    connection.commit()
    return connection


def normalize_tag_name(tag_name: str) -> str:
    return " ".join(tag_name.strip().split())


def load_tag_definitions(db_path: Path = DEFAULT_DB_PATH) -> dict[str, str]:
    connection = connect_tag_db(db_path)
    try:
        definitions = dict(TAG_PROTOTYPES)
        rows = connection.execute(
            """
            SELECT tag_name, keywords
            FROM tag_definitions
            ORDER BY tag_name COLLATE NOCASE
            """
        ).fetchall()
        for row in rows:
            tag_name = normalize_tag_name(row["tag_name"])
            keywords = row["keywords"].strip()
            if tag_name and keywords:
                definitions[tag_name] = keywords
        return definitions
    finally:
        connection.close()


def tag_names(db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    return list(load_tag_definitions(db_path=db_path).keys())


def save_tag_definition(
    tag_name: str,
    keywords: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> str:
    clean_name = normalize_tag_name(tag_name)
    clean_keywords = " ".join(keywords.replace("\n", ", ").split())
    if not clean_name:
        raise ValueError("Enter a tag name.")
    if not clean_keywords:
        raise ValueError("Enter keywords or a short definition for the tag.")

    connection = connect_tag_db(db_path)
    try:
        connection.execute(
            """
            INSERT INTO tag_definitions (tag_name, keywords)
            VALUES (?, ?)
            ON CONFLICT(tag_name) DO UPDATE SET
                keywords=excluded.keywords
            """,
            (clean_name, clean_keywords),
        )
        connection.commit()
        return clean_name
    finally:
        connection.close()


def tag_color(tag_name: str, available_tags: list[str] | None = None) -> str:
    if available_tags is None:
        available_tags = tag_names()

    normalized = tag_name.casefold()
    index = next(
        (
            tag_index
            for tag_index, available_tag in enumerate(available_tags)
            if available_tag.casefold() == normalized
        ),
        len(available_tags),
    )
    return TAG_COLOR_PALETTE[index % len(TAG_COLOR_PALETTE)]


def save_file_tags(
    file_path: Path,
    tags: list[dict[str, float | str]],
    char_count: int,
    connection: sqlite3.Connection,
    commit: bool = True,
) -> None:
    file_path = Path(file_path).expanduser().resolve()
    padded = list(tags[:3])
    while len(padded) < 3:
        padded.append({"tag": None, "score": None})

    connection.execute(
        """
        INSERT INTO file_tags (
            file_name,
            file_path,
            char_count,
            tag_1,
            score_1,
            tag_2,
            score_2,
            tag_3,
            score_3
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            file_name=excluded.file_name,
            char_count=excluded.char_count,
            tag_1=excluded.tag_1,
            score_1=excluded.score_1,
            tag_2=excluded.tag_2,
            score_2=excluded.score_2,
            tag_3=excluded.tag_3,
            score_3=excluded.score_3
        """,
        (
            file_path.name,
            str(file_path),
            char_count,
            padded[0]["tag"],
            padded[0]["score"],
            padded[1]["tag"],
            padded[1]["score"],
            padded[2]["tag"],
            padded[2]["score"],
        ),
    )
    if commit:
        connection.commit()


def prepare_tagging(file_paths, db_path=DEFAULT_DB_PATH, model_name=DEFAULT_MODEL, top_k=3):
    definitions = load_tag_definitions(db_path)
    signature = hashlib.sha256(json.dumps(
        # Version 2: categories come from the mean of stored chunk embeddings.
        [2, canonical_model_name(model_name), top_k, list(definitions.items())], ensure_ascii=False
    ).encode()).hexdigest()
    connection = connect_tag_db(db_path)
    try:
        states = {str(Path(row['pdf_path']).resolve()): dict(row) for row in connection.execute(
            'SELECT p.*, d.pdf_path FROM document_processing p JOIN documents d USING(doc_id)')}
        assignments = {str(Path(row['file_path']).resolve()): dict(row)
                       for row in connection.execute('SELECT * FROM file_tags')}
    finally:
        connection.close()
    cached = {}
    unchanged = set()
    for path, state in states.items():
        row = assignments.get(path)
        if state['tag_signature'] == signature and row is not None:
            cached[state['content_sha256']] = dict(
                tags=[dict(tag=row[f'tag_{i}'], score=row[f'score_{i}']) for i in range(1, 4) if row[f'tag_{i}']],
                char_count=row['char_count'])
            unchanged.add(path)
    needed = set()
    for path in file_paths:
        key = str(Path(path).expanduser().resolve())
        state = states.get(key)
        if key not in unchanged and not (state and state['content_sha256'] in cached):
            needed.add(key)
    return dict(definitions=definitions, signature=signature, states=states,
                cached=cached, unchanged=unchanged, needed=needed)


def load_tagging_vectors(plan, db_path: Path = DEFAULT_DB_PATH):
    """Stored chunk embeddings and char counts for the files that need tags.

    Returns {resolved path: dict(vectors=matrix, char_count=int)} for needed
    paths that are in the index; reads only those documents' rows.
    """
    states = plan.get('states', {})
    wanted = {path: states[path]['doc_id'] for path in plan.get('needed', ()) if path in states}
    if not wanted:
        return {}
    db_path = Path(db_path)
    chunks = load_document_chunks(db_path.parent, wanted.values())
    connection = connect_database(db_path)
    try:
        counts = {}
        ids = list(dict.fromkeys(wanted.values()))
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            counts.update((row['doc_id'], row['char_count']) for row in connection.execute(
                f"SELECT doc_id, char_count FROM documents WHERE doc_id IN ({', '.join('?' * len(batch))})", batch))
    finally:
        connection.close()
    return {path: dict(vectors=chunks[doc_id][1], char_count=counts.get(doc_id, 0))
            for path, doc_id in wanted.items()}


def tag_files(
    file_paths,
    db_path: Path = DEFAULT_DB_PATH,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 3,
    texts_by_path: dict[str, str] | None = None,
    progress=None,
    model=None,
    plan=None,
    vectors_by_path=None,
):
    """Assign categories to PDFs.

    vectors_by_path ({path: dict(vectors, char_count)}, see load_tagging_vectors)
    categorizes from stored chunk embeddings without encoding any text; paths
    not in it fall back to texts_by_path, then to extracting the PDF text.
    """
    file_paths = list(dict.fromkeys(Path(p).expanduser().resolve() for p in file_paths))
    plan = plan or prepare_tagging(file_paths, db_path, model_name, top_k)
    tagged, skipped, work = [], [], []
    predictions, unique_texts, unique_vectors = {}, {}, {}
    for path in file_paths:
        key = str(path)
        if not is_supported(path):
            skipped.append((path, 'unsupported'))
            if progress:
                progress(dict(stage='skipped', path=key, detail='Unsupported file type'))
            continue
        if key in plan['unchanged']:
            skipped.append((path, 'unchanged categories'))
            if progress:
                progress(dict(stage='done', path=key, detail='Unchanged; using saved index and categories'))
            continue
        state = plan['states'].get(key)
        if state and state['content_sha256'] in plan['cached']:
            token = state['content_sha256']
            predictions[token] = plan['cached'][token]['tags']
            char_count = plan['cached'][token]['char_count']
        elif vectors_by_path is not None and key in vectors_by_path:
            stored = vectors_by_path[key]
            vectors = stored['vectors']
            char_count = stored['char_count']
            # Identical content shares one prediction, like the text path.
            token = ('content:' + state['content_sha256'] if state
                     else 'vectors:' + hashlib.sha256(vectors.tobytes()).hexdigest())
            if len(vectors):
                unique_vectors.setdefault(token, vectors)
            else:
                predictions[token] = []
        else:
            text = (texts_by_path[key] if texts_by_path is not None and key in texts_by_path else extract_text(path))
            token = hashlib.sha256(text.strip()[:12000].encode()).hexdigest()
            char_count = len(text)
            if text.strip():
                unique_texts.setdefault(token, text)
            else:
                predictions[token] = []
        work.append((path, token, char_count, state))
    pending = [(token, text) for token, text in unique_texts.items() if token not in predictions]
    pending_vectors = [(token, vectors) for token, vectors in unique_vectors.items() if token not in predictions]
    tagger = None
    if pending or pending_vectors:
        if progress:
            progress(dict(stage='tagging', detail='Preparing cached model and category descriptions'))
        tagger = DocumentTagger(model_name=model_name, db_path=db_path, model=model, definitions=plan['definitions'])
    if pending_vectors:
        if progress:
            for path, token, _, _ in work:
                if token in unique_vectors:
                    progress(dict(stage='tagging', path=str(path),
                                  detail='Categorizing from stored chunk embeddings'))
        values = tagger.top_tags_from_vectors([vectors for _, vectors in pending_vectors], top_k=top_k)
        predictions.update((token, value) for (token, _), value in zip(pending_vectors, values))
    if pending:
        for start in range(0, len(pending), 32):
            batch = pending[start:start + 32]
            batch_tokens = {token for token, _ in batch}
            if progress:
                for path, token, _, _ in work:
                    if token in batch_tokens:
                        progress(dict(stage='tagging', path=str(path),
                                      detail=f'Categorizing batch of {len(batch)} unique document texts'))
            values = tagger.top_tags_many([text for _, text in batch], top_k=top_k)
            predictions.update((token, value) for (token, _), value in zip(batch, values))
    if work:
        connection = connect_tag_db(db_path)
        try:
            with connection:
                for path, token, count, state in work:
                    if progress:
                        progress(dict(stage='tagging', path=str(path), detail='Saving categories'))
                    values = predictions[token]
                    save_file_tags(path, values, count, connection, commit=False)
                    if state:
                        connection.execute('UPDATE document_processing SET tag_signature=? WHERE doc_id=?',
                                           (plan['signature'], state['doc_id']))
                    if values:
                        tagged.append(dict(file_name=path.name, file_path=str(path), char_count=count, tags=values))
                    else:
                        skipped.append((path, 'no extractable text'))
        finally:
            connection.close()
        # Only mark a file complete after the category transaction commits.
        if progress:
            for path, _, _, _ in work:
                progress(dict(stage='done', path=str(path), detail='Indexed and categorized'))
    return dict(tagged=tagged, skipped=skipped)


def load_tag_rows(db_path: Path = DEFAULT_DB_PATH):
    connection = connect_tag_db(db_path)
    try:
        rows = connection.execute(
            """
            SELECT file_name, file_path, char_count, tag_1, score_1, tag_2, score_2, tag_3, score_3
            FROM file_tags
            ORDER BY file_name COLLATE NOCASE
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def exact_tag_matches(query: str, db_path: Path = DEFAULT_DB_PATH):
    normalized = query.strip().casefold()
    matched_tag = next((tag for tag in tag_names(db_path=db_path) if tag.casefold() == normalized), None)
    if not matched_tag:
        return []

    connection = connect_tag_db(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM file_tags WHERE tag_1 = ? OR tag_2 = ? OR tag_3 = ?",
            (matched_tag, matched_tag, matched_tag),
        ).fetchall()
    finally:
        connection.close()
    results = []
    for row in rows:
        entries = [
            (row["tag_1"], row["score_1"]),
            (row["tag_2"], row["score_2"]),
            (row["tag_3"], row["score_3"]),
        ]
        for tag_name, score in entries:
            if tag_name == matched_tag and score is not None:
                results.append(
                    {
                        "file_name": row["file_name"],
                        "file_path": row["file_path"],
                        "matched_tag": matched_tag,
                        "score": float(score),
                        "all_tags": [tag for tag, _ in entries if tag],
                    }
                )
                break

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def delete_file_tags(file_path: Path, db_path: Path = DEFAULT_DB_PATH) -> None:
    connection = connect_tag_db(db_path)
    try:
        connection.execute("DELETE FROM file_tags WHERE file_path = ?", (str(Path(file_path).expanduser().resolve()),))
        connection.commit()
    finally:
        connection.close()
