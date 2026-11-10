from __future__ import annotations

from pathlib import Path
import json
import hashlib

import pymupdf as fitz
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from ocr import DEFAULT_OCR_MODEL, extract_page_texts_with_ocr, ocr_signature, validate_ocr_model
from document_formats import extract_office_texts, is_supported, location_label, progress_unit

from document_database import (
    FILES_DIR, PROJECT_DIR, VECTOR_STORE_DIR, load_index, save_index,
    load_processing_states, apply_index_updates, edit_indexed_document,
    load_index_summary, load_document_chunks,
)
from model_runtime import get_model, canonical_model_name

MODEL_NAME = "all-MiniLM-L6-v2"
ENCODING_NAME = "cl100k_base"
# Fixed, non-overlapping chunks selected for top-three passage retrieval
# in the HP Prime manual paraphrase study.
CHUNK_SIZE = 50
# Chunks from all files being analyzed are embedded together in batches this size.
EMBED_BATCH_SIZE = 64
TOP_K = 5
INCLUDE_CHECKPOINTS = False


def canonical_pdf_key(path: Path) -> str:
    parts = [part for part in path.parts if part != ".ipynb_checkpoints"]
    normalized = Path(*parts)
    if normalized.stem.endswith("-checkpoint"):
        normalized = normalized.with_name(
            normalized.stem.removesuffix("-checkpoint") + normalized.suffix
        )
    return normalized.as_posix().lower()


def iter_document_files(folder: Path, include_checkpoints: bool = False):
    seen = set()
    for pdf_path in sorted(folder.rglob('*')):
        if not pdf_path.is_file() or not is_supported(pdf_path) or pdf_path.name.startswith('~$'):
            continue
        if not include_checkpoints:
            if ".ipynb_checkpoints" in pdf_path.parts:
                continue
            if pdf_path.stem.endswith('-checkpoint'):
                continue

        key = canonical_pdf_key(pdf_path)
        if key in seen:
            continue

        seen.add(key)
        yield pdf_path


# Keep notebook/script callers compatible with the original entry point.
iter_pdf_files = iter_document_files


def extract_page_texts(pdf_path: Path, progress=None):
    if pdf_path.suffix.lower() != '.pdf':
        return extract_office_texts(pdf_path, progress)
    doc = fitz.open(str(pdf_path))
    try:
        if doc.needs_pass:
            return "encrypted", []

        page_texts = []
        for page_number, page in enumerate(doc, start=1):
            if progress:
                progress(dict(stage='extracting', path=str(pdf_path), current=page_number - 1,
                              total=len(doc), detail=f'Reading page {page_number} of {len(doc)}'))
            text = page.get_text("text").strip()
            if text:
                page_texts.append((page_number, text))
        if progress:
            progress(dict(stage='extracting', path=str(pdf_path), current=len(doc), total=len(doc),
                          detail=f'Read {len(doc)} pages'))

        if page_texts:
            return "text", page_texts
        return "none", []
    finally:
        doc.close()


def chunk_page_text(tokenizer, page_number: int, text: str, chunk_size: int):
    token_ids = tokenizer.encode(text)
    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start : start + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids).strip()
        if chunk_text:
            yield {
                "page": page_number,
                "text": chunk_text,
            }


def extract_document_chunks(pdf_path: Path, tokenizer, chunk_size: int = CHUNK_SIZE, progress=None,
                            *, ocr_model=DEFAULT_OCR_MODEL):
    """Try embedded text first; OCR only when the entire PDF produces no chunks."""
    validate_ocr_model(ocr_model)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    method, page_texts = (extract_page_texts(pdf_path, progress=progress) if progress
                         else extract_page_texts(pdf_path))

    def split_pages(pages):
        chunks = []
        for index, (number, text) in enumerate(pages):
            if progress:
                progress(dict(stage='chunking', path=str(pdf_path), current=index, total=len(pages),
                              unit=progress_unit(pdf_path),
                              detail=f'Splitting {location_label(pdf_path, number).lower()}; {len(chunks)} chunks created'))
            chunks.extend(chunk_page_text(tokenizer, number, text, chunk_size))
        if progress:
            progress(dict(stage='chunking', path=str(pdf_path), current=len(pages), total=len(pages),
                          unit=progress_unit(pdf_path), detail=f'{len(chunks)} chunks created'))
        return chunks

    chunks = split_pages(page_texts)
    if not chunks and method != "encrypted" and pdf_path.suffix.lower() == '.pdf':
        print(f"No text chunks found. Running OCR: {pdf_path}", flush=True)
        page_texts = extract_page_texts_with_ocr(pdf_path, progress=progress, ocr_model=ocr_model)
        chunks = split_pages(page_texts)
        method = "ocr" if chunks else "none"
    return method, page_texts, chunks


def next_doc_index(doc_records) -> int:
    highest = -1
    for record in doc_records:
        doc_id = record.get("doc_id", "")
        if doc_id.startswith("doc_"):
            try:
                highest = max(highest, int(doc_id.split("_", 1)[1]))
            except ValueError:
                continue
    return highest + 1


def build_docs_by_id(doc_records):
    return {record["doc_id"]: record for record in doc_records}


def load_vector_store(
    vector_store_dir: Path = VECTOR_STORE_DIR,
    model_name: str = MODEL_NAME,
):
    doc_records, chunk_records, chunk_vectors = load_index(vector_store_dir, model_name)
    model = get_embedding_model(model_name)
    dimension = model.get_sentence_embedding_dimension()
    if len(chunk_vectors) and chunk_vectors.shape[1] != dimension:
        raise ValueError("Stored embeddings do not match the model dimension. Rebuild the index.")
    if not len(chunk_vectors):
        chunk_vectors = np.empty((0, dimension), dtype=np.float32)

    docs_by_id = build_docs_by_id(doc_records)

    return {
        "model": model,
        "doc_records": doc_records,
        "chunk_records": chunk_records,
        "chunk_vectors": chunk_vectors,
        "docs_by_id": docs_by_id,
    }


def build_index(
    files_dir: Path = FILES_DIR,
    model_name: str = MODEL_NAME,
    encoding_name: str = ENCODING_NAME,
    chunk_size: int = CHUNK_SIZE,
    include_checkpoints: bool = INCLUDE_CHECKPOINTS,
    ocr_model=DEFAULT_OCR_MODEL,
):
    validate_ocr_model(ocr_model)
    files_dir = Path(files_dir).expanduser().resolve()
    tokenizer = tiktoken.get_encoding(encoding_name)
    model = get_embedding_model(model_name)

    doc_records = []
    chunk_records = []

    for doc_index, pdf_path in enumerate(iter_document_files(files_dir, include_checkpoints)):
        method, page_texts, chunks = extract_document_chunks(pdf_path, tokenizer, chunk_size,
                                                           ocr_model=ocr_model)
        doc_id = f"doc_{doc_index:05d}"
        char_count = sum(len(text) for _, text in page_texts)

        doc_records.append(
            {
                "doc_id": doc_id,
                "pdf_path": pdf_path.as_posix(),
                "method": method,
                "char_count": char_count,
            }
        )

        for chunk_id, chunk in enumerate(chunks):
            chunk_records.append({
                "doc_id": doc_id,
                "pdf_path": pdf_path.as_posix(),
                "page": chunk["page"],
                "chunk_id": chunk_id,
                "text": chunk["text"],
            })

    print(f"Documents scanned: {len(doc_records)}")
    print(f"Text chunks created: {len(chunk_records)}")
    print(
        "Methods:",
        {
            "text": sum(record["method"] == "text" for record in doc_records),
            "ocr": sum(record["method"] == "ocr" for record in doc_records),
            "none": sum(record["method"] == "none" for record in doc_records),
            "encrypted": sum(record["method"] == "encrypted" for record in doc_records),
        },
    )

    chunk_texts = [record["text"] for record in chunk_records]
    chunk_vectors = (model.encode(chunk_texts, normalize_embeddings=True) if chunk_texts
                     else np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32))
    chunk_vectors = np.asarray(chunk_vectors, dtype=np.float32)

    docs_by_id = build_docs_by_id(doc_records)

    print("Embedding matrix shape:", chunk_vectors.shape)
    print("Example document:", doc_records[0] if doc_records else "No documents found")

    return {
        "tokenizer": tokenizer,
        "model": model,
        "doc_records": doc_records,
        "chunk_records": chunk_records,
        "chunk_vectors": chunk_vectors,
        "docs_by_id": docs_by_id,
    }


def save_vector_store(
    doc_records,
    chunk_records,
    chunk_vectors,
    vector_store_dir: Path = VECTOR_STORE_DIR,
    model_name: str = MODEL_NAME,
):
    save_index(doc_records, chunk_records, chunk_vectors, vector_store_dir, model_name)
    print(f"Saved SQLite search index to {vector_store_dir}")


def remove_document_from_vector_store(file_path: Path, vector_store_dir: Path = VECTOR_STORE_DIR, model_name: str = MODEL_NAME):
    return edit_indexed_document(file_path, vector_store_dir, model_name)


def rename_document_in_vector_store(old_path: Path, new_path: Path, vector_store_dir: Path = VECTOR_STORE_DIR, model_name: str = MODEL_NAME):
    return edit_indexed_document(old_path, vector_store_dir, model_name, new_path)


def get_embedding_model(model_name=MODEL_NAME):
    return get_model(model_name, SentenceTransformer)


def _fingerprint(path, progress=None):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            if progress:
                progress(dict(stage='checking', path=str(path), detail='Checking file content for changes or duplicates'))
            digest.update(block)
    return digest.hexdigest()


def _stored_path_key(pdf_path):
    """Lookup key for a stored document path without touching the filesystem.

    Stored paths are already absolute and resolved (new analyses and the legacy
    import both resolve them), so a per-document resolve() is unnecessary;
    only rare relative legacy paths are joined to the project and resolved.
    """
    path = Path(pdf_path)
    return str(path) if path.is_absolute() else str((PROJECT_DIR / path).resolve())


def _encode_resources(resources, model_name, dimension, progress, batch_size=None):
    """Embed every newly extracted resource in shared batches across files."""
    batch_size = batch_size or EMBED_BATCH_SIZE
    pending = [(record, resource) for record, resource in resources if resource['vectors'] is None]
    owners, texts = [], []
    for record, resource in pending:
        for chunk in resource['chunks']:
            owners.append(record['pdf_path'])
            texts.append(chunk['text'])
    model = None
    if texts:
        if progress:
            progress(dict(stage='loading', path=owners[0], detail='Preparing shared embedding model'))
        model = get_embedding_model(model_name)
        dimension = model.get_sentence_embedding_dimension()
    parts = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        if progress:
            progress(dict(stage='embedding', path=owners[start], current=start, total=len(texts),
                          detail=f'Embedding chunks {start + 1}-{end} of {len(texts)} (all new files)'))
        parts.append(np.asarray(model.encode(texts[start:end], normalize_embeddings=True), dtype=np.float32))
        if progress:
            progress(dict(stage='embedding', path=owners[end - 1], current=end, total=len(texts),
                          detail=f'Embedded {end} of {len(texts)} chunks'))
    vectors = np.concatenate(parts) if parts else np.empty((0, dimension), dtype=np.float32)
    offset = 0
    for _, resource in pending:
        count = len(resource['chunks'])
        resource['vectors'] = vectors[offset:offset + count]
        offset += count
    return model, dimension


def merge_index_delta(index_data, delta):
    """Apply an incremental update result to an already-loaded in-memory index."""
    changed_docs = delta['added_doc_records']
    if not changed_docs:
        return index_data
    changed_ids = {d['doc_id'] for d in changed_docs}
    old_chunks, old_vectors = index_data['chunk_records'], index_data['chunk_vectors']
    kept = [i for i, c in enumerate(old_chunks) if c['doc_id'] not in changed_ids]
    new_vectors = delta['added_chunk_vectors']
    kept_vectors = old_vectors[kept] if kept else None
    if kept_vectors is None or not len(kept_vectors):
        vectors = np.asarray(new_vectors, dtype=np.float32)
    elif not len(new_vectors):
        vectors = kept_vectors
    else:
        vectors = np.concatenate([kept_vectors, new_vectors]).astype(np.float32)
    replacements = {d['doc_id']: d for d in changed_docs}
    old_ids = {d['doc_id'] for d in index_data['doc_records']}
    docs = [replacements.get(d['doc_id'], d) for d in index_data['doc_records']]
    docs.extend(d for d in changed_docs if d['doc_id'] not in old_ids)
    return dict(index_data, doc_records=docs,
                chunk_records=[old_chunks[i] for i in kept] + list(delta['added_chunk_records']),
                chunk_vectors=vectors, docs_by_id=build_docs_by_id(docs),
                model=index_data.get('model') or delta.get('model'))


def update_vector_store(
    file_paths,
    vector_store_dir: Path = VECTOR_STORE_DIR,
    model_name: str = MODEL_NAME,
    encoding_name: str = ENCODING_NAME,
    chunk_size: int = CHUNK_SIZE,
    progress=None,
    full_result=True,
    ocr_model=DEFAULT_OCR_MODEL,
):
    """Index new or changed documents, reading only what those files need.

    The existing index is never loaded in full: unchanged files are decided
    from stored fingerprints, identical content reuses just that document's
    stored chunks, and only changed documents are written.

    With full_result=True (default, for scripts) the whole index is read back
    once after saving and returned. The GUI passes full_result=False and gets
    doc_records/chunk_records/chunk_vectors set to None; it merges the
    added_* delta into its loaded index or loads lazily at the next search.
    """
    validate_ocr_model(ocr_model)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if progress:
        progress(dict(stage='loading', detail='Checking the saved index; unchanged files need no model work'))
    summary = load_index_summary(vector_store_dir, model_name)
    doc_records = summary['documents']
    states = load_processing_states(vector_store_dir)
    def signature_for(path):
        suffix = Path(path).suffix.lower()
        extractor = [ocr_signature(ocr_model)] if suffix == '.pdf' else ['office-text-v1', suffix]
        return json.dumps([2, canonical_model_name(model_name), encoding_name, chunk_size, *extractor])
    by_path = {_stored_path_key(d['pdf_path']): d for d in doc_records}
    # Equal bytes may reuse extraction/OCR and vectors, while keeping separate
    # searchable document records and paths for every selected PDF. Only the
    # fingerprints are known up front; chunks load on demand.
    stored_by_content = {}
    for doc in doc_records:
        state = states.get(doc['doc_id'])
        if state and state['indexing_signature'] == signature_for(doc['pdf_path']):
            stored_by_content.setdefault((state['content_sha256'], state['indexing_signature']), doc)
    resources = {}
    tokenizer = None
    jobs, state_updates, skipped_paths, processed_paths = [], [], [], []
    next_index = next_doc_index(doc_records)
    attempted = set()
    for raw_path in file_paths:
        path = Path(raw_path).expanduser().resolve()
        if progress:
            progress(dict(stage='checking', path=str(path), detail='Checking for changes'))
        if not is_supported(path):
            skipped_paths.append((path, 'unsupported'))
            if progress:
                progress(dict(stage='skipped', path=str(path), detail='Unsupported file type'))
            continue
        if path in attempted:
            skipped_paths.append((path, 'already indexed'))
            continue
        attempted.add(path)
        signature = signature_for(path)
        existing = by_path.get(str(path))
        state = states.get(existing['doc_id']) if existing else None
        stat = path.stat()
        same_settings = state and state['indexing_signature'] == signature
        if same_settings and (state['file_size'], state['mtime_ns']) == (stat.st_size, stat.st_mtime_ns):
            skipped_paths.append((path, 'already indexed'))
            if progress:
                progress(dict(stage='skipped', path=str(path), detail='Unchanged; using saved index'))
            continue
        digest = _fingerprint(path, progress)
        if same_settings and state['content_sha256'] == digest:
            state_updates.append(dict(state, file_size=stat.st_size, mtime_ns=stat.st_mtime_ns))
            skipped_paths.append((path, 'unchanged content'))
            if progress:
                progress(dict(stage='skipped', path=str(path), detail='Contents unchanged; using saved index'))
            continue
        key = (digest, signature)
        resource = resources.get(key)
        if resource is None and key in stored_by_content:
            source = stored_by_content[key]
            chunks, vectors = load_document_chunks(vector_store_dir, [source['doc_id']])[source['doc_id']]
            resource = dict(method=source['method'], char_count=source['char_count'],
                            chunks=chunks, vectors=vectors)
            resources[key] = resource
            if progress:
                progress(dict(stage='reusing', path=str(path), detail='Identical document content; reusing extracted text and embeddings'))
        elif resource is None:
            if tokenizer is None:
                tokenizer = tiktoken.get_encoding(encoding_name)
            method, pages, chunks = extract_document_chunks(path, tokenizer, chunk_size,
                                                           progress=progress, ocr_model=ocr_model)
            resource = dict(method=method, char_count=sum(len(t) for _, t in pages),
                            chunks=chunks, vectors=None)
            resources[key] = resource
        elif progress:
            progress(dict(stage='reusing', path=str(path), detail='Identical document content; reusing extracted text and embeddings'))
        # Never save a result against a fingerprint of a file edited mid-analysis.
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError(f'{path.name} changed during analysis. Retry after saving the file.')
        doc_id = existing['doc_id'] if existing else f'doc_{next_index:05d}'
        if not existing:
            next_index += 1
        record = dict(doc_id=doc_id, pdf_path=path.as_posix(), method=resource['method'], char_count=resource['char_count'])
        jobs.append((record, resource))
        processed_paths.append(path)
        state_updates.append(dict(doc_id=doc_id, content_sha256=digest, file_size=stat.st_size,
                                  mtime_ns=stat.st_mtime_ns, indexing_signature=signature, tag_signature=None))
    # Encode only unique, newly extracted content, batched across all files.
    # Copies share the same resource object, so they share its vectors.
    unique = list({id(resource): (record, resource) for record, resource in jobs}.values())
    model, dimension = _encode_resources(unique, model_name, summary['dimension'], progress)
    changed_docs, changed_chunks, vector_parts = [], [], []
    for record, resource in jobs:
        changed_docs.append(record)
        changed_chunks.extend(dict(doc_id=record['doc_id'], pdf_path=record['pdf_path'], page=c['page'],
                                   chunk_id=i, text=c['text']) for i, c in enumerate(resource['chunks']))
        if len(resource['vectors']):
            vector_parts.append(resource['vectors'])
    changed_vectors = np.concatenate(vector_parts).astype(np.float32) if vector_parts else np.empty((0, dimension), dtype=np.float32)
    if summary['has_chunks'] and len(changed_vectors) and changed_vectors.shape[1] != summary['dimension']:
        raise ValueError('Stored embeddings do not match the model dimension. Rebuild the index.')
    if jobs or state_updates:
        if progress:
            progress(dict(stage='saving', detail=f'Saving {len(jobs)} new/changed documents; retaining the rest of the index'))
        apply_index_updates(changed_docs, changed_chunks, changed_vectors, state_updates, vector_store_dir, model_name)
    if progress:
        progress(dict(stage='indexed', detail='Index ready; unchanged content was reused'))
    result = dict(model=model, tokenizer=tokenizer,
                  added_doc_records=changed_docs, added_chunk_records=changed_chunks,
                  added_chunk_vectors=changed_vectors,
                  processed_paths=processed_paths, skipped_paths=skipped_paths)
    if full_result:
        docs, chunks, vectors = load_index(vector_store_dir, model_name)
        result.update(doc_records=docs, chunk_records=chunks, chunk_vectors=vectors,
                      docs_by_id=build_docs_by_id(docs))
    else:
        result.update(doc_records=None, chunk_records=None, chunk_vectors=None, docs_by_id=None)
    return result


def search_index(
    query: str,
    model,
    chunk_records,
    chunk_vectors,
    docs_by_id,
    top_k: int = TOP_K,
):
    if len(chunk_records) == 0 or len(chunk_vectors) == 0:
        return []

    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = chunk_vectors @ query_vector

    top_k = min(top_k, len(chunk_records))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunk_records[int(index)]
        doc = docs_by_id[chunk["doc_id"]]
        preview = " ".join(chunk["text"].split())
        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "document": Path(doc["pdf_path"]).name,
                "path": doc["pdf_path"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": preview[:700],
                # The UI selects a short passage lazily, after tag filtering and
                # the visible-result cutoff, without embedding the query again.
                "_source_text": chunk["text"],
                "_query_vector": query_vector,
            }
        )

    return results
