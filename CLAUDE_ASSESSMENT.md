# FileSense — Claude assessment: inefficiencies, redundant and unused code

Reviewed November 4, 2026, read-only, against the working tree of `main` (`e1e2353`),
including its uncommitted cleanup edits. The original review was read-only.
Obsolete subsystem references were removed from this report afterward.

## Scope

**Read in full:** `document_search_engine.py`, `document_database.py`,
`document_tag_store.py`, `document_query_visuals.py`, `analysis_progress.py`,
`search_snippets.py`, `model_runtime.py`, `app_storage.py`, `ocr.py`,
`document_formats.py`, `word_page_lookup.py`,
`indra_models.py`, `qt_runtime.py`, `visualize_vectors.py`, `bundle_indra.py`,
the launchers, both requirements files, `.gitignore`, the top of `README.md`, and the existing
`CODEBASE_AUDIT.md`.

**Not re-read in this pass:**
`tools/check_speedups.py`, `bundle_model.py`, the notebook, and `tests/`.
`CODEBASE_AUDIT.md` covers those for dead code.

**Relation to `CODEBASE_AUDIT.md`:** that audit focuses on correctness defects
(F1–F6) and confirmed-unused symbols. Most of its unused-symbol items have already
been removed in the working tree. This report doesn't repeat those findings. It
covers **performance and redundancy** issues the earlier audit didn't record. Where
the two overlap, it says so.

Line numbers refer to the current working-tree files.

---

## Summary — highest impact first

| # | Severity | Area | Problem | Scales with |
|---|---|---|---|---|
| P1 | **High** | Storage / DB | Every DB connection re-applies ACLs/permissions to the data root, every folder on the path, and the DB file. On macOS this also starts `chmod` subprocesses | Every DB call (dozens per UI action) |
| P2 | **High** | Search | Each query builds a result dict for **every chunk** in the library. A tag filter then calls `Path.resolve()` once per chunk | Total chunk count, per query and per filter toggle |
| P3 | **High** | DB access pattern | The GUI opens a new SQLite connection (DDL + migration check + P1) per helper call. Building the tag filter makes one call per tag (N+1) | Number of tags / rows |
| P4 | Medium | GUI thread | Semantic search, passage re-embedding (up to ~320 strings) and **launching Microsoft Word over COM** all run on the UI thread | Results shown / .docx results |
| P5 | Medium | Rename / remove | In-memory updates call `Path.resolve()` on every doc and **every chunk** record. The DB rename scans all documents with `resolve()` | Library size |
| P6 | Medium | Startup | `prepare_storage` walks every file under library/tmp/logs and re-secures each one on every launch | Files in app data |
| P7 | Medium | Tagging | `prepare_tagging` resolves every document path and every `file_tags` path on each analysis | Library size |
| P8 | Low–Med | DB reads | `load_index` decodes and validates embeddings row by row and holds a write lock (`BEGIN IMMEDIATE`) for a read | Chunk count |
| R1–R12 | Low | Redundancy | Duplicated loaders, schemas, path normalizers, HTML builders, constants, and a redundant second tag delete | — |

---

## Performance findings

### P1 — High: every database connection re-secures the storage tree

`document_database.connect_database` (line 22) calls `app_storage.prepare_managed_file`
on **every** connection. That function (`app_storage.py:186–202`):

1. calls `private_directory(PATHS.root)` → `protect_path`,
2. calls `private_directory()` for every path component between the root and the DB folder,
3. calls `protect_path` on the DB file and any `-wal`/`-shm`/`-journal` sidecar.

`protect_path` does real work every time:

- **Windows** (`_secure_windows`): opens the process token, reads the owner SID,
  builds a new ACL, **writes** it with `SetNamedSecurityInfo`, then reads it back to verify.
- **macOS** (`_secure_posix`): **starts a `/bin/chmod -N` subprocess** per path,
  then `chmod` + `lstat`.

So each `connect_database()` makes about 3 or more ACL rewrites on Windows, or about 3 or more process
launches on macOS. This happens in addition to the `CREATE TABLE IF NOT EXISTS` script
(lines 28–57). `connect_tag_db` then adds two more `CREATE TABLE` statements, a
`PRAGMA table_info`, and a `commit()` (`document_tag_store.py:122–176`).

Combined with P3, ordinary UI actions trigger dozens of these. Startup alone opens
roughly 20 or more connections (see P3).

**Fix:** secure the tree once per process (at `configure_runtime`) and cache the result.
Only re-protect a DB file when it's newly created. Run schema creation and migration once
per process per DB path (a module-level `set` of initialized paths). Then
`connect_database` becomes just `sqlite3.connect` + `PRAGMA foreign_keys`.

### P2 — High: full-corpus result materialization on every search and filter toggle

`document_query_visuals._search_documents` (lines 1161–1172) calls `search_index`
with `top_k=len(chunk_records)`. Inside `search_index`
(`document_search_engine.py:504–529`) that means:

- `np.argsort` over all scores (a full sort, O(n log n)),
- for **every chunk**: a dict, `Path(doc["pdf_path"]).name`, a whitespace
  normalization `" ".join(text.split())`, a 700-character slice, and a reference to
  the query vector.

Everything is kept in `self._last_results` until the next query. That is a second
in-memory copy of most of the library's text.

When any tag filter is active, `_filter_result_rows` (lines 650–662) then:

- runs `load_tag_rows()` (a fresh connection, P1) and calls `Path(...).resolve()` on **every
  tagged file**, and
- calls `Path(result[path_key]).resolve()` on **every result**. That's one filesystem
  resolve **per chunk** in the library, repeated on every checkbox toggle.

Only 5 results are ever displayed (`filtered[:TOP_K]`).

**Fix:** keep the `scores` vector (one float per chunk) from the query. To filter, build
a set of allowed `doc_id`s from the tag table **once**, keyed by stored path (no
`resolve()`), and turn it into a boolean mask over the chunks with a precomputed
`chunk_doc_index` array. Then take `np.argpartition(-masked_scores, k)[:k]` and sort
only those k. Build result dicts only for the k shown. This keeps the existing
"filter before the cutoff without re-encoding" behavior at a tiny fraction of the cost.
The same `argpartition` change helps `search_index` generally (the current code
fully sorts even when `top_k=5`).

### P3 — High: connection-per-helper and N+1 patterns in the GUI

Each tag-store helper opens and closes its own connection (each paying P1 + DDL):

- `_build_tag_filter` (line 359–365): calls `tag_names()` once, then `tag_color(tag)` **without**
  `available_tags` for every tag. That's another `tag_names()` → `load_tag_definitions` →
  `connect_tag_db` **per tag**: 1 + N connections (17 or more with the built-in tags).
- Startup also runs `_empty_categories_html` from `_build_categories_section` (line 474,
  2 connections). `setup_ui` line 133 immediately **overwrites** that HTML with
  `_format_categories_html()` (2 more connections, plus the empty view again if there are no rows).
  The first render is wasted work.
- `_search_documents` (line 1135) calls `tag_names()` on every query. For a tag query,
  `exact_tag_matches` calls `tag_names()` again, then opens another connection.
- `_tag_filter_changed` without a query calls `exact_tag_matches()` per checked tag (2 connections each).
- `_format_search_results_html` / `_format_tag_results_html` each call
  `load_tag_rows()`. `_tags_for_path` (lines 832–837) then does a **linear scan with
  `resolve()` per row for every result**: O(results × tagged files) filesystem calls.

**Fix:** cache tag definitions and tag rows in the window and invalidate them on save,
analysis completion, rename, remove, clear, or import. Pass `available_tags` to
`tag_color`. Index tag rows in a dict keyed by stored path. Drop the
initial `_empty_categories_html()` render in `_build_categories_section`. Together with P1,
this removes nearly all DB overhead from interactive actions.

### P4 — Medium: blocking work on the UI thread

`search_documents` runs synchronously in the Qt event handler:

- query encoding (and, on first search, `load_vector_store` + model load),
- `PassageSelector.select` re-embeds up to `MAX_CANDIDATES = 64` passages for each
  of 5 results (`search_snippets.py:126`), up to about 320 strings through the
  transformer on CPU,
- `WordPageLocator.enrich` → `_read_word_pages` **starts a hidden Word instance**
  through `DispatchEx`, opens each .docx, repaginates, and walks pages
  (`word_page_lookup.py:58–101`). That can take several seconds, and the window freezes
  meanwhile.

Analysis already uses a `QThread`, but search doesn't. **Fix:** run search and
enrichment in a worker (or render results first and fill in passage and page labels
asynchronously). Keep one Word instance alive per session rather than one per lookup.

Smaller items in the same path:

- `PassageSelector.select` computes `passage_candidates(source)` (regex splitting
  plus up to 3× sliding concatenation with `part in full` substring checks) **before**
  checking `self._cache`. Cached sources still pay for candidate generation. Check the
  cache first.
- `PageMap.__init__` builds the page map one character at a time in Python with an
  `array('I')` append per character. That's fine for small documents but slow for long ones.
  It's cached, so it's a minor issue.

### P5 — Medium: rename and remove touch every record with `resolve()`

- `edit_indexed_document` (`document_database.py:155–157`) iterates **all**
  `documents` rows and calls `Path(...).resolve()` on each to find one path.
  `_stored_path_key` in the engine documents that stored paths are already
  absolute and resolved. An indexed `WHERE pdf_path = ?` lookup (with a fallback
  only for legacy relative rows) is enough.
- GUI `_rename_file` (lines 783–789) calls `resolve()` on every doc record **and every
  chunk record** to patch the `pdf_path` strings. `_remove_file_from_database`
  (lines 819–825) does the same over docs, then rebuilds the chunk list and vector matrix.
  Chunks reference `doc_id`, so patching one `docs_by_id` entry is enough for a rename,
  provided `pdf_path` isn't duplicated into chunk records (see R6).

### P6 — Medium: startup walks and re-secures every data file

`app_storage.prepare_storage` runs `os.walk` over library, tmp and logs,
calling `protect_path` on every file and folder at each launch: an ACL rewrite
on Windows or a `chmod -N` subprocess on macOS. Accumulated logs and temporary
files can increase startup time. Review whether a versioned deep repair can
preserve the existing protection guarantees while reducing repeated work.

### P7 — Medium: tagging plan resolves the whole library

`prepare_tagging` (`document_tag_store.py:312–315`) loads every processing state and
every `file_tags` row, then calls `Path(...).resolve()` on each, for every analysis,
even when one file is being added. That undercuts the incremental design
in `update_vector_store`, which deliberately avoids per-document `resolve()`.
**Fix:** use the stored strings directly, as `_stored_path_key` does, and fetch only
rows for the requested paths (`WHERE pdf_path IN (...)`).

Related, smaller items:

- `tag_files` progress loops (lines 430–433, 441–444) iterate the entire `work` list for
  every batch, giving O(work × batches) progress emissions.
- `AnalysisProgressDialog.update_progress` (`analysis_progress.py:146`) calls
  `resolve()` on every progress event. `_fingerprint` emits a progress event per MiB
  hashed, so a large file produces hundreds of events, each with a resolve, a Qt signal,
  and `scrollToItem`. Paths are already resolved by the worker, so the dialog can use
  them as-is and throttle repeated "checking" events.
- The same paths are resolved again in `AnalysisWorker.__init__`,
  `update_vector_store`, `prepare_tagging`, and `tag_files`. Resolve once at the entry
  point and pass the result through.

### P8 — Low–Medium: index load and lock usage

- `load_index` (`document_database.py:304–312`) calls `np.frombuffer` +
  `np.isfinite` + a row copy **per chunk**. `load_document_chunks` already uses
  the faster `b"".join(blobs)` → one `frombuffer` → one `reshape` → one `isfinite`.
  Use that approach in both.
- `load_index` and `load_index_summary` open `BEGIN IMMEDIATE` (a write lock) for every
  read, only because the one-time legacy JSONL migration might run. Check for the
  metadata row first and take the write lock only if a migration is needed. Otherwise
  use a plain read transaction.
- No `PRAGMA journal_mode=WAL` is set, though `.gitignore` and `app_storage` both
  anticipate WAL sidecars. WAL would let searches read while analysis writes.
- `apply_index_updates` runs the `sqlite_master` check for `file_tags`
  **inside** the per-document loop (line 123). Move it before the loop.

### Other inefficiencies

- **`assets/filesense.png` is 1.26 MB.** It is decoded for the window icon on
  macOS and Linux. A smaller PNG would reduce asset loading and checkout size.
- `_list_contains_path` (lines 633–637) scans the whole queue, building a `Path` for each
  item, for every added file. Adding many files is O(n²). Keep a `set` of queued paths.
- `chunk_page_text` chunks without overlap. That isn't a performance bug, but sentences
  that straddle a boundary are split across vectors. It's worth considering a small
  overlap if retrieval quality matters.

---

## Redundant code

| # | Location | Redundancy | Suggestion |
|---|---|---|---|
| R1 | `document_query_visuals.py:817` (`_remove_file_from_database`) | Calls `delete_file_tags(file_path)` after `remove_document_from_vector_store`, but `edit_indexed_document` (`document_database.py:162–164`) **already deletes the `file_tags` row** in the same transaction. This is the same pattern `CODEBASE_AUDIT.md` F1 fixed for rename, and it opens one extra connection (P1). It also leaves a window where the DB commit succeeded but the second call can fail and show a misleading warning. | Remove the second delete. `delete_file_tags` then has no GUI callers. |
| R2 | `document_database.load_index`, `load_document_chunks` | Two separate implementations of "read chunk rows → decode float32 blobs → validate dimension and finiteness." | One shared decoder (the `join`/`reshape` version). |
| R3 | `document_tag_store.py:126–137` and `154–160` | The `file_tags` schema is written out twice in one function. | One schema constant. |
| R4 | `document_database.py:123`, `160`, `185–187` | The `SELECT 1 FROM sqlite_master WHERE ... name='file_tags'` existence check is repeated. Since `connect_tag_db` always creates the table, all of these become unnecessary if schema setup happens once (P1). | Create all tables in one place, then drop the checks. |
| R5 | `_stored_path_key`, `edit_indexed_document` inline lambda, `connect_tag_db` migration, `_import_legacy_index`, `prepare_tagging`, GUI comparisons | At least 6 variants of "make this stored path absolute and comparable." | One `canonical_path(stored)` helper used everywhere. |
| R6 | `documents.pdf_path` (stored with `as_posix()`, forward slashes) vs. `file_tags.file_path` (stored with `str()`, backslashes on Windows); `pdf_path` also copied into every chunk dict | The two tables store the same path in different spellings, which forces `resolve()`-based comparison everywhere (root cause of P2, P3, P5 and P7). Copying `pdf_path` into each chunk record duplicates a string per chunk in memory and makes renames touch every chunk. | Store one canonical spelling in both tables and join on `doc_id`. Look up chunk paths through `docs_by_id`. |
| R7 | `document_query_visuals.py:818–825` | Hand-written in-memory index removal duplicates the keep/filter/rebuild logic in `merge_index_delta`. | Add a `remove_from_index(index, doc_ids)` helper beside `merge_index_delta`. |
| R8 | `_format_search_results_html` / `_format_tag_results_html` | Near-identical card HTML (link, title, score label, tag line). | One `_result_card_html(...)`. |
| R9 | `indra_models.checksum` vs. `document_search_engine._fingerprint` | Two SHA-256 file-hash loops with the same 1 MiB block size. | Share one helper with an optional progress callback. |
| R10 | `"all-MiniLM-L6-v2"` literal in `document_database.py` (3×), `model_runtime.py`, `document_search_engine.MODEL_NAME`; `"sentence-transformers/all-MiniLM-L6-v2"` as `document_tag_store.DEFAULT_MODEL`; `'file_tags.sqlite3'` literal in GUI lines 105 and 244 despite `DATABASE_NAME` | Two spellings of one model name are what make `canonical_model_name` necessary, and constants are duplicated. | One `MODEL_NAME` constant and `DATABASE_NAME` everywhere. `clear_database` should use the constant too. |
| R11 | `document_tag_store.DocumentTagger.__init__` (lines 64–77) | The `ModuleNotFoundError` guards for NumPy and sentence-transformers can't trigger: the module already imports both at top level through `document_search_engine` (lines 7–10). `TaggingDependencyError` is therefore dead code. | Remove the guards and the exception class, or make the imports genuinely lazy. |
| R12 | `prepare_tagging` + `load_tagging_vectors` | Together they open 4 connections (`load_tag_definitions`, `connect_tag_db`, `load_document_chunks`, `connect_database`) to read from one DB. | One connection passed through. |

## Unused or misleading code and files

- **`delete_file_tags`** becomes unused once R1 is applied.
- **`_selected_file_paths`** (`document_query_visuals.py:639`) returns **all**
  queued files, not the selection, and `analyze_files` relies on that. The behavior is
  intentional, but the name is misleading next to `remove_selected_files`. Rename it
  `_queued_file_paths`.
- **`remove_document_from_vector_store` / `rename_document_in_vector_store`**
  (`document_search_engine.py:258–263`) are one-line pass-throughs to
  `edit_indexed_document`. Keep them if they're a deliberate public API. Otherwise
  the GUI can call the database function directly.
- **`build_index` + `save_vector_store`** (`document_search_engine.py:175–255`)
  are a full-rebuild path that the GUI doesn't use (it uses `update_vector_store`). It's
  also less efficient: there's no batching or progress, and it encodes the whole corpus in one call.
  This remains a standalone indexing API, with OCR integration-test callers.
  Preserve compatibility, or have it delegate to `update_vector_store`.
- **`requirements.txt` comment on `pywin32`** says it's for Word pagination, but
  `app_storage._secure_windows` imports `win32security`/`ntsecuritycon`
  unconditionally on Windows. pywin32 is **required** to launch at all on Windows.
  The comment should say so, or someone may "optimize" the dependency away.
- **`ipykernel` / `matplotlib`** in the runtime requirements serve only the notebook
  and `visualize_vectors.py`. This is already flagged in `CODEBASE_AUDIT.md`. Splitting
  them into a dev/tools requirements file shrinks the desktop install.
- **`Indexing_and_vector_search.ipynb`** is a stale, divergent prototype (60-token,
  PDF-only, JSONL output). It's already flagged. Archive it or label it historical.
- **`.git/refs/codex/turn-diffs/...`** contains leftover refs from a Codex session. They're harmless,
  but they pin otherwise unreachable objects. They can be removed with `git update-ref -d`.

## Repository hygiene

- **Line-ending churn:** the 6 modified files show as whole-file diffs
  (about 3,000 lines changed) because the working copies are CRLF and the committed blobs are LF,
  with no `.gitattributes` file. The real change is about 126 lines (`git diff --ignore-cr-at-eol`).
  Add `.gitattributes` (`* text=auto eol=lf`, plus `*.bat`/`*.ps1 eol=crlf`) before committing,
  so the pending cleanup produces a reviewable diff.
- **Uncommitted cleanup:** the working tree contains the `CODEBASE_AUDIT.md`
  cleanup (unused helpers removed, rename fix). Commit
  it separately from any changes made from this report.

## Suggested order

1. **P1 + P3** (initialize once per process, cache tag data, pass `available_tags`):
   the largest win for the least risk. Every UI action gets faster.
2. **P2** (score mask + `argpartition`, build only k result dicts) and **R6**
   (one canonical path spelling). Together they remove almost all per-chunk
   `resolve()` calls (P5, P7 follow naturally).
3. **P4**: move search and Word page lookup off the UI thread.
4. **P6, P8, R1**, then the remaining redundancy table as a separate cleanup commit.
5. Add `.gitattributes` before committing any of the above.

Each performance item can be checked against the existing suite
(`tests/test_optimization.py`, `tests/test_pipeline_speedups.py`, `tools/check_speedups.py`).
I'd suggest adding a scaling test with a synthetic 20k–50k-chunk index, so that P2 and P3
regressions show up in tests.
