# Senior Project Vector Search

Local document archive search tool with semantic vector search, automatic document tagging, and a PyQt desktop interface.

## Supported documents

**Load Documents** accepts PDF (`.pdf`), Word (`.docx`), PowerPoint (`.pptx`),
Excel (`.xlsx`, `.xls`), and Rich Text Format (`.rtf`). All use the existing local
index, tags, preview and search; install updated readers with
`python -m pip install -r requirements.txt`. Microsoft Office is not required on
Windows or macOS. Save older `.doc`/`.ppt` files as `.docx`/`.pptx` before importing.

Word paragraphs and tables, PowerPoint text/tables/speaker notes, spreadsheet
cell values and sheet names, and RTF text are extracted directly. Search results
identify **slides** for presentations and **sheets** for spreadsheets. Word and
RTF are treated as one logical document, since their printed page numbers depend
on the viewing application. Spreadsheet formulas use stored/cached values; they
are not recalculated. Images inside Office/RTF documents are not OCR-processed;
export image-only documents to PDF to use the existing OCR fallback. Password-
protected or unreadable files report an extraction error.

Existing PDFs and databases remain compatible. The database's historical
`pdf_path` and `page` fields also store non-PDF paths and slide/sheet indices,
avoiding a schema migration. Unchanged Office/RTF documents reuse saved results.

## Features

- Extracts text from local PDF files with PyMuPDF, with local Indra CPU OCR for scans on Windows and macOS.
- Splits extracted text into ~200-token chunks.
- Embeds chunks with `sentence-transformers`.
- Searches indexed chunks by semantic similarity.
- Assigns simple semantic tags such as Finance, Legal, Travel, Work, and Health.
- Provides a desktop GUI for adding PDFs, indexing them, previewing text, and searching.
- Persists document metadata, text chunks, embeddings, and tags in a local SQLite database.

## New features

- Load PDFs with compact document-name display, full-path hover tooltips, and double-click opening in the system PDF application.
- Filter results with a readable, horizontally scrolling Tag Filter bar. No selection searches all documents; selected tags match any selected category.
- Right-click results to open a file's location, rename the PDF, or remove its database, vector-index, and tag records with confirmation.
- Rename operations update the actual file, SQLite/vector index, saved tags, and visible file list together.
- Windows and macOS launch support, including a Windows OCR compatibility path and PowerShell launcher.
- Expanded application window with a collapsible empty file queue and improved indexing/status feedback.
- **Advanced → Performance** opens optional search history and background retrieval evaluation, including a sample test and a saved-answer-key workflow for your collection.

## Setup

Use **Python 3.12** and create a separate environment on each computer. Do not
copy or commit `.venv`; it is already excluded by `.gitignore`.

macOS / Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Both OCR choices use ONNX Runtime's CPU backend on Windows x64, Intel macOS and
Apple Silicon macOS (including M1/M2/M3/M4). Rerun the install command after
updating an existing checkout. Use native Python 3.12 for each CPU and **macOS 13
or newer** on Macs. The requirements constrain ONNX Runtime/OpenCV/Qt to versions
with Intel and ARM Mac wheels, and select a compatible embedding stack on Intel
Macs. This does not support every historical MacBook or Windows PC: older OS
versions and 32-bit systems are outside this setup. Indra has been smoke-tested
on Windows here; native Intel/Apple Silicon Mac verification remains necessary.
Do not use `pip freeze` to overwrite this maintained list with unrelated packages.

## Run

The embedding model can be included inside the program at
`models/all-MiniLM-L6-v2/` (about 87 MiB). FileSense prefers this folder and loads
it locally, without Hugging Face requests. Include the whole folder when copying
or packaging the application. Large weights are excluded from Git; to prepare a
checkout from an existing cache, run `.\.venv\Scripts\python.exe bundle_model.py`.
Use `--download` if the pinned model is not cached, or `--verify` to check the
bundled files. See [model packaging instructions](models/README.md).
Bundling avoids downloads/cache dependence; the model still loads into memory
once per session. OCR models, tiktoken assets, and Python dependencies are
separate from this embedding-model bundle. Indra's OCR weights can also be included
at `models/indra-ocr/` (about 73 MiB):

```sh
python bundle_indra.py --download
python bundle_indra.py --verify
```

Include that whole folder when copying the app to Windows or Mac. The weights
are platform-independent, excluded from Git, and reused without network checks.

Start the desktop app:

```powershell
python document_query_visuals.py
```

Keep the `assets/` folder with the application when copying it. It contains
the FileSense artwork and the Windows icon used by the window and taskbar.

Optional vector visualization:

```powershell
python visualize_vectors.py
```

## Search and tag filters

**Analyze Files** runs in a background worker and opens a floating progress window.
It shows the current PDF, elapsed time, a file checklist, pages read/OCR-processed,
and completed embedding chunks out of the total for that PDF. Model preparation,
category preparation, and database saving show an activity indicator when no
reliable percentage is available. The main collection controls are temporarily
disabled to prevent conflicting edits; the window continues responding.
Search, indexing, and tagging reuse a lazily loaded embedding model throughout
the session. Cached models are loaded locally first. Category-description vectors
are reused until the definitions change, and document classification is batched.
Use **Cancel** to stop between steps; a running model call or database save must
finish first. Previously saved work remains available. Closing FileSense requests
the same safe stop and closes the app once the worker exits.

Leave every tag unselected to search the whole collection, including untagged
documents. Select one or more tags to include PDFs with **any** selected tag.
Selecting all tags still excludes untagged PDFs; clear the selection to include
them again. The tag row scrolls horizontally to keep category names readable.

For semantic queries, FileSense applies the tag filter before taking the top five
chunk hits. Changing tags reuses the full ranking from the last submitted query,
so a matching document can appear even when it was outside the previous top five.
Filter changes do not re-embed the query or add a search-history entry. An exact
tag-name query remains the submitted query when filters change. Before any query
has been submitted, selecting tags browses documents in those categories.

Semantic results show a concise passage selected from the matching section,
instead of always displaying its first 700 characters. FileSense scores sentences
and nearby context with the existing local embedding model, preferring the
shortest passage within one similarity point of the best candidate (up to 400
characters). Long unpunctuated OCR text uses overlapping word windows. Only visible
results need passage scoring; candidate embeddings are cached in memory for reuse.
The section similarity still determines result order, and the passage similarity
describes the excerpt shown. Both measure semantic similarity, not confidence or
accuracy. Existing indexed PDFs work immediately without reanalysis.

Word results show **Match found in Page X** above the passage, using Microsoft
Word's read-only pagination on Windows. A passage crossing a page boundary shows
both pages. Page maps are cached for the session and refreshed when the file
changes; existing Word indexes work without reanalysis. If Word is unavailable,
the source is missing, or the passage cannot be located unambiguously, the result
shows **Page unavailable** instead of guessing a number. Other document formats
retain their current passage labels.

## Performance in the app

Open **Advanced → Performance**. The **Recent searches** tab has an optional
**Track search performance** toggle, initially off. When enabled, it keeps the
latest 100 searches in memory for the current FileSense session: query text,
elapsed time, and the actual displayed semantic or tag results after filtering.
Repeated PDF chunk hits remain in their displayed order. Similarity and tag scores
are recorded as search output; they do not establish correctness, precision, or
recall. The toggle preference is saved in `settings/performance.ini` inside the private FileSense data folder;
history itself stays in memory and is not written to SQLite. **Export Excel…**
opens a Save dialog for one `.xlsx` workbook with **Summary**, **Searches**, and
**Results** sheets. Results include their query text and file paths, but no
document snippets. **Clear history** removes the session records; turning tracking
off pauses recording without clearing them. Existing export files are never
overwritten, so choose a new filename for each export.

In the **Evaluation** tab, choose **Sample test · ready to run** and click
**Run sample evaluation** to exercise the workflow with three invented PDFs and
four authored queries. Or choose **My collection · saved answer key**, select a
JSON/CSV relevance benchmark and its document root, then click
**Evaluate my collection**. Each query needs explicit relevance grades for every
indexed PDF; FileSense does not generate these labels from similarity scores.
The default evaluates the top five unique PDFs with FileSense and TF-IDF. Inspect
the summary, per-query metrics, and ranked PDFs, then use **Export Excel…** to
save one `.xlsx` workbook with **Summary**, **Aggregate**, **Per query**,
**Rankings**, and **Metadata** sheets. Numeric values remain numbers for sorting
and calculations in Excel.

Evaluation runs in a separate process using the application's local model/tokenizer
caches, with downloads disabled. It can use extra CPU and model memory while you
continue working. **Cancel** stops a run; closing the Performance panel leaves it
running, and quitting FileSense cancels it. The sample builds a new FileSense
index, so even **TF-IDF only** needs the cached FileSense model and tokenizer.
TF-IDF on your existing collection's index does not need the embedding model.
The interface uses PyQt6 and the evaluation dependency scikit-learn. Excel export
adds `openpyxl>=3.1,<4`; update the evaluation requirements in the same Python
environment as FileSense, including when upgrading an existing installation:

```sh
python -m pip install -r evaluation/requirements.txt
```

Each run stays under `evaluations/app-{demo|benchmark}-TIMESTAMP-ID/` inside the private FileSense data folder.
**Do not share the whole run directory:** it contains a private SQLite snapshot
with document text and embeddings, plus a copied answer key. These local paths
are ignored by Git. **Export Excel…** saves the evaluation tables and metadata
in one workbook without the private index or answer-key snapshot. The underlying
`results.json`, `per_query.csv`, `aggregate.csv`, and `rankings.csv` remain in the
run's `report/` folder, accessible through **Open report folder**. Evaluation
reports omit raw query and document text but still contain query IDs and document
names; review them before sharing. Search-history workbooks include query text
and paths and need their own privacy review.

The three-document sample is intentionally easy. With top five results, returning
all three documents reaches the recall ceiling for queries with relevant answers;
equal quality metrics between the systems can be valid. The model and evaluation baseline
are unchanged. Use a representative labeled benchmark to investigate differences,
and keep the baseline fair instead of adjusting it to force a winner. Synthetic
sample results cannot support Chapter 5 claims about real retrieval quality.

See [the evaluation guide](evaluation/README.md) for labeling, metric definitions,
privacy details, and the unchanged `python -m evaluation` command-line workflow.

## Privacy and separate computer accounts

FileSense uses the signed-in Windows or macOS account as its identity. It does
not require another username or password. Separate ordinary computer accounts
have separate libraries, tags, preferences, and evaluation runs; a new account
starts with an empty library. Clearing a library affects only that account.

Default data roots are resolved from the operating system:

| Platform | Private FileSense data folder |
| --- | --- |
| Windows | `%LOCALAPPDATA%\FileSense\` |
| macOS | `~/Library/Application Support/FileSense/` |

Some Windows Python distributions redirect application data into a package-specific
folder in the same user's profile. Hover over the main window's library indicator
to see the effective location. An environment variable or launch working directory
does not select another account's data root.

The folder contains `library/` (SQLite text, embeddings, paths, tags, and processing
state), `settings/`, `evaluations/` (reports, answer keys and index snapshots),
`cache/`, `tmp/`, and `logs/`. Recent searches and Word page maps remain in memory
unless explicitly exported. Python temporary files and model/tokenizer caches
use the private location during desktop launches. Distributed public model
weights can still be read from `models/`; first-use OCR downloads use the private
cache if a bundled model is absent.

Windows directories use protected access-control lists granting access only to
the current user, SYSTEM, and Administrators. macOS directories use owner-only
permissions (`0700`) and private files use `0600`, with extended ACLs removed.
Startup repairs and verifies managed data permissions, rejects links/junctions
in private state, and stops with an error if protection cannot be established.
It never falls back to storing private data beside the program.

A portable copy can run from a USB drive, but its private library stays with the
account on that computer. Copying the program does not carry the library or
another person's history. Installed and portable copies under the same account
use that account's library.

### Importing an existing library

Choose **Advanced → Import Existing Library…**, select the old
`vector_store_notebook/file_tags.sqlite3`, and explicitly confirm the copy into
your current account. The destination must be empty, including custom category
definitions. The import validates the database and vectors, preserves text,
embeddings, tags and processing state, and copies a consistent SQLite snapshot
including committed WAL changes. It never overwrites an existing private library.
An incompatible or damaged import leaves the current library intact.

Only import libraries you are authorized to use. Existing libraries are never
assigned to an account automatically. The source database, its backups, old
reports, and source documents remain unchanged. Old shared copies retain their
previous access permissions; after verifying an import, their owner should
separately decide whether to secure, archive, or remove them.

### Boundaries

This is separation between ordinary computer accounts, not protection against an
administrator, malware running as the same user, or someone using an unlocked
session. Shared source documents remain shared. Exports and command-line output
paths are explicitly chosen by the user and follow their destination's access
permissions. Keep shared installed application files in an administrator-controlled
location; for a portable copy, use a folder other accounts cannot modify. FileSense
does not change permissions on the program folder or on original documents.

## Local Data

Use **Advanced → Clear Database** to start a fresh analysis. After confirmation,
this clears indexed documents, extracted text, embeddings, and saved tag assignments.
Original PDFs, custom category definitions, and queued files are kept. Click
**Analyze Files** to index the queued PDFs again. Existing evaluation reports and
snapshots are separate and are retained.

The app uses `library/file_tags.sqlite3` inside the current computer account's
private FileSense data folder (see **Privacy and separate computer accounts** above).
SQLite is included with Python; no database server or application login is needed.
Installed and portable copies use the same account library regardless of the
launch folder. The old database beside the application is never opened automatically.

- `documents`: document IDs, PDF paths, extraction methods, and character counts.
- `chunks`: page numbers, extracted text, and float32 embedding blobs, linked to documents.
- `index_metadata`: embedding model and vector dimension.
- `file_tags` and `tag_definitions`: assigned categories and custom tag definitions.
- `document_processing`: content fingerprints, size/mtime, indexing settings, and
  the category-definition signature used for each document.

Analyze & Index saves the search index in one transaction. Search reloads it after
restarting the app, and semantic similarity is still computed locally with NumPy.
Files with the same name in different folders are stored separately. Unchanged
PDFs reuse their stored text, embeddings, and category assignments without model
work or database writes, including after restarting the app. Size and nanosecond
modification time provide the initial change check; changed metadata triggers a
SHA-256 content check. Changed content/settings replace that PDF's old chunks in
one transaction, retaining its document ID and leaving other PDFs untouched.
Identical PDF bytes reuse extraction/OCR and embeddings but retain independent
searchable paths. Analysis never loads the whole index: it reads document
fingerprints, plus stored chunks only for identical content it reuses, and the
open window patches just the changed documents into its in-memory index. Chunks
from all newly analyzed PDFs are embedded together in shared batches. Categories
are computed from each document's stored chunk embeddings (their normalized mean),
so tagging needs no second model pass and reflects the whole document. Category edits invalidate assignments without re-extracting PDFs.
Blank/encrypted results are also remembered to avoid repeated failed OCR attempts.
The first analysis of older indexed PDFs without fingerprints rebuilds their
results once to establish reliable change tracking.

If legacy `docs.jsonl`, `chunks.jsonl`, and `vectors.npy` files are explicitly
placed in a selected index folder, they are imported together on its first load.
FileSense never searches another account or the old application folder for them. They remain untouched as backups;
subsequent saves use SQLite. An incomplete or invalid legacy index raises an
error rather than silently discarding it. Legacy indexes did not record the
embedding model, so migration assumes the requested model (the default is
`all-MiniLM-L6-v2`). After migration the saved model is checked on load.
Existing tag assignments and custom definitions are retained. Notebook code that
directly reads the old JSONL/NumPy files will see the pre-migration snapshot; use
`load_vector_store()` / `save_vector_store()` for current data. For a non-default
embedding model, pass the same `model_name` to both functions.

To back up current data, close FileSense and copy `file_tags.sqlite3` along with
your PDFs. PDFs themselves remain on disk and are not copied into the database.

Run the storage and integration tests using your virtual environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

This project intentionally ignores local PDFs and generated vector-store artifacts:

- `Files/`
- `vector_store_notebook/`
- `*.sqlite3`
- `*.npy`
- `*.jsonl`

Keep your private documents locally and regenerate the index through the app when needed.

## OCR fallback

**Indra OCR (CPU)** is used automatically for scanned PDFs during
**Analyze Files**. OCR uses a pinned revision in the indexing signature, so PDFs
are reprocessed when that revision changes.
Existing search results use saved text until those PDFs are analyzed again.

`document_search_engine.py` first extracts embedded text and creates chunks.
Only `if not chunks` (and the file is not encrypted) calls `ocr.py`.
Recognized text goes through the same tokenizer, embeddings, SQLite persistence,
search, and tags as normal text.
Tagging reuses the indexed chunk embeddings, so a scan is not OCR-processed twice.

OCR runs locally on the CPU and renders pages at 150 DPI (longest side capped at
3000 px). Indra runs on Windows and macOS:

- **Indra:** pinned `indra-ocr==1.1.2`, explicitly using `CPUExecutionProvider`,
  with up to four inference threads. All default multilingual networks and
  language models are stored in `models/indra-ocr/`. The folder loads locally;
  a missing folder is prepared once on first OCR. An incomplete local bundle
  reports an error. Use `bundle_indra.py --verify` to validate model hashes.
  Rasterization converts RGB directly to grayscale, avoiding a redundant BGR
  image copy while retaining the same grayscale pixels as the previous path.
  Optional math recognition is off. [Indra documentation](https://github.com/datalab-to/indra)
  describes its dependencies, accuracy limitations and model license; reported
  multi-core throughput is not a speed guarantee for an individual laptop.

Indra is loaded lazily and reused by the process. Allow internet access
only for initial setup when weights were not distributed; local inference does
not upload your PDFs. The existing
Sentence Transformer and tokenizer also need their initial model downloads.
Startup and OCR can take longer than normal extraction. Analysis runs in a
background worker with per-page OCR progress in the floating progress window.

Recovered files show `[ocr]`. Blank pages can still yield no text. A failed OCR
setup or page produces an explicit error; that indexing batch is not saved, and
can be retried after fixing the problem. Existing database records are preserved.

This is deliberately a document-level fallback: if a PDF already produces any
chunks, scanned pages mixed into that PDF are not OCR-processed. Recognition of
handwriting, formulas, and complex reading order still needs evaluation on your
own examples. Original PDFs are unchanged; recognized text is stored in SQLite,
not written back as a selectable PDF text layer.

Run the storage and OCR integration tests with:

```sh
python -m unittest discover -s tests -v
```

These tests use a fake OCR predictor to check control flow without model downloads.
Actual model recognition and SQLite search are also verified separately on a
synthetic image-only invoice during development.
