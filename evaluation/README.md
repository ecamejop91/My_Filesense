# FileSense retrieval evaluation

Evaluate **semantic PDF retrieval** against a TF-IDF baseline from
**Advanced → Performance** in FileSense. The same metric library also supports
the existing `python -m evaluation` command-line workflow, which remains
independent of the GUI. Optional search history records what the app displays;
quality evaluation runs separately against authored relevance judgments.

## Start in FileSense

1. Open **Advanced → Performance**, then the **Evaluation** tab.
2. Choose **Sample test · ready to run** and click **Run sample evaluation** for
   three invented PDFs and four authored, labeled queries. The sample creates an
   isolated index and leaves your current collection unchanged. It demonstrates
   the workflow; its scores do not measure your collection's retrieval quality.
3. For your own documents, choose **My collection · saved answer key**. Select a
   version 1 JSON or CSV benchmark, the **Document root** matching its relative
   PDF paths, and a test or development query split. The app uses a frozen copy
   of its current SQLite index. Every indexed PDF needs an explicit 0–3 relevance
   grade for every query; incomplete judgments stop the run. The suggested root
   comes from indexed paths and should be checked against your answer key.
4. Choose **FileSense + TF-IDF**, **FileSense only**, or **TF-IDF only**, and the
   number of top results. The default is five unique PDFs, not five chunk hits.
   Click **Evaluate my collection** for a custom benchmark.
5. Read **Summary**, **Each query**, and **Ranked PDFs**. Summary quality metrics
   are macro averages; undefined values appear as N/A. Use **Open report folder**
   to inspect the saved JSON/CSV reports or **Export Excel…** to save one `.xlsx`
   workbook. The Save dialog lets you choose the filename; choose a new path,
   because existing files are never overwritten.

Quality labels are never inferred from similarity scores, tag assignments, or
search history. Custom evaluation must be explicitly started with an authored
answer key. It calls semantic search directly, including when a query happens to
match a tag name. The GUI requires complete judgments; the CLI-only
`--allow-unjudged` option described below is not available in this panel.

Runs execute in a separate process so the panel remains responsive. They may
consume additional CPU and model memory and affect concurrent search timings.
Use **Cancel** to stop a run. Closing the Performance panel keeps it running;
quitting FileSense cancels it. Cancelled or failed runs can leave partial local
job files, but they are not presented as completed reports.

In-app evaluation runs offline and inherits the app's model/tokenizer cache
configuration; it never downloads missing assets. Prepare needed assets in the
same FileSense environment before retrying a missing-cache error. The sample
must build a FileSense index, so **TF-IDF only** still needs the cached embedding
model and tokenizer for that sample. A custom **TF-IDF only** run uses the
existing indexed text and does not load the embedding model. The panel uses
PyQt6, scikit-learn for evaluation, and `openpyxl>=3.1,<4` for Excel export.
Openpyxl is a new dependency and is loaded only when exporting a workbook.
After updating FileSense, install `evaluation/requirements.txt` in the same Python
environment as the app; an environment with the previous evaluation dependencies
may still need openpyxl. Missing export dependencies do not change stored history
or the underlying JSON/CSV evaluation reports.

### Interpreting the small sample

The sample has only three short, distinct PDFs and four queries, including a
no-answer query. Equal metrics are possible when both systems identify the
obvious targets. At the default k=5, returning all three searchable PDFs gives
Recall@5=1 for queries with relevant documents, regardless of their ordering.
Precision@5 still divides by five; the no-answer query has undefined recall,
RR, and nDCG. MRR and nDCG can distinguish ranking changes, but they need not
differ on these simple queries.

The embedding model and baseline are unchanged by Excel export and tag-filter
updates. A tie on this sample does not by itself indicate a defect or establish
equal performance on real collections. Compare the systems on a representative,
independently labeled benchmark with paraphrases, distractors, and varied
information needs. Keep the baseline and evaluation protocol fair; do not tune
TF-IDF to lose or change labels after inspecting results. The sample demonstrates
the workflow and cannot provide Chapter 5 evidence about real retrieval quality.

## Optional recent-search history

In **Recent searches**, **Track search performance** is off on a fresh setup.
Its preference is saved in `settings/performance.ini` in the current account's
private FileSense data folder and restored when
the app restarts. When enabled, the last 100 completed searches are retained in
session memory, newest first. Disabling pauses recording while preserving those
entries; **Clear history** removes them. Closing FileSense discards unexported
history. Recording does no additional model run or search and writes no search
data to disk or SQLite.

Each entry holds the query, timestamp, elapsed milliseconds, search type, and
actual displayed results after tag filtering. Display order and repeated PDF
chunk hits are preserved. Semantic hits include their page and raw cosine score;
exact tag hits use their tag score and have no page. Timing includes index
loading when needed and preparation of the displayed results. With no tags
selected, ordinary search includes all documents, including untagged PDFs. One
or more selected tags match **any** selected category; even selecting every tag
excludes untagged PDFs. The tag row scrolls horizontally to keep names readable.

Semantic search caches the full ranked hit list for the submitted query, applies
the selected tags, and then displays the top five chunks. Changing tags reuses
that ranking without re-embedding or creating another history entry. It does not
change the stored history snapshot from the original submission. An exact tag-name
query remains the submitted query when filters change; before any query is
submitted, selecting tags browses tagged documents. These GUI filters do not
alter the benchmark's frozen collection or document-level evaluation. Quality remains
`not_evaluated`: these scores cannot substitute for relevance judgments.

**Export Excel…** in **Recent searches** saves one `.xlsx` workbook containing:

- **Summary**: history scope and export notes.
- **Searches**: one row per search, including searches with no results.
- **Results**: one row per displayed hit, with its search ID and query text.

Counts, elapsed times, pages, and scores retain numeric types. The workbook
contains query text and document paths, but no document text, snippets, or
embeddings. Source strings are stored literally, including leading `=` text;
exports contain no executable spreadsheet formulas or hyperlinks. Review history
workbooks before sharing. This export does not clear history or persist it to
SQLite, and is separate from evaluation export.

For existing programmatic callers, `SearchHistory.export(folder)` still creates
`search_history.json`, `search_history.csv`, and `search_results.csv` in a new
directory. The GUI no longer uses that three-file export. Its CSV formula-like
text is prefixed with an apostrophe, while JSON retains the original strings.

## Local evaluation data and report export

Each in-app job creates a new directory under
`evaluations/app-{demo|benchmark}-TIMESTAMP-ID/` inside the current account's
private FileSense data folder. These runs are not shared between computer accounts.
Explicit command-line output paths and exported workbooks follow the permissions
of the destination chosen by the user. Its `report/` subdirectory
contains `results.json`, `per_query.csv`, `aggregate.csv`, and `rankings.csv`.
These four raw reports are unchanged and remain accessible through **Open report
folder**, including for command-line workflows.

**Export Excel…** in **Evaluation** saves a single new `.xlsx` workbook with
**Summary**, **Aggregate**, **Per query**, **Rankings**, and **Metadata** sheets.
It includes numeric metric values and the evaluator's provenance, with nested
metadata expanded into named entries. Undefined values are represented as N/A,
not zero. Strings remain literal text; the workbook contains no formulas or
hyperlinks. Export refuses an existing destination, so choose another filename
instead of overwriting a report.

Report CSV text that could be interpreted as a spreadsheet formula is prefixed
with an apostrophe. Numeric scores are unchanged; JSON preserves original IDs
and remains the authoritative machine-readable representation.

The **whole job directory is private**: it also retains a copied answer key and
a SQLite snapshot containing indexed document text, embeddings, paths, and
existing tag data. The sample also retains its invented PDFs. These files stay
local, are ignored by Git, and are not uploaded. Do not share or commit the whole
run directory. Export the evaluation workbook, or use only the four raw report
files when a machine-readable format is needed. Review query IDs, relative PDF
names, and model metadata for identifying details. See **Outputs, provenance and
limits** below for the report's exact scope.

## Scope and specification

- Project report, Chapter 1 §§1.2.2 and 1.2.4: precision/recall, ranked retrieval,
  cosine similarity, and comparison with term-weighted vector search.
- Chapter 3 §3.4: known-answer “needle in a haystack” queries and Precision@k.
  **Tag classification accuracy is a separate experiment and is not measured here.**
- **Current Chapter 4: `SP_CH_4_draft_2.pdf`**, §§4.3–4.7. This supersedes the
  older template's Chapter 4. The code inspection agrees with the current draft:
  embedded text first, whole-document OCR fallback, nonoverlapping 60-token chunks,
  `all-MiniLM-L6-v2`, normalized vectors, SQLite, and five chunk hits by default.
  Exact tag-name lookup is separate; evaluation calls semantic `search_index`
  directly even when a query happens to match a tag name.
- Chapter 5: exports support the proposed results tables below. No measured
  retrieval claims can be made without an appropriate labeled collection.
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, supplied
  PDF, Chapter 6 §§6.2–6.3 (printed pp. 117–125), Chapter 8 §§8.1, 8.3–8.5
  (printed pp. 152–167). nDCG uses the exponential gain in Eq. 8.9, p. 163.
  MRR here measures the first relevant result, rather than MAP.

## Files and contracts

- `retrieval.py`: pure precision, recall, Precision@k, Recall@k.
- `ranking.py`: pure reciprocal rank, MRR, DCG/nDCG@k.
- `benchmark.py`: versioned JSON/CSV validation and relevance binding.
- `corpus.py`: read-only SQLite snapshot, normalization, max-score collapse.
- `backends.py`: existing FileSense search adapter and scikit-learn TF-IDF.
- `runner.py`: per-query results, macro/pooled aggregation, provenance, exports.
- `tracking.py`: bounded, opt-in session history and explicit JSON/CSV export.
- `excel_report.py`: one-file history and evaluation workbooks with typed values.
- `performance.py`, `performance_dialog.py`: GUI controller and Performance panel.
- `gui_worker.py`: offline background jobs, frozen index/answer-key copies, status events.
- `__main__.py`: isolated index building, inspection, benchmark execution.
- `demo.py`, `examples/`: invented fixtures only; no user PDFs or labels.
- `../tests/test_evaluation*.py`, `../tests/test_performance_gui.py`,
  `../tests/test_search_tracking.py`: arithmetic,
  worker/GUI integration, and session-history fixtures using `unittest`.

Native SQLite has `documents`, ordered `chunks`, and `index_metadata`. The normal
`load_index`/`load_vector_store` path may create tables or import legacy files.
Evaluation instead opens an existing database with SQLite `mode=ro` and reads one
transactional snapshot. Missing, corrupt, unnormalized, orphaned, or inconsistent
embeddings fail rather than producing a misleading result. Legacy JSONL-only
stores must first be migrated by the application or rebuilt separately.

All indexed PDFs are in the judged universe, including encrypted/no-text PDFs.
PDFs with no chunks are not retrievable by either system. Report their count and
extraction status; relevant unsearchable PDFs reduce recall and nDCG. This makes
extraction failures visible rather than silently removing them from ground truth.

## Metric definitions and edge cases

Binary relevance is grade >= 1. Relevance grades are integers **0–3**:

| Grade | Labeling rule relative to the information need |
|---|---|
| 0 | Does not satisfy the need; a shared keyword alone is insufficient |
| 1 | Marginally useful background or a partial answer |
| 2 | Substantially useful, but incomplete |
| 3 | Direct and complete answer / intended target document |

Let A be the unique retrieved PDFs up to `--depth`, R the relevant PDFs, and
A_k the first k unique PDFs. TP = |A ∩ R|.

| Metric | Definition |
|---|---|
| precision | TP / |A|; **0** for an empty result list |
| recall | TP / |R|; **null** when |R| = 0 |
| Precision@k | |A_k ∩ R| / k, even if fewer than k PDFs are returned |
| Recall@k | |A_k ∩ R| / |R|; **null** when |R| = 0 |
| reciprocal rank | 1 / first relevant rank; 0 if missed; null when |R| = 0 |
| MRR | Mean reciprocal rank over queries with at least one relevant PDF |
| DCG@k | sum from i=1 to k of (2^grade_i − 1) / log2(i + 1) |
| nDCG@k | DCG@k / ideal DCG@k; **null** when ideal DCG is zero |

IDCG sorts **all judgments**, including relevant PDFs not retrieved. Missing ranks
have zero gain. Pure metric functions remove duplicate IDs at their first position;
the search adapter first combines chunk scores using max and sorts unique PDFs.
Cutoffs must be positive integers and <= depth. RR/MRR are truncated at depth
(`mrr_at_depth` in aggregate output), not unrestricted whole-corpus MRR.

Macro means give each eligible query equal weight. Precision, Precision@k and DCG
include zero-relevant queries (with score 0); recall, RR and nDCG exclude them.
Exports record total, eligible, and zero-relevant query counts. All-undefined
averages are null. Pooled precision is sum(TP)/sum(retrieved), and pooled recall is
sum(TP)/sum(relevant). Pooled Recall@k uses summed cutoff TP; pooled Precision@k
uses k × number of queries, so it equals macro Precision@k under this fixed-k
policy. Pooled counts refer to **query-document pairs**, not distinct PDFs across
queries. Zero pooled retrieval gives precision 0; zero pooled relevance gives null.
No pooled nDCG or pooled MRR is reported.

## Fair comparison and preprocessing

Both systems use the same in-memory SQLite snapshot, query strings, query split,
judgments, searchable PDFs, depth, cutoffs, and document-level metric code.
FileSense calls the actual `document_search_engine.search_index` with **all chunks**;
only after taking each PDF's maximum chunk score do we truncate to depth.
Fetching k chunks and then deduplicating would unfairly shorten result lists.
Equal scores break by case-sensitive, Unicode NFC, corpus-relative PDF ID ascending
for both methods. Zero and negative scores are not thresholded away.

TF-IDF uses one document per PDF, joining stored chunks in (page, chunk_id) order
with newlines. Thus it uses the same extracted/OCR text, without reading PDF text
again. This join preserves current chunk text; stripped chunk boundaries may
insert whitespace within an original word. It is a documented approximation to
full extracted page text. Baseline pooling is **whole-document TF-IDF**, while
FileSense pooling is **max chunk cosine**; this is a system comparison, not an
isolated causal test of embedding models or pooling strategies.

TF-IDF fits vocabulary and IDF on corpus documents **only**, including empty PDFs
in document count N. Queries are transform-only; judgments are never passed to a
vectorizer. Fitting on the fixed retrieval collection is standard unsupervised IR
indexing, not query-label leakage. Parameters are explicit: lowercase, Unicode
word tokens with >=2 characters, unigrams, no stemming, no stop-word removal,
no accent stripping, raw term counts, smoothed IDF ln((1+N)/(1+df))+1, L2 norm.
An OOV query produces zero scores, sorted by the shared tie rule. An entirely empty
vocabulary is handled likewise. This smoothing differs from the unsmoothed
textbook log(N/df). See the [official TfidfVectorizer reference](https://sklearn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

Canonical benchmark IDs are PDF paths relative to `--corpus-root`, with forward
slashes. Absolute paths within that root and native index IDs (`doc_...`) are
accepted aliases, but relative paths are preferable because native IDs may change
on rebuild. Separators and dot segments are normalized; case is preserved to
avoid merging distinct files on case-sensitive systems. Duplicate normalized
paths/aliases, outside-root PDFs, unknown IDs, and repeated judgments are errors.
Use a common corpus root for scattered PDFs, or a dedicated corpus folder. Stored
relative paths use `--source-base` (the repository by default). Cross-platform
absolute paths require rebuilding the evaluation index on the target machine.

## Label a benchmark

1. Freeze a representative, shareable corpus. Use anonymous filenames. Keep
   documents and real benchmarks in `evaluation/local/` (ignored by Git).
2. Write information needs and queries before inspecting either system's results.
   Include paraphrases, exact keywords, multiple-answer queries, and needles amid
   distractors. Keep explicit no-answer queries to test edge cases.
3. Independently judge **every PDF for every query**, including explicit zeros.
   Use the grade rubric above. Two assessors should label a subset independently,
   resolve disagreements, and document agreement and adjudication. A tag/category
   label alone is not a query-specific relevance judgment.
4. Reserve `dev` queries for tuning chunk size/settings. Freeze untouched `test`
   queries before final comparison; don't tune on reported test results. A robust
   study needs many information needs and varied PDFs, not only the tiny demo.
5. Version `benchmark_id` whenever queries, corpus membership, or labels change.
   Archive the original benchmark privately with each run. JSON/CSV examples show
   the exact schema. Each CSV query has one row per document, including all zeros
   for no-answer queries. JSON supports an empty judgment map only with explicit
   `--allow-unjudged`; that is not equivalent to an exhaustively judged negative.

Incomplete judgments **stop all in-app runs** and stop CLI runs by default.
The CLI's `--allow-unjudged` treats absent
judgments as grade 0 and exports judgment coverage. Such recall and nDCG are
relative to known judgments and can be biased; report that limitation. Unknown
IDs still fail. All benchmark splits are validated against the corpus, while only
the selected split is evaluated. Duplicate JSON keys and conflicting CSV rows fail.

## Environment and retained CLI workflow

Run from the actual FileSense repository root in its Python **3.12** environment.
The application requirements supply PyQt6, NumPy, PyMuPDF, Sentence Transformers
and `tiktoken`; evaluation requirements supply scikit-learn and openpyxl for Excel
export. No additional test framework is needed. The commands below remain
available independently of the Performance panel.

```sh
python -m pip install -r requirements.txt -r evaluation/requirements.txt
python -m unittest discover -s tests -p 'test_evaluation*.py' -v
python -m unittest discover -s tests -v
```

Pure metric tests work with the Python standard library alone:

```sh
python -m unittest discover -s tests -p 'test_evaluation_metrics.py' -v
```

The CLI can generate a small invented corpus and run the real model. Unlike the
offline in-app worker, CLI index setup may download model/tokenizer assets when
they are missing; OCR downloads occur only when a document requires OCR:

```sh
python -m evaluation demo --output evaluation/local/demo-v1
python -m evaluation index --corpus-root evaluation/local/demo-v1/pdfs --output evaluation/local/demo-v1/index --chunk-size 60
python -m evaluation run --database evaluation/local/demo-v1/index/file_tags.sqlite3 --corpus-root evaluation/local/demo-v1/pdfs --benchmark evaluation/local/demo-v1/benchmark.json --depth 3 --k 1 3 --output evaluation/runs/demo-v1
```

Use a new output path per run; existing outputs are never overwritten. The demo
is a smoke check, **not a representative test collection**. Unit integration tests
use a deterministic mock encoder and must not be cited as model performance.

For a real benchmark, create `evaluation/local/study/pdfs` and place only the
chosen PDFs there, then build an isolated index and get canonical IDs for labeling:

```sh
python -m evaluation index --corpus-root evaluation/local/study/pdfs --output evaluation/local/study/index-60 --chunk-size 60
python -m evaluation inspect --database evaluation/local/study/index-60/file_tags.sqlite3 --corpus-root evaluation/local/study/pdfs --output evaluation/local/study/corpus.json
# Author benchmark-v1.json from the rubric and examples before the next command.
python -m evaluation run --database evaluation/local/study/index-60/file_tags.sqlite3 --corpus-root evaluation/local/study/pdfs --benchmark evaluation/local/study/benchmark-v1.json --split test --depth 10 --k 1 5 10 --output evaluation/runs/study-v1-test
python -m pip freeze > evaluation/runs/study-v1-test/environment.txt
```

Use `--systems tfidf` on the CLI's `run` command to evaluate an existing index
without loading the embedding model. Both methods still require the same existing
index; creating that index remains a separate step. CPU/GPU placement follows Sentence
Transformers; preserve model weights, hardware details and package versions.
Set `HF_HUB_OFFLINE=1` when weights are already cached and network-free execution
is required. Tokenizer/OCR caches must also exist for offline index building.

## Outputs, provenance and limits

The underlying `results.json` contains metadata, per-query metrics, aggregate
metrics and rankings. `per_query.csv`, `aggregate.csv`, and `rankings.csv` provide
matching tables. Null JSON values are blank CSV cells; a zero is a defined failure
score. Existing output directories fail closed. These CLI/raw reports remain
available alongside the GUI's optional Excel workbook. Benchmark report exports
omit raw queries, extracted text and absolute PDF paths, but query IDs and relative
PDF names may still identify personal material: a local model directory may also
appear in model metadata. Keep real runs private
and review reports before sharing or committing. Search-history exports have a
different scope: they include query text and document paths, as described above.
Only invented examples and tests belong in Git.

Metadata records schema/benchmark version, model name, package/Python/platform
versions, evaluator and search/storage source hashes, benchmark hash, settings,
counts and corpus fingerprint. The portable corpus fingerprint covers ordered
stored text/chunks and aligned embeddings, not original PDF bytes. Preserve the
frozen original corpus privately if exact PDF reproduction matters. A snapshot
cannot detect PDFs edited on disk after indexing: rebuild from the frozen corpus.
Native SQLite does not record tokenizer, chunk size, or model-weight revision.
The isolated `index` command writes a checked sidecar with known build settings;
legacy values remain explicitly unknown. Preserve local weights or use a fixed
local model directory for rigorous reproducibility; matching dimensions alone
cannot prove two weight revisions are identical.

Latency is one timed call per query including query encoding/transform, scoring,
sorting and collapse, with model loading/TF-IDF fitting recorded separately. It
includes first-query effects, has no warmup, and is **descriptive**, not a controlled
latency benchmark. Backend runs are sequential; cache/order and hardware can affect
measurements. There are no confidence intervals or significance claims built in.

For optional chunk-size experiments, build separate indexes with `--chunk-size`
60, 120 and 240 against the same frozen PDFs and dev queries. Run both systems on
each, verify corpus membership/text extraction and judgment coverage, and record
build time, chunk counts and latency. Chunk joins can change TF-IDF tokenization;
report that effect. Choose settings on dev, then run the frozen test benchmark
once. Automated sweeps are intentionally deferred in favor of metric correctness.

## Proposed Chapter 5 results

| Proposed table/figure | Export / content |
|---|---|
| 5.1 Dataset and protocol | PDF/query counts, dev/test split, label rubric, agreement, coverage, OCR/no-text counts, parameters, versions |
| 5.2 Retrieval comparison | System, macro P@1/5/10, R@1/5/10, MRR at depth, nDCG@5/10; eligible query counts |
| 5.3 Per-query cases | Paired system rows, target PDFs, failed paraphrases, ties, extraction failures; anonymized examples |
| 5.4 Efficiency | Setup/build time and descriptive mean/median latency with hardware and timing scope |
| Figure 5.1 | Paired per-query nDCG@10 plot; omit nulls and show sample size |
| Figure 5.2 | P@k and R@k curves over the recorded cutoffs for both methods |

Discuss semantic paraphrases versus exact terms, multi-relevant queries, max-chunk
versus full-document pooling, extraction/OCR failures, distractors, no-answer
queries and deterministic ties. Report macro and pooled recall separately. For
uncertainty estimates, use a paired bootstrap over the same eligible queries and
state seed/sample count; do not treat chunk hits as independent observations.
Avoid claiming superiority or statistical significance from the synthetic demo or
from differences lacking uncertainty analysis. If no real labeled benchmark is
available, Chapter 5 should say that retrieval effectiveness remains unmeasured.
