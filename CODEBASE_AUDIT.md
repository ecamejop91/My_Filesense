# FileSense codebase audit

Reviewed November 4, 2026, on branch `feature_doc_types` at `32e285a`.
This is a review report; the findings below have not been fixed or removed.

## Scope and method

Reviewed all **72 tracked files**: **54 Python files / 10,023 Python lines**,
the seven code cells and metadata in the notebook, launchers, requirements,
documentation, examples, ignore rules, and the binary spreadsheet fixture.
The 71 text files contain 11,376 physical lines, including notebook JSON,
comments, and blank lines. The spreadsheet was decoded with Calamine and is
also exercised by the document-format integration tests.

Read the complete first-party source and tests, traced imports and callers,
checked names used through Qt signals, properties, strings, and subprocess
entry points, and compared those results with execution of the test suite.
An AST scan found no production statements immediately following an
unconditional return/raise/break/continue in the same statement block. That
scan does not prove that every conditional branch is reachable.

The excluded local directories are `.git`, `.venv`, `.cache`, `__pycache__`,
generated `vector_store_notebook` data, and downloaded model contents. They
contain repository internals, dependencies, caches, private runtime data, or
binary weights rather than first-party source. Both distributed model bundles
were nevertheless hash-verified. No private library was cleared, renamed, or
reindexed for this review; behavioral reproductions used temporary synthetic
files and isolated databases.

"Unused" below means no consumer in this checkout after tracing these entry
points. Undocumented external Python callers cannot be ruled out. A function
that did not execute during tests is not automatically dead code.

## Findings that affect working features

### F1 — High: rename recovery can disconnect the file from the index

Locations: `document_query_visuals.py:693–726`,
`document_database.py:146–172`, `document_tag_store.py:566–572`.

The GUI renames the physical file, calls `rename_document_in_vector_store`
(which commits both the document path and existing tag paths), then calls
`rename_file_tags` again. Subsequent memory/UI refresh work is in the same
`try`. If anything after the database commit raises, the exception handler
renames only the physical file back. It never restores the database path.

**Reproduced:** created an indexed temporary `old.pdf`, accepted a rename to
`renamed.pdf`, and injected `OSError` into the follow-up `rename_file_tags` call.
The warning appeared and `old.pdf` existed again, but the committed document
record still pointed to the nonexistent `renamed.pdf`.

This can make document-opening links fail and leave search metadata inconsistent.
The duplicate tag update is live redundant code, not an unused function.
Make the database operation the single owner of tag/path changes; distinguish
pre-commit recovery from post-commit display errors and implement explicit
compensation if reverting a committed rename. Add a failure-after-commit
regression test. The existing GUI rename test mocks out the persistence call
and therefore cannot detect this inconsistency.

### F2 — Medium: new categories have no filter checkbox until a rebuild

Locations: `document_query_visuals.py:303–338`, `911–930`, `1013–1038`.

`save_tag_configuration` persists the definition and refreshes the category
display, but does not rebuild `self.tag_filters`. Analysis completion does not
rebuild them either. The checkboxes are built at startup and after a successful
library import.

**Reproduced:** saved `Audit category` using the real tag-definition persistence
function against an isolated database. The definition appeared in SQLite but
was absent from the current window's filter dictionary. Reanalysis does not
provide a checkbox; reopening the app does. Exact tag-name search remains an
available workaround after assignments exist.

Refresh the filter bar after a successful definition change, preserving checked
categories and the submitted-query state. The library-import flow already
contains an example of replacing that widget. No existing test exercises the
save-new-category-to-current-filter workflow.

### F3 — Medium: invalid rename paths escape the GUI warning handler

Location: `document_query_visuals.py:675–696`, especially line 685.

`file_path.with_name(new_name)` executes before the `try`. A name containing a
directory separator raises `ValueError` outside `_show_warning` handling.

**Reproduced:** supplied `nested/renamed.pdf` to the real method through a mocked
input dialog. `ValueError` escaped; no warning was displayed. The original
file was unchanged. Through a Qt callback this is an unhandled exception.

Validate a single filename inside the error-handling boundary, before any
filesystem change. Cover separators, dot names, and platform-invalid names.

### F4 — Medium: the check script reports completion despite failed checks

Location: `run_checks.ps1:15–37`.

`Run-Step` records `$process.ExitCode` but does not return a failing status,
throw, or set a script failure. The final green `Done` message runs regardless.
The install step also installs only the root requirements, although the full
suite directly imports `openpyxl` and exercises the evaluation dependencies
declared in `evaluation/requirements.txt`.

**Reproduced:** copied the unchanged `Run-Step` function into a temporary harness
and ran a Python child that exited with status 7. It printed `[exit code 7]`,
then the completion message, and the PowerShell harness exited with status 0.
The actual install/update script was not run, so the environment was not
changed for this experiment.

Collect step failures and exit nonzero, or stop on the first failure. Install
both maintained requirement files when promising to run the complete suite.

### F5 — Medium: worker diagnostics accumulate without reaching the user

Location: `evaluation/performance.py:58`, `107`, `127–174`.

Standard error and non-event standard output accumulate in `_diagnostics`,
bounded to 6,000 characters. That field is read only to append more output;
no view, log, export, or failure message consumes its content. If the worker
exits before sending a structured error, `_finished` presents only a generic
failure. This is a diagnostic dead end, rather than a reason to remove the
stdout/stderr readers: those readers also handle live process output/events.

Static evidence: all references to `_diagnostics` are initialization, reset,
and accumulation in this file. Surface the captured details on failures or
provide a diagnostic log. Keep document/query privacy in mind when choosing
where those details are shown or saved.

### F6 — Low: a zero search limit returns the full ranking

Location: `document_search_engine.py:516–535`.

`np.argsort(scores)[-top_k:]` becomes `[0:]` when `top_k=0`.
**Reproduced:** a one-chunk synthetic index returned one result for a zero
limit. Negative limits also produce arbitrary slices. The desktop's normal
nonempty-index path supplies a positive count, so this is a public-function
boundary defect, not an established normal-GUI failure.

Define the zero-limit contract and validate integer/nonnegative limits before
encoding the query. If zero is supported, return an empty result immediately.

## Confirmed unused implementation within this checkout

These symbols have no useful application consumer. The five functions/methods
below were also uncalled during the instrumented full test run. None is called
from any notebook cell, launcher, example, or documented command.

| Location | Symbol/state | Evidence and cleanup implication |
| --- | --- | --- |
| `document_search_engine.py:136–144` | `find_entry` | Only called by the unused `skip_entry`; no live root reaches this helper chain. Incremental indexing uses `by_path` and processing states instead. |
| `document_search_engine.py:147–148` | `skip_entry` | No caller anywhere in the checkout. |
| `document_search_engine.py:151–156` | `load_jsonl_records` | No caller. SQLite's actual legacy importer reads the files itself and does not use this function. |
| `document_search_engine.py:557–571` | `print_search_results` | No caller. The notebook duplicates its printing loop instead of importing it. |
| `document_tag_store.py:135–146` | `DocumentTagger.top_tags` | No caller. Live tagging uses `top_tags_many` or `top_tags_from_vectors`. |
| `document_tag_store.py:17–34` | `TAGS` | Unreferenced duplicate of the names in `TAG_PROTOTYPES`. Definitions and displayed names come from `load_tag_definitions` / `tag_names`. |
| `document_query_visuals.py:32–49` | imports `update_vector_store`, `tag_files` | Neither imported name is used in this module. `AnalysisWorker` imports and calls the real functions from their defining modules. |
| `document_query_visuals.py:94`, `200`, `242`, `986`, `1023` | `self.tag_data` | Written but never read by production code. Category rendering reads SQLite. One GUI test writes/asserts this obsolete attribute, so remove that assertion along with the state. Preserve the completed signal's signature if removing the assignment. |

These are small cleanup candidates, not evidence that the search/tagging
subsystems as a whole are unused. Prefer a separate behavior-preserving cleanup
commit, preserving any external API deliberately supported by the project.

## Files and APIs requiring a deliberate retention decision

| Item | Classification | Recommendation |
| --- | --- | --- |
| `evaluation/.Rhistory` | Empty tracked file, no references or runtime effect | Remove as editor/history residue. |
| `Indexing_and_vector_search.ipynb` | Standalone prototype; no desktop imports | Archive or label historical if still useful. It retains 60-token PDF-only extraction, no OCR, direct model loading, and JSONL/NumPy output under `vector_store_notebook`. Running it does not update the account-private desktop SQLite library. Do not describe it as unreachable: its seven cells form an executable workflow. |
| `document_search_engine.py:63`, `iter_pdf_files` | Unused internally, explicitly marked compatibility alias | Keep unless intentionally retiring external notebook/script compatibility. The tracked notebook defines its own function of this name. |
| `evaluation/ranking.py:21–28`, `mean_reciprocal_rank` | Called only by metric tests | Public pure metric helper. The runner computes MRR through `_mean` over reciprocal ranks; retaining a tested helper is reasonable. |
| `evaluation/tracking.py:134–184`, `SearchHistory.export` | Called by tests, not the current GUI | Keep: `evaluation/README.md:116` explicitly supports programmatic callers. Excel export replaced its GUI path, not its documented API. |
| `document_tag_store.py:149`, `extract_text` and text tagging fallbacks | Not executed during this suite, but reachable | Keep: standalone `tag_files` callers without stored vectors or supplied text reach this path. |
| `visualize_vectors.py` | Documented independent CLI entry point | Keep; synthetic empty/populated-index smoke checks passed. |
| `bundle_model.py`, `bundle_indra.py` | Independent setup/verification CLIs | Keep; both real local bundles verified successfully. |
| `tools/check_speedups.py` | Independent diagnostic script, called by `run_checks.ps1` | Keep. No normal desktop import is required. Its OCR failures are printed and swallowed, so its process status alone is not evidence that OCR succeeded. |
| root `ipykernel` and `matplotlib` requirements | Notebook and visualization tooling | Optional desktop-install split candidates, not globally unused packages. OCR's native and model-stack dependencies also must not be judged by direct imports alone. |

## Stale descriptions that lead readers to removed behavior

- `models/README.md:52–53` describes an OCR selector that remembers a choice.
  That selector is absent; `tests/test_analysis_progress.py` explicitly verifies
  its removal and the GUI passes `DEFAULT_OCR_MODEL` directly.
- `README.md:67` still says "Both OCR choices" although there is one accepted
  OCR model. `validate_ocr_model` rejects anything other than `indra`.
- `evaluation/README.md:159–162` says code inspection agrees with 60-token
  chunks. Current engine, CLI, and worker defaults are 200. Keep explicit
  60-token experimental CLI examples if intentional; correct the current-code
  description and distinguish it from the historical paper draft.
- `evaluation/__init__.py:1` says the package is never imported by the GUI.
  The GUI imports `evaluation.performance` and `evaluation.performance_dialog`.
  The CLI remains independent, but the package is now shared.

## Validation and practical limits

- Baseline: `.venv\Scripts\python.exe -m unittest discover -s tests -v`:
  **218 discovered, 217 passed, one POSIX-only test skipped**, 16.282 seconds.
  This Windows run includes the real Indra OCR smoke test.
- Repeated under a temporary standard-library line/call tracer:
  **218 discovered, 217 passed, one skipped**, 19.136 seconds. Execution counts
  in the inventory are unique Python line events, including definitions/imports;
  they are not statement/branch coverage percentages. Subprocesses and every
  native-created Qt thread are not comprehensively traced.
- All 54 Python files and all seven notebook code cells compile.
- `bundle_model.py --verify` and `bundle_indra.py --verify` passed against the
  actual local bundles; `python -m evaluation --help` passed.
- `visualize_vectors.main()` passed with synthetic empty/populated indexes and
  a noninteractive Matplotlib backend. No private index was plotted.
- Four focused behavioral reproductions confirmed F1, F2, F3, and F6. A separate
  isolated PowerShell harness confirmed F4. F5 is supported by the complete
  reference/caller review rather than a new worker-crash reproduction.
- macOS-native launch/plugin handling and real Word COM pagination were read
  but not executed on a Mac/Word installation during this audit. The tests
  exercise platform mocks and injected page readers. These are coverage gaps,
  not deletion candidates.
- No claim is made that a passing suite proves absence of other defects.

## Suggested change order

1. Fix rename transaction/recovery behavior and invalid-name handling (F1/F3),
   with real temporary SQLite/file regression cases.
2. Repair custom-tag filter refresh (F2) and check-script failure propagation
   (F4); expose worker diagnostic details (F5).
3. Remove the confirmed unused symbols/imports/state and empty history file in
   a separate cleanup commit. Resolve explicit compatibility/API decisions
   before deleting the retained candidates.
4. Define the search-limit boundary (F6), correct stale docs, and label or
   archive the prototype notebook. Split optional tooling dependencies only
   if simplifying desktop installation is desired.

## Complete tracked-file inventory

The following inventory records every file present at the reviewed commit.
"Observed lines" is from the instrumented suite in the main Python process,
not a usage verdict. A dash means non-Python; zero may mean a separate entry
point that the suite did not invoke. The report itself was added afterward.

| File | Physical lines | Observed lines | Role / disposition |
| --- | ---: | ---: | --- |
| `.gitignore` | 36 | - | Active exclusions for environments, models, generated/private data. |
| `Indexing_and_vector_search.ipynb` | 238 | - | Historical standalone prototype; retention decision and divergence documented above. |
| `README.md` | 407 | - | Active user guide; stale OCR-choice wording. |
| `Run FileSense.bat` | 5 | - | Windows entry point to the desktop app. |
| `Run FileSense.command` | 3 | - | macOS/zsh entry point; source reviewed, no native Mac execution. |
| `Run FileSense.ps1` | 4 | - | Windows PowerShell desktop entry point. |
| `START-HERE-MAC.md` | 80 | - | Active platform setup/launch guide. |
| `analysis_progress.py` | 207 | 157 | Active Qt analysis worker/progress/cancel workflow. |
| `app_storage.py` | 202 | 146 | Active private storage, permissions, cache and temporary-path setup. |
| `bundle_indra.py` | 21 | 0 | Independent OCR bundle CLI; verified against local weights. |
| `bundle_model.py` | 97 | 63 | Independent embedding bundle CLI; verified against local weights. |
| `document_database.py` | 313 | 200 | Active SQLite schema, migration and incremental persistence; rename coordination F1. |
| `document_formats.py` | 108 | 85 | Active Office/RTF readers and location/progress adapters. |
| `document_query_visuals.py` | 1141 | 694 | Desktop root; F1/F2/F3 plus unused imports and tag_data. |
| `document_search_engine.py` | 571 | 388 | Active indexing/search; four unused helpers, compatibility alias, limit defect F6. |
| `document_tag_store.py` | 572 | 333 | Active tagging/persistence; unused TAGS/top_tags; redundant follow-up tag edits F1. |
| `evaluation/.Rhistory` | 0 | - | Empty, unreferenced history residue; removal candidate. |
| `evaluation/.gitignore` | 3 | - | Active exclusions for local corpora and reports. |
| `evaluation/README.md` | 417 | - | Active evaluation/API guide; stale current chunk-size description. |
| `evaluation/__init__.py` | 3 | 2 | Version consumed by runner; stale GUI-import docstring. |
| `evaluation/__main__.py` | 117 | 90 | Independent demo/index/inspect/run CLI, used by integration tests. |
| `evaluation/backends.py` | 55 | 40 | Active FileSense and TF-IDF adapters. |
| `evaluation/benchmark.py` | 121 | 100 | Active benchmark validation and judgment binding. |
| `evaluation/corpus.py` | 158 | 124 | Active read-only corpus/ID normalization and ranking collapse. |
| `evaluation/demo.py` | 32 | 23 | Active synthetic corpus generator, used by CLI and worker. |
| `evaluation/examples/benchmark.csv` | 7 | - | Documented schema example, compared with JSON by tests. |
| `evaluation/examples/benchmark.json` | 18 | - | Documented schema example, compared with CSV by tests. |
| `evaluation/excel_report.py` | 392 | 310 | Active Excel export; consumed by PerformanceDialog. |
| `evaluation/gui_worker.py` | 287 | 214 | Active QProcess module entry point; tested in-process and as subprocess. |
| `evaluation/performance.py` | 212 | 151 | Active history/process controller; diagnostic sink F5. |
| `evaluation/performance_dialog.py` | 371 | 282 | Active performance UI, report rendering and Excel save workflows. |
| `evaluation/ranking.py` | 43 | 30 | Active ranking metrics; standalone MRR helper is test-only internally. |
| `evaluation/requirements.txt` | 5 | - | Active evaluation and Excel dependencies; omitted by run_checks install F4. |
| `evaluation/retrieval.py` | 32 | 18 | Active binary retrieval metrics. |
| `evaluation/runner.py` | 139 | 120 | Active metrics aggregation, provenance and raw report export. |
| `evaluation/tracking.py` | 184 | 122 | Active session history; documented legacy export API retained. |
| `indra_models.py` | 103 | 69 | Active local bundle selection, preparation and integrity checks. |
| `library_import.py` | 75 | 59 | Active explicit library import; called from Advanced menu. |
| `model_runtime.py` | 58 | 42 | Active process-local model/category cache. |
| `models/README.md` | 61 | - | Active packaging guide; stale removed-selector instructions. |
| `ocr.py` | 122 | 84 | Active Indra CPU adapter, rasterization, OCR fallback and signature. |
| `qt_runtime.py` | 43 | 14 | Active standard-stream and Mac Qt-plugin startup preparation. |
| `requirements.txt` | 28 | - | Active app/native-reader stack; notebook/plotting tooling are optional split candidates. |
| `run_checks.ps1` | 37 | - | Documented developer entry point; failure propagation/dependency issues F4. |
| `search_snippets.py` | 143 | 105 | Active query passage selection and bounded embedding cache. |
| `tests/fixtures/README.md` | 4 | - | Active provenance description for synthetic XLS fixture. |
| `tests/fixtures/invoice.xls` | binary | - | Used by document-format tests; decoded and checked as a two-sheet synthetic workbook. |
| `tests/test_analysis_progress.py` | 221 | 183 | Active unittest module; 7 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_app_storage.py` | 147 | 124 | Active unittest module; 12 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_bundled_model.py` | 111 | 95 | Active unittest module; 8 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_document_formats.py` | 164 | 141 | Active unittest module; 7 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_evaluation_integration.py` | 295 | 259 | Active unittest module; 20 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_evaluation_metrics.py` | 77 | 59 | Active unittest module; 10 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_evaluation_worker.py` | 277 | 243 | Active unittest module; 14 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_excel_export_gui.py` | 128 | 109 | Active unittest module; 4 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_excel_report.py` | 147 | 128 | Active unittest module; 6 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_indra_models.py` | 126 | 113 | Active unittest module; 8 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_library_import.py` | 82 | 72 | Active unittest module; 7 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_ocr_gui.py` | 31 | 28 | Active unittest module; 1 discovered test method; assertions/fixtures reviewed. |
| `tests/test_ocr_integration.py` | 163 | 141 | Active unittest module; 11 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_optimization.py` | 270 | 233 | Active unittest module; 15 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_performance_gui.py` | 325 | 262 | Active unittest module; 8 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_pipeline_speedups.py` | 224 | 188 | Active unittest module; 14 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_qt_runtime.py` | 36 | 29 | Active unittest module; 2 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_search_snippets.py` | 125 | 107 | Active unittest module; 10 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_search_tracking.py` | 211 | 187 | Active unittest module; 12 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_sqlite_integration.py` | 223 | 195 | Active unittest module; 16 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_tag_filter_gui.py` | 366 | 327 | Active unittest module; 18 discovered test methods; assertions/fixtures reviewed. |
| `tests/test_word_page_lookup.py` | 68 | 58 | Active unittest module; 8 discovered test methods; assertions/fixtures reviewed. |
| `tools/check_speedups.py` | 116 | 0 | Independent pipeline diagnostic invoked by run_checks; reviewed without full speed run. |
| `visualize_vectors.py` | 29 | 0 | Independent documented plot CLI; synthetic smoke checks passed. |
| `word_page_lookup.py` | 139 | 67 | Active Word page enrichment; real COM reader not executed in this review. |
