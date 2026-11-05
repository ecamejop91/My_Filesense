"""Opt-in, memory-only history of the results actually displayed by the GUI.

Recording performs no I/O or model calls. Only ``export`` writes files, and it
requires a new destination directory. Query text and document paths are retained;
document text, snippets, embeddings, and relevance judgments are never copied.
CSV text cells that could be interpreted as formulas are prefixed with an
apostrophe. JSON preserves the original text and is the authoritative export.
"""
from collections import deque
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


_HISTORY_FIELDS = (
    "search_id", "timestamp", "query", "kind", "elapsed_ms", "result_count",
    "quality_status",
)
_RESULT_FIELDS = (
    "search_id", "rank", "document", "path", "page", "score",
)


def _finite_number(value, name):
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _csv_value(value):
    # Check after whitespace as spreadsheet importers may strip it before
    # interpreting a formula. Leading tabs/newlines are guarded as well.
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n"))
        or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


class SearchHistory:
    """Bounded session history. Pass only the final, displayed result rows.

    Disabling pauses recording and leaves existing history intact. ``clear``
    removes retained searches without changing the enabled state. Results are
    preserved in display order, including multiple hits from the same document.
    Scores describe retrieval output; quality remains unevaluated because no
    relevance judgments are supplied here.
    """

    def __init__(self, limit=100):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        self._limit = limit
        self._enabled = False
        self._records = deque(maxlen=limit)
        self._next_id = 1

    @property
    def enabled(self):
        return self._enabled

    @property
    def limit(self):
        return self._limit

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        self._enabled = enabled

    @property
    def records(self):
        """Return a detached newest-first snapshot safe for callers to mutate."""
        return deepcopy(list(reversed(self._records)))

    def clear(self):
        self._records.clear()

    def record(self, query, kind, results, elapsed_ms):
        """Snapshot one completed search, or do nothing while disabled.

        ``kind`` is ``semantic`` or ``tag``. Semantic rows use ``path`` and
        ``page``; tag rows use ``file_path`` and have no page. Both use ``score``.
        Original ranks are replaced with 1-based displayed positions so tag
        filtering cannot leave gaps. No validation or result iteration occurs
        while recording is disabled.
        """
        if not self._enabled:
            return None
        if not isinstance(query, str):
            raise ValueError("query must be text")
        if kind not in ("semantic", "tag"):
            raise ValueError("kind must be 'semantic' or 'tag'")
        elapsed = _finite_number(elapsed_ms, "elapsed_ms")
        if elapsed < 0:
            raise ValueError("elapsed_ms must not be negative")

        hits = []
        path_key = "path" if kind == "semantic" else "file_path"
        for rank, row in enumerate(results, 1):
            path = str(row[path_key])
            page = row.get("page") if kind == "semantic" else None
            if page is not None:
                number = _finite_number(page, "page")
                if not number.is_integer() or number < 1:
                    raise ValueError("page must be a positive integer or None")
                page = int(number)
            hits.append(dict(
                rank=rank, document=Path(path).name, path=path, page=page,
                score=_finite_number(row["score"], "score"),
            ))

        record = dict(
            search_id=self._next_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query, kind=kind, elapsed_ms=elapsed,
            result_count=len(hits), results=hits,
            metrics={"quality_status": "not_evaluated"},
        )
        self._records.append(record)
        self._next_id += 1
        return deepcopy(record)

    def export(self, folder):
        """Export retained snapshots into exactly ``folder``, which must be new.

        Existing destinations are refused even when empty. CSV empty cells mean
        null (for example, tag results have no page). Export does not clear or
        enable history. The returned Path is the created destination directory.
        """
        records = self.records
        payload = dict(
            schema_version=1,
            exported_at=datetime.now(timezone.utc).isoformat(),
            scope={
                "storage": "session memory; explicit export only",
                "history_limit": self._limit,
                "record_count": len(records),
                "search_order": "newest first",
                "result_order": "displayed order after GUI filtering; no deduplication",
                "elapsed_ms": "elapsed milliseconds supplied by the GUI for the completed search",
                "scores": "raw retrieval scores; not relevance judgments or quality measurements",
                "quality_status": "not_evaluated",
            },
            privacy={
                "includes_query_text": True,
                "includes_document_paths": True,
                "includes_document_text": False,
                "includes_embeddings": False,
                "network_upload": False,
                "csv_text_policy": "formula-like text is prefixed with an apostrophe; JSON preserves original text",
            },
            records=records,
        )
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / "search_history.json").open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        with (folder / "search_history.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_HISTORY_FIELDS)
            writer.writeheader()
            for record in records:
                row = {key: record[key] for key in _HISTORY_FIELDS if key != "quality_status"}
                row["quality_status"] = record["metrics"]["quality_status"]
                writer.writerow({key: _csv_value(value) for key, value in row.items()})
        with (folder / "search_results.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_RESULT_FIELDS)
            writer.writeheader()
            for record in records:
                for hit in record["results"]:
                    row = dict(search_id=record["search_id"], **hit)
                    writer.writerow({key: _csv_value(value) for key, value in row.items()})
        return folder
