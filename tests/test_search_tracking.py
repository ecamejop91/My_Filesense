"""Session tracking fixtures; no GUI, model, index, or third-party imports."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.tracking import SearchHistory


def semantic_rows():
    return [
        dict(rank=2, document="first.pdf", path="/private/first.pdf", page=2,
             score=.75, text="SOURCE TEXT MUST NOT BE COPIED"),
        dict(rank=5, document="first.pdf", path="/private/first.pdf", page=4,
             score=.72, text="SECOND SOURCE TEXT"),
        dict(rank=8, document="other.pdf", path="/private/other.pdf", page=1,
             score=-.1, text="THIRD SOURCE TEXT"),
    ]


class SearchTrackingTests(unittest.TestCase):
    def setUp(self):
        self.history = SearchHistory()

    def test_disabled_by_default_does_not_even_iterate_results(self):
        def unexpected_iteration():
            raise AssertionError("disabled tracking must not inspect results")
            yield
        self.assertFalse(self.history.enabled)
        self.assertIsNone(self.history.record(None, "invalid", unexpected_iteration(), float("nan")))
        self.assertEqual(self.history.records, [])

    def test_toggle_pauses_without_clearing(self):
        self.history.set_enabled(True)
        record = self.history.record("first", "semantic", [], 0)
        self.history.set_enabled(False)
        self.assertIsNone(self.history.record("ignored", "semantic", [], 1))
        self.assertEqual(self.history.records, [record])
        self.history.set_enabled(True)
        self.history.record("second", "semantic", [], 2)
        self.assertEqual([row["query"] for row in self.history.records], ["second", "first"])

    def test_bounded_history_clear_and_monotonic_ids(self):
        history = SearchHistory(limit=2)
        history.set_enabled(True)
        for query in ("one", "two", "three"):
            history.record(query, "semantic", [], 1)
        self.assertEqual(history.limit, 2)
        self.assertEqual([r["query"] for r in history.records], ["three", "two"])
        history.clear()
        self.assertEqual(history.records, [])
        self.assertTrue(history.enabled)
        self.assertEqual(history.record("four", "tag", [], 3)["search_id"], 4)

    def test_semantic_hits_preserve_display_order_duplicates_and_raw_score(self):
        self.history.set_enabled(True)
        record = self.history.record("question", "semantic", semantic_rows(), 12.5)
        self.assertEqual(record["elapsed_ms"], 12.5)
        self.assertEqual(record["result_count"], 3)
        self.assertEqual([hit["rank"] for hit in record["results"]], [1, 2, 3])
        self.assertEqual([hit["page"] for hit in record["results"]], [2, 4, 1])
        self.assertEqual([hit["score"] for hit in record["results"]], [.75, .72, -.1])
        self.assertEqual(record["results"][0]["document"], "first.pdf")
        self.assertEqual(record["results"][0]["path"], "/private/first.pdf")
        self.assertEqual(record["metrics"], {"quality_status": "not_evaluated"})
        self.assertNotIn("text", record["results"][0])
        self.assertTrue(record["timestamp"].endswith("+00:00"))

    def test_tag_hits_and_empty_searches(self):
        self.history.set_enabled(True)
        record = self.history.record("Invoice", "tag", [
            dict(file_name="invoice.pdf", file_path=Path("/local/invoice.pdf"),
                 score=.9, matched_tag="Invoice", all_tags=["Invoice", "Finance"]),
        ], 1)
        self.assertEqual(record["results"], [dict(
            rank=1, document="invoice.pdf", path=str(Path('/local/invoice.pdf')), page=None, score=.9,
        )])
        for kind in ("semantic", "tag"):
            empty = self.history.record("no match", kind, [], 1)
            self.assertEqual(empty["result_count"], 0)
            self.assertEqual(empty["results"], [])

    def test_snapshots_are_isolated_from_inputs_return_values_and_readers(self):
        self.history.set_enabled(True)
        inputs = semantic_rows()
        returned = self.history.record("original", "semantic", inputs, 1)
        inputs[0]["path"] = "changed.pdf"
        inputs.clear()
        returned["results"][0]["path"] = "changed again.pdf"
        returned["metrics"]["quality_status"] = "invented"
        snapshot = self.history.records
        snapshot[0]["results"].clear()
        snapshot[0]["query"] = "changed"
        snapshot.clear()
        actual = self.history.records[0]
        self.assertEqual(actual["query"], "original")
        self.assertEqual(len(actual["results"]), 3)
        self.assertEqual(actual["results"][0]["path"], "/private/first.pdf")
        self.assertEqual(actual["metrics"]["quality_status"], "not_evaluated")

    def test_recording_does_no_file_io(self):
        self.history.set_enabled(True)
        with patch("builtins.open", side_effect=AssertionError("no file I/O")), \
                patch.object(Path, "open", side_effect=AssertionError("no file I/O")), \
                patch.object(Path, "resolve", side_effect=AssertionError("no filesystem resolution")):
            self.history.record("query", "semantic", semantic_rows(), 1)
            self.history.records
            self.history.clear()

    def test_invalid_numeric_values_do_not_append_partial_records(self):
        self.history.set_enabled(True)
        for invalid in (float("nan"), float("inf"), -float("inf"), True, "2"):
            with self.subTest(elapsed_ms=invalid), self.assertRaises(ValueError):
                self.history.record("q", "semantic", [], invalid)
            rows = semantic_rows()
            rows[-1]["score"] = invalid
            with self.subTest(score=invalid), self.assertRaises(ValueError):
                self.history.record("q", "semantic", rows, 1)
        with self.assertRaises(ValueError):
            self.history.record("q", "semantic", [], -1)
        for invalid in (0, -1, 1.5, float("nan"), True):
            rows = semantic_rows()
            rows[-1]["page"] = invalid
            with self.subTest(page=invalid), self.assertRaises(ValueError):
                self.history.record("q", "semantic", rows, 1)
        self.assertEqual(self.history.records, [])
        self.assertEqual(self.history.record("valid", "semantic", [], 1)["search_id"], 1)

    def test_invalid_arguments(self):
        for invalid in (0, -1, 2.5, True, "3"):
            with self.subTest(limit=invalid), self.assertRaises(ValueError):
                SearchHistory(limit=invalid)
        with self.assertRaises(ValueError):
            self.history.set_enabled(1)
        self.history.set_enabled(True)
        with self.assertRaises(ValueError):
            self.history.record("q", "search", [], 1)
        with self.assertRaises(ValueError):
            self.history.record(None, "semantic", [], 1)

    def test_export_csv_json_schema_privacy_and_nonoverwrite(self):
        self.history.set_enabled(True)
        self.history.record("first query", "semantic", semantic_rows(), 12.25)
        self.history.record("=HYPERLINK(\"example\")", "tag", [], 0)
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "export"
            self.assertFalse(folder.exists())
            self.assertEqual(self.history.export(folder), folder)
            with (folder / "search_history.csv").open(newline="", encoding="utf-8") as handle:
                searches = list(csv.DictReader(handle))
            with (folder / "search_results.csv").open(newline="", encoding="utf-8") as handle:
                hits = list(csv.DictReader(handle))
            raw_json = (folder / "search_history.json").read_text(encoding="utf-8")
            exported = json.loads(raw_json)
            self.assertEqual(len(searches), 2)
            self.assertEqual(len(hits), 3)
            self.assertEqual(searches[0]["result_count"], "0")
            self.assertEqual(searches[0]["query"], "'=HYPERLINK(\"example\")")
            self.assertEqual(exported["records"][0]["query"], '=HYPERLINK("example")')
            self.assertEqual(exported["records"], self.history.records)
            self.assertEqual([hit["rank"] for hit in hits], ["1", "2", "3"])
            self.assertTrue(all(hit["search_id"] == "1" for hit in hits))
            self.assertEqual(searches[0]["quality_status"], "not_evaluated")
            self.assertTrue(exported["privacy"]["includes_query_text"])
            self.assertTrue(exported["privacy"]["includes_document_paths"])
            self.assertFalse(exported["privacy"]["includes_document_text"])
            self.assertFalse(exported["privacy"]["network_upload"])
            self.assertNotIn("SOURCE TEXT", raw_json)
            self.assertNotIn("precision", raw_json)
            self.assertNotIn("recall", raw_json)
            before = {file.name: file.read_bytes() for file in folder.iterdir()}
            with self.assertRaises(FileExistsError):
                self.history.export(folder)
            self.assertEqual(before, {file.name: file.read_bytes() for file in folder.iterdir()})
            self.assertEqual(len(self.history.records), 2)
            self.assertTrue(self.history.enabled)

    def test_empty_export_headers_and_existing_empty_directory_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError):
                self.history.export(root)
            folder = self.history.export(Path(root) / "empty")
            for name in ("search_history.csv", "search_results.csv"):
                with (folder / name).open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertIn("search_id", reader.fieldnames)
                    self.assertEqual(list(reader), [])
            self.assertEqual(json.loads((folder / "search_history.json").read_text())["records"], [])
            self.assertFalse(self.history.enabled)

    def test_tag_csv_page_is_empty_and_formula_text_escaped(self):
        self.history.set_enabled(True)
        self.history.record("  @SUM(1)", "tag", [dict(
            file_path="+filename.pdf", score=.9,
        )], 1)
        with tempfile.TemporaryDirectory() as root:
            folder = self.history.export(Path(root) / "tag")
            with (folder / "search_results.csv").open(newline="", encoding="utf-8") as handle:
                hit = next(csv.DictReader(handle))
            self.assertEqual(hit["page"], "")
            self.assertEqual(hit["document"], "'+filename.pdf")
            self.assertEqual(hit["path"], "'+filename.pdf")
            with (folder / "search_history.csv").open(newline="", encoding="utf-8") as handle:
                search = next(csv.DictReader(handle))
            self.assertEqual(search["query"], "'  @SUM(1)")


if __name__ == "__main__":
    unittest.main()
