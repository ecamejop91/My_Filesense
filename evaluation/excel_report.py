"""Excel snapshots for the desktop app; no searches, metric calculations, or downloads.

openpyxl is imported only when exporting. Existing CLI JSON/CSV exports remain
independent. Every cell stores a value, including formula-like source strings.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import tempfile


EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384
EXCEL_MAX_CELL_CHARS = 32_767
EXCEL_MAX_LINE_BREAKS = 253
DATE_FORMAT = 'yyyy-mm-dd hh:mm:ss.000 "UTC"'
_INVALID_XML = re.compile('[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]')
_GREEN = '205D4C'
_PALE_GREEN = 'EEF5F0'
_TEXT = '20362D'


@dataclass
class _Sheet:
    name: str
    title: str
    notes: list
    headers: list
    rows: list
    widths: dict = None

    @property
    def header_row(self):
        return len(self.notes) + 3


def _utc_datetime(value, label):
    if value is None:
        return None
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('timezone missing')
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        if parsed.year < 1900:
            raise ValueError('date predates Excel support')
        return parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a date from 1900 onward with an explicit time zone.') from exc


def _value(value):
    if value is None:
        return 'N/A'
    if isinstance(value, str):
        if _INVALID_XML.search(value):
            raise ValueError('Excel cannot store a text value containing invalid XML control characters.')
        if len(value.encode('utf-16-le')) // 2 > EXCEL_MAX_CELL_CHARS:
            raise ValueError(f'An Excel cell exceeds the {EXCEL_MAX_CELL_CHARS:,}-character limit. '
                             'Shorten the source text before exporting.')
        if value.count('\n') > EXCEL_MAX_LINE_BREAKS:
            raise ValueError(f'An Excel cell exceeds the {EXCEL_MAX_LINE_BREAKS} line-break limit.')
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        # Excel numbers have 15 significant digits. Keep larger identifiers and
        # metadata integers exact as text instead of silently rounding them.
        return int(value) if abs(value) < 10 ** 15 else str(value)
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError('Excel export requires finite numbers; use None for undefined values.')
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        if value.year < 1900:
            raise ValueError('Excel dates must be from 1900 onward.')
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return _value(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f'Excel cannot represent this structured field: {exc}') from exc
    raise ValueError(f'Excel cannot represent a value of type {type(value).__name__}.')


def _validate(sheets):
    for sheet in sheets:
        if not sheet.headers or len(sheet.headers) > EXCEL_MAX_COLUMNS:
            raise ValueError(f'{sheet.name} exceeds the Excel limit of {EXCEL_MAX_COLUMNS:,} columns.')
        if sheet.header_row + len(sheet.rows) > EXCEL_MAX_ROWS:
            raise ValueError(f'{sheet.name} exceeds the Excel limit of {EXCEL_MAX_ROWS:,} rows, '
                             'including titles and headers. Export fewer records.')
        _value(sheet.title)
        for note in sheet.notes:
            _value(note)
        for header in sheet.headers:
            if not isinstance(header, str) or not header:
                raise ValueError('Excel column names must be nonempty text.')
            _value(header)
        for row in sheet.rows:
            if len(row) != len(sheet.headers):
                raise ValueError(f'{sheet.name} has a row with the wrong number of columns.')
            for value in row:
                _value(value)


def _table_fields(rows, default_headers):
    fields = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Report rows must be mappings of field names to values.')
        for key in row:
            if key not in fields:
                fields.append(key)
    fields = fields or default_headers
    return fields, [[row.get(key) for key in fields] for row in rows]


def _metadata_rows(value, path='', ancestors=None):
    """JSON Pointer paths avoid collisions between punctuation in source keys."""
    ancestors = set() if ancestors is None else ancestors
    if isinstance(value, (dict, list, tuple)):
        if id(value) in ancestors:
            raise ValueError('Report metadata must not contain circular references.')
        branch = ancestors | {id(value)}
        if not value:
            yield [path, 'object' if isinstance(value, dict) else 'array', '{}' if isinstance(value, dict) else '[]']
        else:
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, item in items:
                if isinstance(value, dict) and not isinstance(key, str):
                    raise ValueError('Metadata object keys must be text.')
                escaped = str(key).replace('~', '~0').replace('/', '~1')
                yield from _metadata_rows(item, path + '/' + escaped, branch)
    else:
        kind = ('null' if value is None else 'boolean' if isinstance(value, bool) else
                'integer' if isinstance(value, Integral) else 'number' if isinstance(value, Real) else
                'datetime' if isinstance(value, datetime) else 'string' if isinstance(value, str) else None)
        if kind is None:
            raise ValueError(f'Unsupported metadata type: {type(value).__name__}.')
        yield [path, kind, value]


def _number_format(field, value):
    if isinstance(value, datetime):
        return DATE_FORMAT
    if isinstance(value, str):
        return '@'
    normalized = field.lower().replace(' ', '_')
    if normalized.startswith(('precision', 'recall', 'macro_precision', 'macro_recall',
                              'pooled_precision', 'pooled_recall')):
        return '0.0%'
    if 'latency' in normalized or '(ms)' in normalized:
        return '0.00'
    if any(metric in normalized for metric in ('score', 'dcg', 'mrr', 'reciprocal_rank')):
        # eligible_recall_rr_ndcg_queries is a count, not a ranking metric.
        if normalized.endswith('queries'):
            return '#,##0'
        return '0.000'
    return '#,##0' if isinstance(value, Integral) and not isinstance(value, bool) else '0.000' if isinstance(value, Real) else 'General'


def _width(header):
    lower = header.lower()
    if lower in ('query', 'text'):
        return 54
    if lower in ('path', 'document_id', 'pdf', 'document'):
        return 54
    if 'time' in lower and 'latency' not in lower:
        return 30
    if 'query_id' in lower:
        return 26
    return min(32, max(15, len(header) + 2))


def _build_workbook(sheets):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise ImportError('Excel export requires openpyxl. In the FileSense Python environment, '
                          'run: python -m pip install -r evaluation/requirements.txt') from exc

    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.creator = 'FileSense'
    workbook.properties.title = sheets[0].title
    body_font = Font(name='Arial', size=11, color=_TEXT)
    heading_font = Font(name='Arial', size=11, color='FFFFFF', bold=True)
    title_font = Font(name='Arial', size=16, color=_GREEN, bold=True)
    fill = PatternFill('solid', fgColor=_GREEN)
    stripe = PatternFill('solid', fgColor=_PALE_GREEN)

    def write(cell, value, field=''):
        value = _value(value)
        cell.value = value
        # Force all source strings to literal text, including =, #N/A, URLs,
        # and strings which spreadsheet applications might treat as formulas.
        if isinstance(value, str):
            cell.data_type = 's'
        cell.hyperlink = None
        cell.font = body_font
        cell.number_format = _number_format(field, value)
        cell.alignment = Alignment(vertical='top', horizontal='right' if (
            isinstance(value, Real) and not isinstance(value, bool)) or value == 'N/A' else 'left',
            wrap_text=True)

    for plan in sheets:
        sheet = workbook.create_sheet(plan.name)
        sheet.sheet_view.showGridLines = False
        sheet.sheet_view.zoomScale = 90
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.sheet_properties.tabColor = _GREEN if plan.name == 'Summary' else '78A18C'
        write(sheet.cell(1, 1), plan.title)
        if len(plan.headers) > 1:
            sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(plan.headers))
        sheet.cell(1, 1).font = title_font
        sheet.row_dimensions[1].height = 27
        last_column = get_column_letter(len(plan.headers))
        for row_number, note in enumerate(plan.notes, 2):
            # Only explanatory rows span the table; all data cells stay unmerged.
            if len(plan.headers) > 1:
                sheet.merge_cells(start_row=row_number, start_column=1,
                                  end_row=row_number, end_column=len(plan.headers))
            write(sheet.cell(row_number, 1), note)
            sheet.row_dimensions[row_number].height = 34 if len(note) > 100 else 26
        widths = [((plan.widths or {}).get(header, _width(header))) for header in plan.headers]
        for column, (header, width) in enumerate(zip(plan.headers, widths), 1):
            cell = sheet.cell(plan.header_row, column)
            write(cell, header)
            cell.font = heading_font
            cell.fill = fill
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(right=Side(style='thin', color='FFFFFF'))
            sheet.column_dimensions[get_column_letter(column)].width = width
        sheet.row_dimensions[plan.header_row].height = 42
        for offset, row in enumerate(plan.rows, plan.header_row + 1):
            line_count = 1
            for column, (header, value) in enumerate(zip(plan.headers, row), 1):
                cell = sheet.cell(offset, column)
                write(cell, value, header)
                if offset % 2 == 0:
                    cell.fill = stripe
                if isinstance(cell.value, str):
                    line_count = max(line_count, sum(max(1, math.ceil(len(line) / max(8, widths[column - 1] - 2)))
                                                    for line in cell.value.split('\n')))
            sheet.row_dimensions[offset].height = min(150, max(23, line_count * 16 + 6))
        last_row = plan.header_row + len(plan.rows)
        sheet.auto_filter.ref = f'A{plan.header_row}:{last_column}{last_row}'
        sheet.freeze_panes = f'B{plan.header_row + 1}'
        sheet.print_title_rows = f'{plan.header_row}:{plan.header_row}'
        sheet.print_options.horizontalCentered = True
        sheet.page_setup.orientation = 'landscape'
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.print_area = f'A1:{last_column}{last_row}'
    return workbook


def _export(sheets, output_file):
    output = Path(output_file).expanduser().absolute()
    if output.suffix.lower() != '.xlsx':
        raise ValueError('Choose an output filename ending in .xlsx.')
    if os.path.lexists(output):
        raise FileExistsError('The Excel file already exists. Choose a new filename; existing files are never replaced.')
    _validate(sheets)
    workbook = _build_workbook(sheets)
    temporary = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix='.filesense-', suffix='.xlsx', dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        workbook.save(temporary)
        # Windows _commit (used by fsync) requires a writable descriptor.
        with temporary.open('r+b') as handle:
            os.fsync(handle.fileno())
        # Linking a complete sibling file publishes it atomically and fails if
        # the chosen name was created meanwhile. No partial target is exposed.
        os.link(temporary, output)
    finally:
        workbook.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def export_history_workbook(records, output_file):
    """Export every retained search and displayed result, preserving their order."""
    records = list(records)
    searches, results, seen = [], [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Search history records must be mappings.')
        try:
            search_id = record['search_id']
            if isinstance(search_id, bool) or not isinstance(search_id, (str, Integral)):
                raise ValueError('Search IDs must be text or integers.')
            join_id = str(search_id)
            if not join_id or join_id in seen:
                raise ValueError('Each search must have a unique, nonempty search ID.')
            seen.add(join_id)
            timestamp = _utc_datetime(record['timestamp'], 'Search timestamp')
            hits = record['results']
            if record['result_count'] != len(hits):
                raise ValueError('A search result count does not match its displayed result rows.')
            searches.append([search_id, timestamp, record['query'], record['kind'],
                             record['elapsed_ms'], record['result_count'],
                             record.get('metrics', {}).get('quality_status', 'not_evaluated')])
            for hit in hits:
                results.append([search_id, record['query'], timestamp, record['kind'],
                                hit['rank'], hit['document'], hit['path'], hit['page'], hit['score']])
        except (KeyError, TypeError) as exc:
            raise ValueError(f'Search history is incomplete: {exc}') from exc
    summary = [
        ['Exported (UTC)', datetime.now(timezone.utc)],
        ['Searches', len(records)],
        ['Displayed result rows', len(results)],
        ['Searches without results', sum(record['result_count'] == 0 for record in records)],
        ['Quality', 'Not evaluated. Similarity and tag scores do not establish relevance.'],
        ['Storage', 'Snapshot of session memory. Query text and file paths are included.'],
        ['Ordering', 'Searches retain session order. Results retain displayed order, including repeated PDFs.'],
        ['Time', 'Dates are UTC. Response time includes the completed search and preparing its display.'],
        ['Missing values', 'N/A means unavailable or not applicable. Numeric zero remains zero.'],
        ['Joining sheets', 'Search ID joins Searches to Results. Empty searches remain on Searches.'],
    ]
    return _export([
        _Sheet('Summary', 'FileSense search history', ['Search behavior from this session. Quality has not been evaluated.'],
               ['Measure', 'Value'], summary, {'Measure': 31, 'Value': 100}),
        _Sheet('Searches', 'Search history', ['One row per search, including searches with no displayed results. All dates are UTC.'],
               ['Search ID', 'Time (UTC)', 'Query', 'Search type', 'Response (ms)', 'Displayed results', 'Quality status'], searches),
        _Sheet('Results', 'Displayed search results', ['One row per displayed result. Search ID and Query identify its search. Repeated PDFs are preserved.'],
               ['Search ID', 'Query', 'Time (UTC)', 'Search type', 'Rank', 'PDF', 'Path', 'Page', 'Score'], results),
    ], output_file)


def export_evaluation_workbook(result, output_file):
    """Copy evaluator values into an Excel report without recomputing any metric."""
    try:
        metadata = result['metadata']
        aggregates, per_query, rankings = result['aggregate'], result['per_query'], result['rankings']
        cutoffs, depth = metadata['cutoffs'], metadata['depth']
        if not isinstance(metadata, dict) or not all(isinstance(rows, list) for rows in (aggregates, per_query, rankings)):
            raise ValueError('Evaluation metadata must be a mapping and report tables must be lists.')
        if not cutoffs or any(type(k) is not int or k < 1 for k in cutoffs):
            raise ValueError('Evaluation cutoffs must contain positive integers.')
        if type(depth) is not int or depth < max(cutoffs):
            raise ValueError('Evaluation depth must be at least every reported cutoff.')
        k = max(cutoffs)
        headers = ['System', 'Queries', f'Macro precision@{k}', f'Macro recall@{k}', f'MRR@{depth}',
                   f'Macro nDCG@{k}', f'Pooled precision@{k}', f'Pooled recall@{k}', 'Mean query (ms)']
        summary = [[row.get('system'), row.get('queries'), row.get(f'macro_precision@{k}'),
                    row.get(f'macro_recall@{k}'), row.get('mrr_at_depth'), row.get(f'macro_ndcg@{k}'),
                    row.get(f'pooled_precision@{k}'), row.get(f'pooled_recall@{k}'), row.get('mean_latency_ms')]
                   for row in aggregates]
        aggregate_fields, aggregate_rows = _table_fields(aggregates, ['system'])
        query_fields, query_rows = _table_fields(per_query, ['system', 'query_id'])
        ranking_fields, ranking_rows = _table_fields(rankings, ['system', 'query_id', 'rank', 'document_id', 'score', 'relevance', 'judged'])
        context = (f"Benchmark: {metadata.get('benchmark_id', 'N/A')}. Split: {metadata.get('split', 'N/A')}. "
                   f"PDFs: {metadata.get('documents', 'N/A')}. Retrieval depth: {depth}. "
                   f"Reported cutoffs: {', '.join(map(str, cutoffs))}.")
        sample_note = ('SAMPLE DATA: Synthetic smoke test. These scores demonstrate the workflow and do not establish '
                       'retrieval quality on real documents.' if metadata.get('synthetic_demo') else
                       'Quality results for this labeled benchmark and frozen indexed collection.')
        notes = [sample_note, context,
                 'Macro averages give each eligible query equal weight. Pooled precision and recall combine counts across queries.',
                 'All metrics are copied from the evaluator. N/A marks undefined or unavailable values; zero remains zero. '
                 'Aggregate and Per query preserve every metric and cutoff.',
                 str(metadata.get('timing_note') or 'Timing is a local observation from one run, not a definitive speed comparison. '
                     'Query timing excludes model loading, indexing, and TF-IDF fitting.')]
        metadata_rows = list(_metadata_rows({key: value for key, value in result.items()
                                             if key not in ('aggregate', 'per_query', 'rankings')}))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'Evaluation report is incomplete: {exc}') from exc
    return _export([
        _Sheet('Summary', 'FileSense evaluation', notes, headers, summary),
        _Sheet('Aggregate', 'Complete aggregate metrics', ['Every aggregate field and cutoff is copied from the evaluator. Macro and pooled values retain their original meanings.'],
               aggregate_fields, aggregate_rows),
        _Sheet('Per query', 'Per-query metrics', ['Every query and metric is preserved. N/A means undefined or unavailable, not zero.'], query_fields, query_rows),
        _Sheet('Rankings', 'Ranked PDFs', ['Document rankings and scores as returned by the evaluator. Scores from different systems use different scales.'],
               ranking_fields, ranking_rows),
        _Sheet('Metadata', 'Evaluation metadata and provenance', ['Paths use JSON Pointer notation: ~1 represents / and ~0 represents ~ in a key. Types distinguish null from literal N/A text.'],
               ['Path', 'Type', 'Value'], metadata_rows, {'Path': 76, 'Type': 16, 'Value': 100}),
    ], output_file)
