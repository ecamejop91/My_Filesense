"""Small, local text adapters for Office documents and Rich Text Format."""
from pathlib import Path
import re

SUPPORTED_EXTENSIONS = frozenset({'.pdf', '.docx', '.pptx', '.xlsx', '.xls', '.rtf'})
FILE_FILTER = 'Supported documents (' + ' '.join('*' + ext for ext in sorted(SUPPORTED_EXTENSIONS)) + ')'


def is_supported(path):
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def location_label(path, number):
    suffix = Path(path).suffix.lower()
    unit = {'.pdf': 'Page', '.pptx': 'Slide', '.xls': 'Sheet', '.xlsx': 'Sheet'}.get(suffix)
    return f'{unit} {number}' if unit else 'Document'


def progress_unit(path):
    return {'.pdf': 'Pages', '.pptx': 'Slides', '.xls': 'Sheets', '.xlsx': 'Sheets'}.get(
        Path(path).suffix.lower(), 'Documents')


def _word_text(container):
    for block in container.iter_inner_content():
        if hasattr(block, 'text'):
            yield block.text
        else:
            for row in block.rows:
                yield '\t'.join('\n'.join(_word_text(cell)) for cell in row.cells)


def _slide_text(shapes):
    for shape in shapes:
        if shape.has_text_frame:
            yield shape.text
        elif shape.has_table:
            yield from ('\t'.join(cell.text for cell in row.cells) for row in shape.table.rows)
        elif hasattr(shape, 'shapes'):
            yield from _slide_text(shape.shapes)


def _office_sections(path):
    suffix = path.suffix.lower()
    if suffix == '.docx':
        from docx import Document
        doc = Document(path)
        parts = list(_word_text(doc))
        seen = set()
        for section in doc.sections:
            for part in (section.header, section.footer):
                if part._element not in seen:
                    seen.add(part._element)
                    parts.extend(_word_text(part))
        yield 1, '\n'.join(parts)
    elif suffix == '.pptx':
        from pptx import Presentation
        for number, slide in enumerate(Presentation(path).slides, 1):
            parts = list(_slide_text(slide.shapes))
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame
                if notes is not None:
                    parts.append(notes.text)
            yield number, '\n'.join(parts)
    elif suffix in ('.xlsx', '.xls'):
        from python_calamine import CalamineWorkbook
        with CalamineWorkbook.from_path(str(path)) as workbook:
            for number, name in enumerate(workbook.sheet_names, 1):
                rows = ['\t'.join('' if value is None else str(value) for value in row)
                        for row in workbook.get_sheet_by_name(name).iter_rows()
                        if any(value not in ('', None) for value in row)]
                yield number, f'Sheet: {name}\n' + '\n'.join(rows) if rows else ''
    elif suffix == '.rtf':
        from striprtf.striprtf import rtf_to_text
        # Preserve raw bytes as RTF hex escapes so the parser applies the file's
        # declared codepage, including when the source uses raw 8-bit text.
        source = path.read_bytes().decode('latin-1')
        if not source.lstrip().startswith('{\\rtf'):
            raise ValueError('This file is not a valid RTF document')
        source = re.sub(r'[\x80-\xff]', lambda m: "\\'%02x" % ord(m[0]), source)
        yield 1, rtf_to_text(source)
    else:
        raise ValueError(f'Unsupported file type: {suffix}')


def _read_sections(path):
    try:
        yield from _office_sections(path)
    except ImportError as exc:
        raise RuntimeError('Document readers are missing. Run python -m pip install -r requirements.txt') from exc
    except Exception as exc:
        raise ValueError(f'Could not read {path.name}: {exc}') from exc


def extract_office_texts(path, progress=None):
    path = Path(path)
    pages = []
    number = 0
    for number, text in _read_sections(path):
        if progress:
            progress(dict(stage='extracting', path=str(path), current=number - 1, total=0,
                          unit=progress_unit(path), detail=f'Reading {location_label(path, number).lower()}'))
        if text.strip():
            pages.append((number, text.strip()))
    if progress:
        progress(dict(stage='extracting', path=str(path), current=number, total=number,
                      unit=progress_unit(path), detail='Document text extracted'))
    return ('text' if pages else 'none'), pages
