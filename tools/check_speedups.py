"""Measure the PDF pipeline changes in this environment.

Run with the project's interpreter, e.g. .venv\\Scripts\\python.exe tools\\check_speedups.py
It prints:
  1. how many model word-pieces a 200-token chunk becomes (MiniLM keeps 256);
  2. Indra OCR seconds per page on a synthetic scanned PDF.
Nothing is written to the FileSense index.
"""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def chunk_fit():
    import tiktoken
    from transformers import AutoTokenizer
    import document_search_engine as engine
    from model_runtime import BUNDLED_MODEL_DIR

    source = str(BUNDLED_MODEL_DIR) if BUNDLED_MODEL_DIR.exists() else 'sentence-transformers/all-MiniLM-L6-v2'
    wordpiece = AutoTokenizer.from_pretrained(source)
    limit = 256
    encoding = tiktoken.get_encoding(engine.ENCODING_NAME)
    samples = {
        'README prose': (ROOT / 'README.md').read_text(encoding='utf-8'),
        'invoice-like': '\n'.join(
            f'Invoice INV-{40000 + i} dated 2026-{i % 12 + 1:02d}-{i % 28 + 1:02d} '
            f'Qty {i % 7 + 1} x SKU-{i * 37 % 9973:04d} @ ${i * 13.37:,.2f} subtotal ${i * 91.5:,.2f} '
            f'tax 8.25% account 0045-{i * 7919 % 100000:05d}' for i in range(400)),
        'legal-like': ('The Lessee shall indemnify and hold harmless the Lessor, its successors and assigns, '
                       'from any and all claims, liabilities, damages, costs and expenses (including '
                       'reasonable attorneys’ fees) arising under Section 12(b)(iii) hereof. ') * 40,
    }
    print(f'Chunk size: {engine.CHUNK_SIZE} {engine.ENCODING_NAME} tokens; model window: {limit} word-pieces')
    for name, text in samples.items():
        ids = encoding.encode(text)
        lengths = []
        for start in range(0, len(ids), engine.CHUNK_SIZE):
            chunk = encoding.decode(ids[start:start + engine.CHUNK_SIZE])
            lengths.append(len(wordpiece(chunk)['input_ids']))  # includes [CLS]/[SEP]
        lengths.sort()
        over = sum(length > limit for length in lengths)
        print(f'  {name:13s} chunks={len(lengths):4d} median={lengths[len(lengths) // 2]:3d} '
              f'max={lengths[-1]:3d} over_limit={over}')


def scanned_pdf(folder, pages=3):
    import pymupdf
    lines = [f'Invoice 47{n:02d}   Customer account 0045-{n * 313:05d}   Amount due ${n * 127.45:,.2f}'
             for n in range(1, 31)]
    text_pdf, scan = folder / 'text.pdf', folder / 'scan.pdf'
    with pymupdf.open() as doc:
        for p in range(pages):
            page = doc.new_page(width=612, height=792)
            page.insert_textbox(pymupdf.Rect(54, 54, 558, 738), f'Page {p + 1}\n' + '\n'.join(lines), fontsize=11)
        doc.save(text_pdf)
    with pymupdf.open(text_pdf) as doc, pymupdf.open() as out:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            out.new_page(width=612, height=792).insert_image(pymupdf.Rect(0, 0, 612, 792), pixmap=pix)
        out.save(scan)
    return scan


def time_ocr():
    import ocr
    ocr._get_engine.cache_clear()
    try:
        start = time.perf_counter()
        name = ocr.ocr_backend_name()
        load = time.perf_counter() - start
        ocr.extract_page_texts_with_ocr(SCAN)  # warm-up
        start = time.perf_counter()
        pages = ocr.extract_page_texts_with_ocr(SCAN)
        elapsed = time.perf_counter() - start
        found = sum('4729' in text or 'Invoice' in text for _, text in pages)
        print(f'  {name}, {ocr.OCR_DPI} DPI -> {elapsed / max(len(pages), 1):.2f} s/page '
              f'(model load {load:.1f} s; {found}/{len(pages)} pages recognized)')
    except Exception as exc:
        print(f'  Indra: unavailable ({type(exc).__name__}: {exc})')
    finally:
        ocr._get_engine.cache_clear()


def versions():
    import importlib.metadata as metadata
    print(sys.version.split()[0], sys.platform)
    for name in ('indra-ocr', 'onnxruntime', 'opencv-python-headless', 'pymupdf', 'sentence-transformers', 'tiktoken'):
        try:
            print(f'  {name}: {metadata.version(name)}')
        except metadata.PackageNotFoundError:
            print(f'  {name}: not installed')
    import cv2
    print(f'  cv2 import OK ({cv2.__version__})')


if __name__ == '__main__':
    print('== Environment ==')
    try:
        versions()
    except Exception as exc:
        print(f'  environment check failed: {type(exc).__name__}: {exc}')
    print('\n== Chunk fit ==')
    try:
        chunk_fit()
    except Exception as exc:
        print(f'  chunk check failed: {type(exc).__name__}: {exc}')
    print('\n== OCR speed (3-page synthetic scan) ==')
    import ocr
    with tempfile.TemporaryDirectory() as folder:
        SCAN = scanned_pdf(Path(folder))
        time_ocr()
