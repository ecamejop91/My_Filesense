"""Process-cached Indra CPU OCR for PDFs without extractable text."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pymupdf as fitz

from indra_models import MODEL_REVISION, ensure_model_dir

# Rasterization for OCR. 150 DPI keeps printed text legible while rendering and
# detecting ~44% fewer pixels than 200 DPI. The longest side is still capped so
# large-format pages stay bounded.
OCR_DPI = 150
OCR_MAX_SIDE = 3000

DEFAULT_OCR_MODEL = 'indra'
OCR_MODEL_NAME = 'Indra OCR (CPU)'


class OCRError(RuntimeError):
    """An actionable OCR setup or inference failure."""


def validate_ocr_model(model):
    if model != DEFAULT_OCR_MODEL:
        raise ValueError(f'Unknown OCR model: {model!r}')
    return model


def ocr_signature(model):
    validate_ocr_model(model)
    revision = f'1.1.2:{MODEL_REVISION}'
    return f'{model}:{revision}:dpi={OCR_DPI}:max_side={OCR_MAX_SIDE}'


class _IndraOCREngine:
    name = OCR_MODEL_NAME

    def __init__(self, converter):
        self._converter = converter

    def predict(self, image):
        # The PDF path already supplies grayscale; retain BGR support for callers
        # that provide color arrays directly.
        if image.ndim == 2:
            gray = image
        else:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        page = self._converter.ocr(gray)
        if page.source == 'error':
            raise OCRError(page.note)
        return [{'rec_texts': page.text.splitlines()}]


@lru_cache(maxsize=1)
def _get_engine(model=DEFAULT_OCR_MODEL):
    validate_ocr_model(model)
    try:
        import indra_ocr
        converter = indra_ocr.load(
            path=str(ensure_model_dir()), threads=min(4, os.cpu_count() or 1),
            mode='ocr', providers=['CPUExecutionProvider'], furniture=True,
        )
        return _IndraOCREngine(converter)
    except Exception as exc:
        raise OCRError(
            f'{OCR_MODEL_NAME} could not start. In the FileSense Python 3.12 environment, '
            'run python -m pip install -r requirements.txt. '
            'For offline Indra use, prepare models/indra-ocr with '
            'python bundle_indra.py --download and include it when copying FileSense. '
            f'Details: {exc}'
        ) from exc


def ocr_backend_name(model=DEFAULT_OCR_MODEL):
    """Human-readable name of the Indra backend (loads it)."""
    return _get_engine(model).name


def _render_page(page):
    scale = min(OCR_DPI / 72, OCR_MAX_SIDE / max(page.rect.width, page.rect.height))
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    import cv2
    # MuPDF's direct grayscale conversion uses a different color transform;
    # retain the existing OpenCV conversion for Indra's input pixels.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def extract_page_texts_with_ocr(pdf_path: Path, progress=None, *, ocr_model=DEFAULT_OCR_MODEL):
    """Render one page at a time; return (one-based page number, recognized text)."""
    validate_ocr_model(ocr_model)
    page_texts = []
    with fitz.open(str(pdf_path)) as doc:
        if doc.needs_pass:
            return []
        if progress:
            progress(dict(stage='ocr', path=str(pdf_path),
                          detail=f'Loading {OCR_MODEL_NAME} for scanned pages'))
        engine = _get_engine(ocr_model)
        for page_number, page in enumerate(doc, start=1):
            if progress:
                progress(dict(stage='ocr', path=str(pdf_path), current=page_number - 1, total=len(doc),
                              detail=f'Recognizing scanned page {page_number} of {len(doc)}'))
            try:
                results = engine.predict(_render_page(page))
                lines = []
                for result in results:
                    lines.extend(text.strip() for text in result['rec_texts'] if text.strip())
                text = '\n'.join(lines)
                if text:
                    page_texts.append((page_number, text))
            except Exception as exc:
                raise OCRError(f'OCR failed for {Path(pdf_path).name}, page {page_number}: {exc}') from exc
        if progress:
            progress(dict(stage='ocr', path=str(pdf_path), current=len(doc), total=len(doc),
                          detail=f'Recognized {len(doc)} pages'))
    return page_texts
