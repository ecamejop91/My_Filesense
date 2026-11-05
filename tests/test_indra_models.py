"""Offline lookup, portable packaging, integrity and grayscale OCR adaptation."""
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pymupdf as fitz

import indra_models as models
import ocr


class IndraBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'bundle'
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('INDRA_OCR_MODEL_DIR', None)

    def snapshot(self):
        folder = self.root / 'snapshot'
        folder.mkdir()
        for name in (*models.MODEL_FILES, 'LICENSE', 'README.md'):
            (folder / name).write_bytes(b'fixture bytes')
        return folder

    def test_bundle_is_portable_idempotent_and_never_downloads_when_local(self):
        with patch('huggingface_hub.snapshot_download', return_value=self.snapshot()) as download:
            manifest = models.bundle_indra(self.folder, download=True)
            self.assertEqual(models.bundle_indra(self.folder), manifest)
            self.assertEqual(models.verify_bundle(self.folder), manifest)
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs['revision'], models.MODEL_REVISION)
        self.assertFalse(any(p.is_symlink() for p in self.folder.iterdir()))
        with patch.object(models, 'BUNDLED_INDRA_DIR', self.folder), \
                patch('huggingface_hub.snapshot_download', side_effect=AssertionError('network')):
            self.assertEqual(models.ensure_model_dir(), self.folder)

    def test_explicit_incomplete_folder_never_falls_back_to_download(self):
        os.environ['INDRA_OCR_MODEL_DIR'] = str(self.folder)
        with patch('huggingface_hub.snapshot_download') as download:
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                models.ensure_model_dir()
        download.assert_not_called()

    def test_changed_weights_fail_verification(self):
        with patch('huggingface_hub.snapshot_download', return_value=self.snapshot()):
            models.bundle_indra(self.folder)
        (self.folder / 'rec.onnx').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed: rec.onnx'):
            models.verify_bundle(self.folder)

    def test_failed_copy_does_not_publish_partial_bundle(self):
        with patch('huggingface_hub.snapshot_download', return_value=self.snapshot()), \
                patch.object(models.shutil, 'copyfile', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                models.bundle_indra(self.folder)
        self.assertFalse(self.folder.exists())

    def test_indra_adapter_converts_bgr_to_gray_and_keeps_text(self):
        converter = Mock()
        converter.ocr.return_value = Mock(source='ocr', text='Invoice\n4729')
        image = np.zeros((12, 20, 3), np.uint8)
        image[:, :, 2] = 255
        result = ocr._IndraOCREngine(converter).predict(image)
        self.assertEqual(result, [{'rec_texts': ['Invoice', '4729']}])
        gray = converter.ocr.call_args.args[0]
        self.assertEqual(gray.shape, (12, 20))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(int(gray[0, 0]), 76)
        converter.ocr.reset_mock()
        self.assertEqual(ocr._IndraOCREngine(converter).predict(gray), result)
        self.assertIs(converter.ocr.call_args.args[0], gray)
        converter.ocr.return_value = Mock(source='error', note='bad page')
        with self.assertRaisesRegex(ocr.OCRError, 'bad page'):
            ocr._IndraOCREngine(converter).predict(image)

    def test_grayscale_fast_path_keeps_identical_pixels_on_colored_pdf(self):
        import cv2
        with fitz.open() as doc:
            page = doc.new_page(width=320, height=200)
            for i, color in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1), (.2, .5, .7))):
                page.draw_rect(fitz.Rect(i * 80, 0, (i + 1) * 80, 200), color=color, fill=color)
            page.insert_text((10, 40), 'Invoice 4729', color=(.8, .1, .3))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(ocr.OCR_DPI / 72, ocr.OCR_DPI / 72),
                                     colorspace=fitz.csRGB, alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
            old = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, ::-1]), cv2.COLOR_BGR2GRAY)
            new = ocr._render_page(page)
        self.assertEqual(new.ndim, 2)
        self.assertTrue(new.flags.c_contiguous)
        np.testing.assert_array_equal(new, old)

    def test_indra_pdf_path_delivers_grayscale_directly(self):
        pdf = self.root / 'scan.pdf'
        with fitz.open() as doc:
            doc.new_page(width=612, height=792)
            doc.save(pdf)
        predictor = Mock()
        predictor.predict.return_value = [{'rec_texts': ['invoice']}]
        with patch.object(ocr, '_get_engine', return_value=predictor):
            self.assertEqual(ocr.extract_page_texts_with_ocr(pdf, ocr_model='indra'), [(1, 'invoice')])
        raster = predictor.predict.call_args.args[0]
        self.assertEqual(raster.shape, (1650, 1275))
        self.assertEqual(raster.dtype, np.uint8)

    def test_indra_loads_local_weights_once_with_cpu_provider(self):
        ocr._get_engine.cache_clear()
        self.addCleanup(ocr._get_engine.cache_clear)
        package = Mock()
        with patch.dict(sys.modules, {'indra_ocr': package}), \
                patch.object(ocr, 'ensure_model_dir', return_value=self.folder):
            first = ocr._get_engine('indra')
            self.assertIs(ocr._get_engine('indra'), first)
        package.load.assert_called_once()
        options = package.load.call_args.kwargs
        self.assertEqual(options['path'], str(self.folder))
        self.assertEqual(options['providers'], ['CPUExecutionProvider'])
        self.assertEqual(options['mode'], 'ocr')
