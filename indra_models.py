"""Portable, pinned Indra weights; shared by setup and the OCR worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import Lock

MODEL_ID = 'datalab-to/indra-ocr'
# Indra 1.1.2's v1.1.0 model set, pinned to immutable content.
MODEL_REVISION = 'bd02167fdf2005a19962cc15188f8496a2c87d3b'
BUNDLED_INDRA_DIR = Path(__file__).resolve().parent / 'models' / 'indra-ocr'
MODEL_FILES = (
    'det.onnx', 'rec.onnx', 'order.onnx', 'orient.onnx', 'script.onnx',
    'rec_cjk.onnx', 'rec_rtl.onnx', 'rec_world.onnx', 'meta.json',
    'lm.npz', 'lm_cjk.npz', 'lm_world.npz',
)
_lock = Lock()


def local_model_dir():
    override = os.environ.get('INDRA_OCR_MODEL_DIR')
    return Path(override).expanduser().resolve() if override else BUNDLED_INDRA_DIR


def check_model_files(folder):
    folder = Path(folder)
    missing = [name for name in MODEL_FILES
               if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
    if missing:
        raise ValueError('Incomplete Indra model folder: ' + ', '.join(missing)
                         + '. Restore the folder, or move it aside and run '
                         'python bundle_indra.py --download.')
    return folder


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(folder):
    folder = check_model_files(folder)
    manifest = json.loads((folder / 'bundle.json').read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('model_id') != MODEL_ID
            or manifest.get('revision') != MODEL_REVISION
            or not set((*MODEL_FILES, 'LICENSE')) <= set(manifest.get('files', {}))):
        raise ValueError('Invalid Indra bundle manifest')
    for name, expected in manifest['files'].items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()):
            raise ValueError('Invalid path in Indra bundle manifest')
        if not target.is_file() or checksum(target) != expected:
            raise ValueError(f'Indra bundle file is missing or changed: {name}')
    return manifest


def bundle_indra(destination=BUNDLED_INDRA_DIR, *, download=False):
    """Copy real files (no cache symlinks), and publish only a complete bundle."""
    destination = Path(destination).resolve()
    with _lock:
        if destination.exists():
            return verify_bundle(destination)
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=not download,
            allow_patterns=[*MODEL_FILES, 'LICENSE', 'README.md'],
        ))
        check_model_files(snapshot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.indra-', dir=destination.parent) as scratch:
            staging = Path(scratch) / 'model'
            staging.mkdir()
            for name in (*MODEL_FILES, 'LICENSE', 'README.md'):
                shutil.copyfile(snapshot / name, staging / name)
            manifest = dict(schema_version=1, model_id=MODEL_ID, revision=MODEL_REVISION,
                            files={p.name: checksum(p) for p in sorted(staging.iterdir())})
            (staging / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            verify_bundle(staging)
            staging.rename(destination)
        return manifest


def ensure_model_dir():
    """Never contact the Hub when a local bundle exists, even when incomplete."""
    folder = local_model_dir()
    if folder.exists() or os.environ.get('INDRA_OCR_MODEL_DIR'):
        return check_model_files(folder)
    from app_storage import PATHS, private_directory
    # Installed/portable application folders may be read-only. First-use
    # downloads belong to this account; distributed weights remain shared.
    folder = private_directory(PATHS.cache) / 'indra-ocr'
    if folder.exists():
        return check_model_files(folder)
    bundle_indra(folder, download=True)
    return folder
