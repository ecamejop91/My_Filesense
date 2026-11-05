"""Package the exact cached embedding model into FileSense's portable models folder.

Run with no arguments to copy from the local HF cache, or --download to prepare
the pinned revision on a machine without the cache. No PDF/index data is read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from model_runtime import BUNDLED_MODEL_DIR, BUNDLED_MODEL_FILES

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'bundle.json').read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('model_id') != MODEL_ID
            or not set(BUNDLED_MODEL_FILES) <= set(manifest.get('files', {}))):
        raise ValueError('Invalid or incomplete bundled-model manifest')
    for name, expected in manifest['files'].items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()):
            raise ValueError('Invalid file path in bundled-model manifest')
        if not target.is_file() or checksum(target) != expected:
            raise ValueError(f'Bundled model file is missing or changed: {name}')
    return manifest


def bundle_model(destination=BUNDLED_MODEL_DIR, *, download=False):
    destination = Path(destination).resolve()
    if destination.exists():
        return verify_bundle(destination)
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    snapshot = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=not download,
        allow_patterns=[*BUNDLED_MODEL_FILES, 'README.md', 'LICENSE*', 'NOTICE*'],
    ))
    missing = [name for name in BUNDLED_MODEL_FILES if not (snapshot / name).is_file()]
    if missing:
        raise ValueError('Cached model is incomplete. Run python bundle_model.py --download. Missing: '
                         + ', '.join(missing))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bundle-', dir=destination.parent) as scratch:
        staging = Path(scratch) / 'model'
        staging.mkdir()
        names = [*BUNDLED_MODEL_FILES, 'README.md']
        names += [p.name for p in snapshot.iterdir() if p.is_file() and p.name.startswith(('LICENSE', 'NOTICE'))]
        for name in names:
            source = snapshot / name
            if source.is_file():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                # Copy actual bytes, not links into this computer's HF cache.
                shutil.copyfile(source, target)
        model = SentenceTransformer(str(staging), local_files_only=True)
        dimension = model.get_sentence_embedding_dimension()
        manifest = dict(schema_version=1, model_id=MODEL_ID, revision=MODEL_REVISION,
                        embedding_dimension=dimension, files={
                            p.relative_to(staging).as_posix(): checksum(p)
                            for p in sorted(staging.rglob('*')) if p.is_file()})
        (staging / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        verify_bundle(staging)
        staging.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Allow downloading the pinned model if not cached')
    parser.add_argument('--verify', action='store_true', help='Verify the existing bundled files and their hashes')
    args = parser.parse_args()
    try:
        manifest = verify_bundle(BUNDLED_MODEL_DIR) if args.verify else bundle_model(download=args.download)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Could not prepare the model: {exc}\n')
    size = sum((BUNDLED_MODEL_DIR / name).stat().st_size for name in manifest['files'])
    print(f'Bundled model ready: {BUNDLED_MODEL_DIR}')
    print(f'{size / 1024 / 1024:.1f} MiB; {manifest["embedding_dimension"]} dimensions; revision {manifest["revision"]}')


if __name__ == '__main__':
    main()
