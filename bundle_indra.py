"""Prepare Indra once for offline Windows, Intel Mac and Apple Silicon use."""
import argparse

from indra_models import BUNDLED_INDRA_DIR, bundle_indra, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Allow the initial model download')
    parser.add_argument('--verify', action='store_true', help='Check local model hashes without network access')
    args = parser.parse_args()
    try:
        manifest = verify_bundle(BUNDLED_INDRA_DIR) if args.verify else bundle_indra(download=args.download)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Could not prepare Indra: {exc}\n')
    size = sum((BUNDLED_INDRA_DIR / name).stat().st_size for name in manifest['files'])
    print(f'Indra ready: {BUNDLED_INDRA_DIR}\n{size / 1024 / 1024:.1f} MiB; revision {manifest["revision"]}')


if __name__ == '__main__':
    main()
