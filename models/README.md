FileSense's embedding model can live here, independent of the Hugging Face cache.

From the project root, run:

```powershell
.\.venv\Scripts\python.exe bundle_model.py
```

This copies the pinned `sentence-transformers/all-MiniLM-L6-v2` revision from the
local cache, preserves the upstream model card and any cached license/notice files,
verifies that it loads locally, and records file hashes in `bundle.json`.
Use `--download` on a machine without the cached model, or `--verify` to check an
existing bundle. Existing bundles are verified rather than overwritten.

Include the **whole `models/all-MiniLM-L6-v2/` folder** when distributing FileSense
as a folder or archive. The runtime discovers it relative to the program files,
loads it with `local_files_only=True`, and shares it across indexing, tagging,
search, and evaluation. SQLite continues recording the original logical model
name so moving the program does not change its model identity.

The weights are excluded from Git because the model is about 87 MiB. A plain Git
clone needs the preparation command above or the separately distributed model
folder. If the folder is absent, the existing HF cache/first-download workflow
still works. If it is present but incomplete, FileSense reports an error rather
than downloading different weights silently.

This bundles the **embedding model**, separately from the Python environment,
tiktoken assets and OCR weights. Model weights load into memory once per session.

## Indra OCR bundle

Prepare the scanned-PDF model once from the project root:

```sh
python bundle_indra.py --download
python bundle_indra.py --verify
```

The first command stores about 73 MiB in **`models/indra-ocr/`**, including all
default multilingual ONNX networks, language models, the upstream model card and
weight license. The version is pinned to Indra 1.1.2's v1.1.0 model set at commit
`bd02167fdf2005a19962cc15188f8496a2c87d3b`. Optional math models are not included.
The manifest records SHA-256 hashes; `--verify` checks them without network access.
With no flags, the preparation command only uses an existing Hugging Face cache.

Include the **entire folder** when copying or packaging FileSense. It contains
ordinary files, with no symlinks into this computer's cache. The same weights
serve Windows x64, Intel Macs and Apple Silicon Macs; install each platform's
native dependencies separately. Both model folders are excluded from Git, so a
plain clone needs preparation or the separately distributed model folders.

The **OCR model** selector near the top of the app defaults to Indra and remembers
your choice. Inference loads the local folder directly, without Hub checks. A
missing bundle is downloaded once on first OCR in the analysis worker. A present
but incomplete bundle raises an error instead of silently downloading weights.
`INDRA_OCR_MODEL_DIR` can point to another complete local model folder; when set,
it must already contain the required files and never triggers a fallback download.

Indra's code is Apache 2.0; its weights have a modified OpenRAIL-M license with
commercial-use conditions. Review the included `LICENSE` before redistribution
or commercial deployment: [upstream model license](https://huggingface.co/datalab-to/indra-ocr/blob/v1.1.0/LICENSE).
