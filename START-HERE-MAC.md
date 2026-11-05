# FileSense on your Mac

Use the FileSense Git working copy for development. The virtual environment and
local model caches stay on your computer; dependencies are listed in `requirements.txt`.

## Run the app

Double-click **Run FileSense.command** in Finder, or open a terminal in this folder and run:

```sh
./"Run FileSense.command"
```

The desktop window is called **FileSense**. Click **Load Documents**, select PDF,
Word (`.docx`), PowerPoint (`.pptx`), Excel (`.xlsx`/`.xls`), or RTF (`.rtf`) files,
then click **Analyze Files**. Enter a question, then press **Enter** or click the
search icon inside the search bar. An image-only
PDF automatically uses OCR if normal extraction produces no chunks.
Indra OCR runs automatically during **Analyze Files**, using
the bundled `models/indra-ocr/` folder offline; include that folder when moving
the app. If it is absent, the first OCR run downloads it once.

## Edit in VS Code

Open Visual Studio Code and install Microsoft's Python extension if needed. Use **File > Open Folder** to open this FileSense folder. Select the interpreter at `.venv/bin/python` using **Python: Select Interpreter** in the Command Palette. Open **Terminal > New Terminal** and run the launch command above.

## Fresh installation

Install Python 3.12, then open a terminal in the repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements use ONNX Runtime for OCR on both Intel and Apple Silicon Macs,
including M1/M2/M3/M4. Use native Python for your CPU (`x86_64` on Intel, `arm64`
on Apple Silicon). The maintained wheel setup targets **macOS 13 or newer**;
older MacBooks unable to run that OS are not supported by this dependency set.
Intel Macs receive compatible NumPy, PyTorch and Transformers versions for the
existing semantic search model. OCR itself does not use PyTorch.

Prepare the OCR model once if it was not included:

```sh
python bundle_indra.py --download
python bundle_indra.py --verify
```

Keep `.venv/` and `.cache/`
in `.gitignore`; commit the requirement list and source files instead.
Never copy a virtual environment between computers.

## Local setup

- Python 3.12 virtual environment: `.venv`
- Python packages: installed from `requirements.txt`
- Portable OCR weights: `models/indra-ocr/` (keep this folder when copying the app)
- Local Mac compatibility change: PDF and folder links use Qt's cross-platform opener.
- Private libraries, preferences, reports, temporary files, and caches stay under
  `~/Library/Application Support/FileSense/` for the current Mac account.
- Other ordinary Mac accounts start with their own empty FileSense library.
- This applies to installed and portable copies. Moving the app does not move private data.
- Import an older library with **Advanced → Import Existing Library…**. Select its
  `file_tags.sqlite3` and confirm. Your private library must be empty. The old copy
  remains unchanged and retains its original permissions.
- FileSense verifies owner-only data permissions and stops if it cannot protect them.
  Source documents and exports in shared locations remain accessible there.
- See the README section **Privacy and separate computer accounts** for boundaries.

Keep this folder in place because the virtual environment contains absolute paths. If you move it, recreate `.venv` in the new location.

## Qt startup compatibility

Some macOS copies mark Qt's display plugins as hidden, which causes Python to
abort before the window opens. FileSense detects this at startup and copies the
installed plugin contents into the private `cache/qt-plugins` folder, without copying the hidden
flags. Qt uses this local cache automatically, including when running the Python
entry point from VS Code. Plugin updates are reflected in the cache on startup.
