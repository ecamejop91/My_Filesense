"""Process-local model and category-vector reuse, shared by analysis and search."""
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from weakref import WeakKeyDictionary

_lock = RLock()
_models = OrderedDict()
_categories = WeakKeyDictionary()
BUNDLED_MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'all-MiniLM-L6-v2'
BUNDLED_MODEL_FILES = (
    'config.json', 'modules.json', 'config_sentence_transformers.json',
    'sentence_bert_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'vocab.txt', '1_Pooling/config.json', 'model.safetensors',
)


def canonical_model_name(name):
    return name.removeprefix('sentence-transformers/') if name in (
        'all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L6-v2') else name


def get_model(name, factory):
    bundled = canonical_model_name(name) == 'all-MiniLM-L6-v2' and BUNDLED_MODEL_DIR.exists()
    source = str(BUNDLED_MODEL_DIR) if bundled else name
    if bundled:
        missing = [filename for filename in BUNDLED_MODEL_FILES if not (BUNDLED_MODEL_DIR / filename).is_file()]
        if missing:
            raise RuntimeError('The bundled embedding model is incomplete. Restore models/all-MiniLM-L6-v2 '
                               'from a complete distribution, or move that folder aside and run bundle_model.py. Missing: '
                               + ', '.join(missing))
    key = (canonical_model_name(name), str(BUNDLED_MODEL_DIR) if bundled else None, factory)
    with _lock:
        if key not in _models:
            try:
                # A complete local cache needs no Hub metadata/network checks.
                _models[key] = factory(source, local_files_only=True)
            except OSError:
                if bundled:
                    # An explicit packaged model must never silently switch weights
                    # or fall back to a download if its contents cannot be loaded.
                    raise
                # Preserve first-use setup when the model has never been cached.
                _models[key] = factory(name)
            while len(_models) > 2:
                _models.popitem(last=False)
        _models.move_to_end(key)
        return _models[key]


def category_vectors(model, definitions):
    signature = tuple(definitions.items())
    with _lock:
        cached = _categories.get(model)
        if cached is None or cached[0] != signature:
            cached = (signature, model.encode(list(definitions.values()), normalize_embeddings=True))
            _categories[model] = cached
        return cached[1]
