"""Find concise, query-relevant quotations inside retrieved text chunks."""
from collections import OrderedDict
import re

import numpy as np

MAX_PASSAGE_CHARS = 400
MAX_CANDIDATES = 64
# Prefer a shorter passage when its cosine similarity is within one point of
# the best candidate. This balances brevity with retaining the relevant context.
SCORE_TOLERANCE = 0.01


def _clean(text):
    return ' '.join(text.split())


def _windows(text):
    words = text.split()
    if len(words) > 1:
        for size in (12, 24, 36):
            starts = list(range(0, max(1, len(words) - size + 1), size // 2))
            starts.append(max(0, len(words) - size))
            for start in dict.fromkeys(starts):
                part = ' '.join(words[start:start + size])
                if len(part) <= MAX_PASSAGE_CHARS:
                    yield part
    else:
        # OCR can return long unspaced runs, including Chinese/Japanese text.
        for size in (96, 192, 320):
            starts = list(range(0, max(1, len(text) - size + 1), size // 2))
            starts.append(max(0, len(text) - size))
            for start in dict.fromkeys(starts):
                yield text[start:start + size]


def passage_candidates(text):
    """Whole sentences/paragraphs and adjacent context; windows for long OCR runs.

    Candidates quote the source with whitespace normalized. They are not generated
    summaries, and sentence windows let a match retain qualifications or context.
    """
    full = _clean(text)
    if not full:
        return ['']
    units = []
    for paragraph in re.split(r'\n\s*\n', text.strip()):
        # Keep wrapped PDF lines together, and do not split decimal numbers.
        sentences = re.split(r'(?<=[.!?])\s+|(?<=[。！？])\s*', paragraph)
        for sentence in sentences:
            cleaned = _clean(sentence)
            if not cleaned:
                continue
            if len(cleaned) <= MAX_PASSAGE_CHARS:
                units.append(cleaned)
            else:
                # Forms and OCR often have lines/clauses without sentence stops.
                parts = re.split(r'\n+|(?<=[;:])\s+', sentence)
                for part in parts:
                    part = _clean(part)
                    if part:
                        units.extend([part] if len(part) <= MAX_PASSAGE_CHARS else _windows(part))
    candidates = []
    for width in (1, 2, 3):
        for start in range(len(units) - width + 1):
            # Adjacent natural units provide context; sliding windows already
            # contain overlapping words and must not be concatenated together.
            part = ' '.join(units[start:start + width])
            if part in full and len(part) <= MAX_PASSAGE_CHARS:
                candidates.append(part)
    if len(full) <= MAX_PASSAGE_CHARS:
        candidates.append(full)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        candidates = list(dict.fromkeys(_windows(full)))
    if len(candidates) > MAX_CANDIDATES:
        # Cover the whole source, including its end, rather than truncating the
        # candidate list at the beginning of a long chunk.
        positions = np.linspace(0, len(candidates) - 1, MAX_CANDIDATES, dtype=int)
        candidates = [candidates[i] for i in positions]
    return candidates or [full[:MAX_PASSAGE_CHARS]]


class PassageSelector:
    """Bounded, memory-only candidate embeddings for one embedding model."""

    def __init__(self, cache_size=128):
        self.cache_size = cache_size
        self._model = None
        self._cache = OrderedDict()

    def select(self, query, results, model):
        """Enrich only displayed rows; preserve retrieval scores and result order."""
        if model is not self._model:
            self._model = model
            self._cache.clear()
        query_key = (query, id(model))
        pending = [row for row in results if row.get('_snippet_query') != query_key]
        if not pending:
            return results
        sources = {}
        needs_scoring = []
        for row in pending:
            source = row.get('_source_text', row['text'])
            row.setdefault('_source_text', source)
            candidates = passage_candidates(source)
            sources[id(row)] = (source, candidates)
            if len(candidates) == 1 and candidates[0] == _clean(source):
                # The chunk already is a single short passage. Reuse its score.
                row.update(text=candidates[0], snippet_score=row['score'], _snippet_query=query_key)
            else:
                needs_scoring.append(row)
        if not needs_scoring:
            return results
        query_vector = next((row['_query_vector'] for row in needs_scoring if '_query_vector' in row), None)
        if query_vector is None:
            query_vector = model.encode([query], normalize_embeddings=True)[0]
        query_vector = np.asarray(query_vector, dtype=np.float32)
        missing = OrderedDict()
        for row in needs_scoring:
            source, candidates = sources[id(row)]
            if source not in self._cache:
                missing.setdefault(source, candidates)
        if missing:
            texts = list(dict.fromkeys(text for parts in missing.values() for text in parts))
            vectors = np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=64), dtype=np.float32)
            if vectors.shape != (len(texts), len(query_vector)):
                raise ValueError('Passage embeddings do not match the search model dimension')
            by_text = dict(zip(texts, vectors))
            for source, candidates in missing.items():
                self._cache[source] = (candidates, np.stack([by_text[text] for text in candidates]))
        for row in needs_scoring:
            source, _ = sources[id(row)]
            candidates, vectors = self._cache[source]
            self._cache.move_to_end(source)
            scores = vectors @ query_vector
            best = float(np.max(scores))
            eligible = [i for i, score in enumerate(scores) if score >= best - SCORE_TOLERANCE]
            chosen = min(eligible, key=lambda i:(len(candidates[i]), -float(scores[i]), i))
            row.update(text=candidates[chosen], snippet_score=float(scores[chosen]), _snippet_query=query_key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return results
