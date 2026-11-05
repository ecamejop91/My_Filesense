"""Adapters share one frozen corpus and document-level scoring contract."""
from .corpus import deduplicate_hits, rank_scores

TFIDF_SETTINGS = dict(lowercase=True, strip_accents=None, analyzer='word',
                      token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1, 1),
                      stop_words=None, min_df=1, max_df=1.0, max_features=None,
                      binary=False, norm='l2', use_idf=True, smooth_idf=True,
                      sublinear_tf=False)


class FileSenseSearch:
    def __init__(self, corpus, model=None, search_fn=None):
        import document_search_engine as engine
        self.corpus = corpus
        self.search_fn = search_fn or engine.search_index
        self.model = model if model is not None else engine.get_embedding_model(corpus.model_name)
        if corpus.vectors.shape[1] != self.model.get_sentence_embedding_dimension():
            raise ValueError("Model and stored embeddings have different dimensions")
        self.docs_by_id = {doc['doc_id']: doc for doc in corpus.documents.values()}

    def search(self, query):
        # Fetch ALL chunks before deduplication. Fetching k chunks can miss a
        # unique PDF when one document occupies several high-ranking positions.
        hits = self.search_fn(query, self.model, self.corpus.chunks,
                              self.corpus.vectors, self.docs_by_id,
                              top_k=len(self.corpus.chunks))
        ranked = deduplicate_hits(hits, self.corpus)
        if {r['document_id'] for r in ranked} != set(self.corpus.searchable_ids):
            raise ValueError("FileSense search did not return every searchable document")
        return ranked


class TfidfSearch:
    def __init__(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.ids = sorted(corpus.documents)
        self.searchable = set(corpus.searchable_ids)
        self.vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)
        self.matrix = None
        try:
            # Fit on documents only, never queries or relevance judgments.
            # Include empty documents in N for idf; don't retrieve them in either backend.
            self.matrix = self.vectorizer.fit_transform([corpus.documents[d]['text'] for d in self.ids])
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc):
                raise

    def search(self, query):
        if self.matrix is None:
            scores = [0.0] * len(self.ids)
        else:
            vector = self.vectorizer.transform([query])
            scores = (self.matrix @ vector.T).toarray().ravel()
        # Zero scores, including OOV queries, retain the shared deterministic tie order.
        return rank_scores({doc: score for doc, score in zip(self.ids, scores) if doc in self.searchable})
