import time
import re
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


class RAGEngine:
    """
    Lightweight hybrid retrieval engine.

    Uses:
    - BM25 for keyword relevance
    - TF-IDF cosine similarity for semantic-ish lexical relevance
    - score fusion for more robust retrieval

    No large embedding model is required.
    """

    def __init__(self):
        self.documents: List[Dict] = []

        self.bm25 = None
        self.vectorizer = None
        self.tfidf_matrix = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Normalize text into tokens for BM25."""
        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

    def add_document(
        self,
        text: str,
        metadata: Dict = None
    ):
        """
        Add a document/chunk to the retrieval collection.
        """

        metadata = metadata or {}

        self.documents.append({
            "text": text.strip(),
            "metadata": metadata
        })

    def build_index(self):
        """
        Build both BM25 and TF-IDF indexes.
        """

        if not self.documents:
            print("No documents to index.")
            return

        texts = [
            doc["text"]
            for doc in self.documents
        ]

        # -------------------------
        # BM25
        # -------------------------

        tokenized_documents = [
            self._tokenize(text)
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        # -------------------------
        # TF-IDF
        # -------------------------

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                texts
            )
        )

        print(
            "Indexed",
            len(self.documents),
            "chunks"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[Dict], float]:

        if not self.documents:
            return [], 0.0

        if self.bm25 is None:
            self.build_index()

        start = time.perf_counter()

        # -------------------------
        # BM25 scores
        # -------------------------

        query_tokens = self._tokenize(
            query
        )

        bm25_scores = np.asarray(
            self.bm25.get_scores(
                query_tokens
            ),
            dtype=float
        )

        # -------------------------
        # TF-IDF scores
        # -------------------------

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        tfidf_scores = (
            self.tfidf_matrix
            @ query_vector.T
        ).toarray().ravel()

        # -------------------------
        # Normalize scores
        # -------------------------

        def normalize(scores):
            minimum = scores.min()
            maximum = scores.max()

            if maximum - minimum < 1e-9:
                return np.zeros_like(
                    scores
                )

            return (
                scores - minimum
            ) / (
                maximum - minimum
            )

        bm25_norm = normalize(
            bm25_scores
        )

        tfidf_norm = normalize(
            tfidf_scores
        )

        # -------------------------
        # Hybrid fusion
        # -------------------------

        hybrid_scores = (
            0.65 * bm25_norm
            +
            0.35 * tfidf_norm
        )

        top_indices = np.argsort(
            hybrid_scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            if hybrid_scores[index] <= 0:
                continue

            result = dict(
                self.documents[index]
            )

            result["score"] = float(
                hybrid_scores[index]
            )

            result["bm25_score"] = float(
                bm25_norm[index]
            )

            result["tfidf_score"] = float(
                tfidf_norm[index]
            )

            results.append(
                result
            )

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        return results, latency