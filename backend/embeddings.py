from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        print(
            "Loading embedding model:",
            model_name
        )

        self.model = SentenceTransformer(
            model_name
        )

        self.dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

    def encode(
        self,
        texts: List[str]
    ) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(
            embeddings,
            dtype="float32"
        )