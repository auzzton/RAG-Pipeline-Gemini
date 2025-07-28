from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

from utils.config import EMBEDDING_MODEL

class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initializes the embedding model."""
        self.model = SentenceTransformer(model_name)
        print(f"✅ Embedding model '{model_name}' loaded.")

    def create_embeddings(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Creates normalized embeddings for a list of texts."""
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=normalize  # native option if model supports it
        )

        # Fallback normalization if model doesn't support `normalize_embeddings`
        if normalize and not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings
