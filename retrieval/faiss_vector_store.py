import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Any, Optional

class FAISSVectorStore:
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, embedder=None) -> list:
        """Retrieves the top-k most relevant chunks for a given query."""
        if embedder is None:
            raise ValueError("Embedder must be provided for query embedding.")
        query_embedding = embedder.create_embeddings([query])
        results = self.search(query_embedding, top_k)
        retrieved_chunks = []
        for result in results:
            metadata = result.get("metadata", {})
            retrieved_chunks.append({
                "chunk": result.get("chunk", ""),
                "source": metadata.get("source", "unknown"),
                "confidence": round(1 - result.get("distance", 0), 4)
            })
        return retrieved_chunks
    def __init__(self, dim: int, index_path: str = "faiss_index.bin", meta_path: str = "faiss_meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self.texts = []
        self.ids = []
        self._load_index()

    def add_documents(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]], texts: List[str], ids: Optional[List[str]] = None):
        if ids is None:
            ids = [str(i + len(self.ids)) for i in range(len(texts))]
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadatas)
        self.texts.extend(texts)
        self.ids.extend(ids)
        self._save_index()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        D, I = self.index.search(query_embedding.astype(np.float32), top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.texts):
                results.append({
                    "chunk": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "id": self.ids[idx],
                    "distance": float(dist)
                })
        return results

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"metadata": self.metadata, "texts": self.texts, "ids": self.ids}, f)

    def _load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", [])
                self.texts = data.get("texts", [])
                self.ids = data.get("ids", [])

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []
        self.texts = []
        self.ids = []
        self._save_index()
