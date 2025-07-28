import chromadb
from typing import List, Dict, Any
import uuid

from utils.config import DB_PATH, CHROMA_COLLECTION, TOP_K
from retrieval.embedder import EmbeddingModel

class VectorStore:
    def __init__(self, embedder: EmbeddingModel):
        """Initializes the vector store with a persistent ChromaDB client."""
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)
        self.embedder = embedder
        print(f"ChromaDB client initialized. Collection '{CHROMA_COLLECTION}' ready.")

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Embeds and adds documents to the collection if it's empty."""
        if not chunks:
            print("No new documents to add.")
            return

        if self.collection.count() > 0:
            print(f"Collection '{CHROMA_COLLECTION}' already contains documents. Skipping addition.")
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        embeddings = self.embedder.create_embeddings(texts)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        print(f"✅ Added {len(chunks)} documents to ChromaDB.")

    def retrieve_relevant_chunks(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieves the top-k most relevant chunks for a given query."""
        query_embedding = self.embedder.create_embeddings([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )

        retrieved_chunks = []
        if results and results.get('documents'):
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                retrieved_chunks.append({
                    "chunk": doc,
                    "source": metadata.get('source', 'unknown'),
                    "confidence": round(1 - distance, 4)  # distance to similarity
                })

        return retrieved_chunks
