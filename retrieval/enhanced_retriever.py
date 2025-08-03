import chromadb
import os
import json
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from pathlib import Path

from utils.config import DB_PATH, CHROMA_COLLECTION, TOP_K
from retrieval.embedder import EmbeddingModel

class EnhancedVectorStore:
    """
    Enhanced vector store with persistent storage and better document management.
    """
    
    def __init__(self, embedder: EmbeddingModel):
        """Initializes the enhanced vector store with persistent ChromaDB client."""
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)
    # EnhancedVectorStore class and its logic removed as FAISS is now used.