import os
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(PROJECT_ROOT, "data", "docs")
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB
CHROMA_COLLECTION = "policy_documents"

# LangChain Text Splitter
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY is not set. Please add it to your .env file.")

# ✅ Use GPT-4o by default (supports JSON mode)
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")

# Retriever
TOP_K = 5
