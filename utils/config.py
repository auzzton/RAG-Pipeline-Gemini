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

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check which API is available
if not OPENAI_API_KEY and not GEMINI_API_KEY:
    print("⚠️  Warning: No API keys found. Some features may not work.")
    OPENAI_API_KEY = "dummy-key-for-parsing-only"  # Allow parsing to work without API

# Model Configuration
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Use Gemini by default if available, otherwise use OpenAI
USE_GEMINI = bool(GEMINI_API_KEY)

# Retriever
TOP_K = 5
