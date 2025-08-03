# RAG System for Document Querying

This project is a modular Python-based Retrieval-Augmented Generation (RAG) system. It's designed to answer natural language queries about unstructured documents by leveraging Large Language Models (LLMs) and semantic vector search.

## Usage Options

### 1. Command Line Interface (main.py)

For development and testing with local documents:

```bash
python main.py
```

### 2. FastAPI Web Service (api.py)

For production use and hackathon demos with URL-based documents:

```bash
python api.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints:

- `POST /hackrx/run` - Process documents and answer questions
- `GET /health` - Health check
- `GET /` - API information

#### Example API Usage:

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic?", "What are the key points?"]
  }'
```

## 🚀 Enhanced Chunking System

### Features:

- **Persistent Storage**: Chunks are cached and reused, no reprocessing needed
- **Optimal Chunking Strategies**: Different strategies for different document types
- **Intelligent Change Detection**: Only reprocesses when documents change
- **Performance Monitoring**: Track processing times and improvements

### Chunking Strategies:

- **Legal Documents**: 800 chars, 150 overlap (precise clause matching)
- **Medical Documents**: 600 chars, 100 overlap (specific procedures)
- **Technical Documents**: 1200 chars, 250 overlap (comprehensive context)
- **Financial Documents**: 900 chars, 180 overlap (policy details)
- **Default**: 1000 chars, 200 overlap (balanced approach)

### Demo:

```bash
python demo_enhanced_chunking.py
```

### Commands:

- `stats` - Show collection statistics
- `cache` - Show cache information
- `reprocess` - Force reprocess all documents
- `api` - Show API status

## 🌟 Gemini API Integration

### Features:

- **Dual API Support**: Works with both OpenAI and Gemini APIs
- **Automatic Fallback**: Uses available API automatically
- **JSON Response Parsing**: Handles both OpenAI and Gemini response formats
- **Error Handling**: Graceful fallback when APIs are unavailable

### Setup:

1. Add your Gemini API key to `.env`:

   ```
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

2. Test the integration:

   ```bash
   python test_gemini_integration.py
   ```

3. Run the main system:
   ```bash
   python main.py
   ```
