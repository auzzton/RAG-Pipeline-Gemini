from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os
import datetime
import asyncio
import aiohttp
import tempfile
from pathlib import Path
import uvicorn

# Import your existing modules
from parsing.enhanced_parser import EnhancedParser
from retrieval.embedder import EmbeddingModel
from retrieval.faiss_vector_store import FAISSVectorStore
from generation.enhanced_generator import EnhancedGenerator
from utils.config import DOCS_PATH

app = FastAPI(
    title="Document Processing API",
    description="LLM-powered document processing system for hackathon",
    version="1.0.0"
)
# Debug: Print GEMINI_API_KEY to verify environment loading
print("[DEBUG] GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "your-secure-api-key")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials

# Request/Response Models
class HackathonRequest(BaseModel):
    documents: str = Field(..., description="URL to the document to process")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackathonResponse(BaseModel):
    answers: List[str] = Field(..., description="Answers corresponding to the questions")

# Global variables for pipeline components
embedder = None
vector_store = None
generator = None
parser = None
current_document_url = None

async def download_document(url: str, temp_dir: str) -> str:
    """Download document from URL and save to temp directory"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download document: {response.status}")
            
            # Get filename from URL or use default
            filename = url.split('/')[-1].split('?')[0]
            if not filename.endswith(('.pdf', '.docx')):
                filename = "document.pdf"  # Default to PDF
            
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
            
            return filepath

def setup_pipeline_for_document(document_path: str):
    """Setup pipeline for a specific document"""
    global embedder, vector_store, generator, parser
    
    print("🚀 Initializing Enhanced RAG Pipeline ---")
    embedder = EmbeddingModel()
    # Determine embedding dimension from model
    test_embed = embedder.create_embeddings(["test"])
    dim = test_embed.shape[1] if len(test_embed.shape) > 1 else 384
    vector_store = FAISSVectorStore(dim)
    generator = EnhancedGenerator()
    parser = EnhancedParser()

    # Process the single document with enhanced parser
    print(f"📄 Processing document: {document_path}")
    chunks = parser.parse_document(document_path)
    if chunks:
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        embeddings = embedder.create_embeddings(texts)
        vector_store.add_documents(embeddings, metadatas, texts)
        print(f"✅ Added {len(chunks)} chunks to vector store")
    else:
        print("❌ No chunks generated from document")
    
    print("🎯 Enhanced Pipeline Ready!")
    return embedder, vector_store, generator

def process_single_question(question: str) -> str:
    """Process a single question and return the answer"""
    global embedder, vector_store, generator
    
    if not all([embedder, vector_store, generator]):
        return "Pipeline not initialized properly"
    
    try:
        # Step 1: Retrieve relevant chunks with higher precision using FAISS
        retrieved_chunks = vector_store.retrieve_relevant_chunks(question, top_k=7, embedder=embedder)
        if not retrieved_chunks:
            return "I couldn't find relevant information in the document to answer this question."
        # Step 2: Use direct question answering for better accuracy
        answer = generator.answer_direct_question(question, retrieved_chunks)
        return answer.strip() if answer else "Unable to generate a proper response."
    except Exception as e:
        print(f"Error processing question '{question}': {str(e)}")
        return f"Error processing question: {str(e)}"

@app.post("/hackrx/run")
async def process_document_questions(
    request: HackathonRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Main endpoint for processing documents and answering questions"""
    global current_document_url
    
    start_time = datetime.datetime.now()
    
    try:
        # Support both URLs and local file paths
        document_path = None
        temp_dir = None
        doc_input = request.documents
        
        # If input starts with 'file://', treat as local file path
        if doc_input.startswith('file://'):
            document_path = doc_input.replace('file://', '')
            print(f"Using local file: {document_path}")
        # If input looks like a Windows or Unix path, use directly
        elif doc_input[1:3] == ':/' or doc_input.startswith('/'):
            document_path = doc_input
            print(f"Using local file: {document_path}")
        else:
            # Otherwise, treat as URL and download
            temp_dir = tempfile.mkdtemp()
            print(f"Downloading new document: {doc_input}")
            document_path = await download_document(doc_input, temp_dir)
            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_dir, temp_dir)
        
        # Only reprocess if document changes
        if current_document_url != doc_input:
            setup_pipeline_for_document(document_path)
            current_document_url = doc_input
        
        # Process all questions
        answers = []
        for i, question in enumerate(request.questions):
            print(f"Processing question {i+1}/{len(request.questions)}: {question}")
            answer = process_single_question(question)
            answers.append(answer)
        
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "answers": answers,
            "success": True,
            "processing_time": f"{processing_time:.2f}s",
            "questions_processed": len(request.questions),
            "timestamp": end_time.isoformat()
        }
    except Exception as e:
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"Error in main endpoint: {str(e)}")
        return {
            "answers": [f"Error processing question: {str(e)}" for _ in request.questions],
            "success": False,
            "error": str(e),
            "processing_time": f"{processing_time:.2f}s",
            "questions_processed": 0,
            "timestamp": end_time.isoformat()
        }

def cleanup_temp_dir(temp_dir: str):
    """Clean up temporary directory"""
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temp directory: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document Processing API",
        "version": "1.0.0",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    ) 