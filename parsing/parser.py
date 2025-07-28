import os
import fitz  # PyMuPDF
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_PATH

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def chunk_text(text: str, source: str) -> List[Dict[str, any]]:
    """Chunks text and adds metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    
    chunk_list = []
    for i, chunk in enumerate(chunks):
        chunk_list.append({
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_number": i + 1
            }
        })
    return chunk_list

def parse_document(file_path: str) -> List[Dict[str, any]]:
    """Parses a single document (PDF or DOCX), extracts text, and chunks it."""
    filename = os.path.basename(file_path)
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    chunks = chunk_text(text, filename)
    print(f"Parsed and chunked {filename}. Total chunks: {len(chunks)}")
    return chunks

def load_and_parse_documents(docs_path: str = DOCS_PATH) -> List[Dict[str, any]]:
    """Loads all documents from a directory, parses, and chunks them."""
    all_chunks = []
    print(f"Scanning for documents in: {docs_path}")
    for filename in os.listdir(docs_path):
        if filename.startswith('~'): # Ignore temporary files
            continue
        file_path = os.path.join(docs_path, filename)
        if os.path.isfile(file_path):
            try:
                chunks = parse_document(file_path)
                all_chunks.extend(chunks)
            except (ValueError, docx.opc.exceptions.PackageNotFoundError) as e:
                print(f"Could not parse {filename}, skipping. Error: {e}")
    print(f"Total chunks generated from all documents: {len(all_chunks)}")
    return all_chunks
