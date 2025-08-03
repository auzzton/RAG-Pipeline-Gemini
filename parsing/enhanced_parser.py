import os
import json
import hashlib
import fitz  # PyMuPDF
import docx
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import DOCS_PATH

class EnhancedParser:
    """
    Enhanced document parser with persistent chunk storage and optimal chunking strategies.
    """
    
    def __init__(self, cache_dir: str = "cache/chunks"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimal chunking strategies for different document types
        self.chunking_strategies = {
            "default": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]
            },
            "legal": {
                "chunk_size": 800,
                "chunk_overlap": 150,
                "separators": ["\n\n", "\n", ". ", " ", ""]
            },
            "medical": {
                "chunk_size": 600,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n", ". ", " ", ""]
            },
            "technical": {
                "chunk_size": 1200,
                "chunk_overlap": 250,
                "separators": ["\n\n", "\n", ". ", " ", ""]
            },
            "financial": {
                "chunk_size": 900,
                "chunk_overlap": 180,
                "separators": ["\n\n", "\n", ". ", " ", ""]
            }
        }
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 hash of file for cache identification."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_cache_path(self, file_path: str) -> Path:
        """Get cache file path for a document."""
        file_hash = self._get_file_hash(file_path)
        filename = Path(file_path).stem
        return self.cache_dir / f"{filename}_{file_hash}.pkl"
    
    def _load_cached_chunks(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Load cached chunks if they exist and file hasn't changed."""
        cache_path = self._get_cache_path(file_path)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify file hasn't changed
                if cached_data.get('file_hash') == self._get_file_hash(file_path):
                    print(f"📁 Loading cached chunks for {Path(file_path).name}")
                    return cached_data.get('chunks', [])
                else:
                    print(f"🔄 File changed, regenerating chunks for {Path(file_path).name}")
            except Exception as e:
                print(f"⚠️  Error loading cache for {Path(file_path).name}: {e}")
        
        return None
    
    def _save_cached_chunks(self, file_path: str, chunks: List[Dict[str, Any]]):
        """Save chunks to cache for future use."""
        cache_path = self._get_cache_path(file_path)
        
        cache_data = {
            'file_hash': self._get_file_hash(file_path),
            'chunks': chunks,
            'created_at': datetime.now().isoformat(),
            'file_path': file_path
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Cached chunks for {Path(file_path).name}")
        except Exception as e:
            print(f"⚠️  Error saving cache for {Path(file_path).name}: {e}")
    
    def _detect_document_type(self, text: str, filename: str) -> str:
        """Detect document type based on content and filename for optimal chunking."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Legal documents
        legal_keywords = ['policy', 'terms', 'conditions', 'agreement', 'contract', 'clause', 'liability']
        if any(keyword in text_lower or keyword in filename_lower for keyword in legal_keywords):
            return "legal"
        
        # Medical documents
        medical_keywords = ['medical', 'health', 'treatment', 'diagnosis', 'surgery', 'patient', 'clinical']
        if any(keyword in text_lower or keyword in filename_lower for keyword in medical_keywords):
            return "medical"
        
        # Technical documents
        technical_keywords = ['technical', 'specification', 'manual', 'guide', 'procedure', 'protocol']
        if any(keyword in text_lower or keyword in filename_lower for keyword in technical_keywords):
            return "technical"
        
        # Financial documents
        financial_keywords = ['financial', 'cost', 'price', 'payment', 'claim', 'coverage', 'premium']
        if any(keyword in text_lower or keyword in filename_lower for keyword in financial_keywords):
            return "financial"
        
        return "default"
    
    def _extract_text_with_structure(self, file_path: str) -> Dict[str, Any]:
        """Extract text with structural information (page numbers, sections, etc.)."""
        filename = os.path.basename(file_path)
        file_type = Path(file_path).suffix.lower()
        
        if file_type == '.pdf':
            return self._extract_pdf_with_structure(file_path)
        elif file_type == '.docx':
            return self._extract_docx_with_structure(file_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    def _extract_pdf_with_structure(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF text with page and structural information."""
        doc = fitz.open(file_path)
        pages = []
        full_text = ""
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            pages.append({
                'page_number': page_num + 1,
                'text': page_text,
                'length': len(page_text)
            })
            full_text += page_text + "\n\n"
        
        return {
            'text': full_text.strip(),
            'pages': pages,
            'total_pages': len(pages),
            'file_type': 'pdf'
        }
    
    def _extract_docx_with_structure(self, file_path: str) -> Dict[str, Any]:
        """Extract DOCX text with paragraph and structural information."""
        doc = docx.Document(file_path)
        paragraphs = []
        full_text = ""
        
        for para_num, para in enumerate(doc.paragraphs):
            para_text = para.text.strip()
            if para_text:
                paragraphs.append({
                    'paragraph_number': para_num + 1,
                    'text': para_text,
                    'style': para.style.name if para.style else 'Normal'
                })
                full_text += para_text + "\n\n"
        
        return {
            'text': full_text.strip(),
            'paragraphs': paragraphs,
            'total_paragraphs': len(paragraphs),
            'file_type': 'docx'
        }
    
    def _chunk_text_optimally(self, text: str, source: str, doc_type: str = "default") -> List[Dict[str, Any]]:
        """Chunk text using optimal strategy for the document type."""
        strategy = self.chunking_strategies.get(doc_type, self.chunking_strategies["default"])
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy["chunk_size"],
            chunk_overlap=strategy["chunk_overlap"],
            separators=strategy["separators"],
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        
        chunk_list = []
        for i, chunk in enumerate(chunks):
            chunk_list.append({
                "text": chunk,
                "metadata": {
                    "source": source,
                    "chunk_number": i + 1,
                    "chunk_type": doc_type,
                    "chunk_size": len(chunk),
                    "created_at": datetime.now().isoformat()
                }
            })
        
        return chunk_list
    
    def parse_document(self, file_path: str, force_reprocess: bool = False) -> List[Dict[str, Any]]:
        """Parse a single document with caching and optimal chunking."""
        filename = os.path.basename(file_path)
        
        # Check cache first (unless force reprocess)
        if not force_reprocess:
            cached_chunks = self._load_cached_chunks(file_path)
            if cached_chunks:
                return cached_chunks
        
        print(f"🔄 Processing document: {filename}")
        
        try:
            # Extract text with structure
            doc_data = self._extract_text_with_structure(file_path)
            
            # Detect document type for optimal chunking
            doc_type = self._detect_document_type(doc_data['text'], filename)
            print(f"📋 Detected document type: {doc_type}")
            
            # Chunk text optimally
            chunks = self._chunk_text_optimally(doc_data['text'], filename, doc_type)
            
            # Add structural metadata
            for chunk in chunks:
                chunk['metadata'].update({
                    'file_type': doc_data['file_type'],
                    'total_pages': doc_data.get('total_pages'),
                    'total_paragraphs': doc_data.get('total_paragraphs'),
                    'document_type': doc_type
                })
            
            # Cache the chunks
            self._save_cached_chunks(file_path, chunks)
            
            print(f"✅ Parsed and chunked {filename}. Total chunks: {len(chunks)}")
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            raise
    
    def load_and_parse_documents(self, docs_path: str = DOCS_PATH, force_reprocess: bool = False) -> List[Dict[str, Any]]:
        """Load and parse all documents in a directory with caching."""
        all_chunks = []
        processed_files = []
        skipped_files = []
        
        print(f"🔍 Scanning for documents in: {docs_path}")
        
        if not os.path.exists(docs_path):
            print(f"❌ Documents directory not found: {docs_path}")
            return all_chunks
        
        for filename in os.listdir(docs_path):
            if filename.startswith('~') or filename.startswith('.'):
                continue
                
            file_path = os.path.join(docs_path, filename)
            if os.path.isfile(file_path):
                file_ext = Path(filename).suffix.lower()
                
                if file_ext not in ['.pdf', '.docx']:
                    skipped_files.append(filename)
                    continue
                
                try:
                    chunks = self.parse_document(file_path, force_reprocess)
                    all_chunks.extend(chunks)
                    processed_files.append(filename)
                except Exception as e:
                    print(f"❌ Could not parse {filename}: {e}")
                    skipped_files.append(filename)
        
        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Processed: {len(processed_files)} files")
        print(f"   ⏭️  Skipped: {len(skipped_files)} files")
        print(f"   📄 Total chunks: {len(all_chunks)}")
        
        if skipped_files:
            print(f"   ⚠️  Skipped files: {', '.join(skipped_files)}")
        
        return all_chunks
    
    def clear_cache(self, file_path: Optional[str] = None):
        """Clear cache for specific file or all files."""
        if file_path:
            cache_path = self._get_cache_path(file_path)
            if cache_path.exists():
                cache_path.unlink()
                print(f"🗑️  Cleared cache for {Path(file_path).name}")
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print(f"🗑️  Cleared all cached chunks")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached chunks."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_info = {
            'cache_directory': str(self.cache_dir),
            'total_cached_files': len(cache_files),
            'cached_files': []
        }
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                cache_info['cached_files'].append({
                    'filename': Path(cached_data.get('file_path', '')).name,
                    'created_at': cached_data.get('created_at'),
                    'chunks_count': len(cached_data.get('chunks', [])),
                    'file_hash': cached_data.get('file_hash')[:8] + '...'
                })
            except Exception as e:
                cache_info['cached_files'].append({
                    'filename': cache_file.stem,
                    'error': str(e)
                })
        
        return cache_info 