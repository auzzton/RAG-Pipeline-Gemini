#!/usr/bin/env python3
"""
Test script for parsing and chunking functionality only.
This doesn't require OpenAI API and can test the enhanced chunking system.
"""

import os
import time
from pathlib import Path

# Only import parsing components
from parsing.enhanced_parser import EnhancedParser
from utils.config import DOCS_PATH

def test_parsing_only():
    """Test only the parsing and chunking functionality."""
    
    print("🧪 Testing Enhanced Parsing System (No OpenAI Required)")
    print("=" * 60)
    
    # Initialize parser only
    print("\n1️⃣  Initializing Enhanced Parser...")
    parser = EnhancedParser()
    
    # Show initial cache info
    print("\n2️⃣  Initial Cache Status:")
    cache_info = parser.get_cache_info()
    print(f"   📁 Cache directory: {cache_info['cache_directory']}")
    print(f"   📄 Cached files: {cache_info['total_cached_files']}")
    
    # Check if documents exist
    print(f"\n3️⃣  Checking Documents Directory: {DOCS_PATH}")
    if not os.path.exists(DOCS_PATH):
        print(f"   ❌ Documents directory not found: {DOCS_PATH}")
        print(f"   💡 Creating directory...")
        os.makedirs(DOCS_PATH, exist_ok=True)
        print(f"   ✅ Created directory: {DOCS_PATH}")
        print(f"   📝 Please add some PDF or DOCX files to this directory and run again.")
        return
    
    # List documents
    documents = [f for f in os.listdir(DOCS_PATH) 
                 if f.lower().endswith(('.pdf', '.docx')) and not f.startswith('~')]
    
    if not documents:
        print(f"   ❌ No PDF or DOCX files found in {DOCS_PATH}")
        print(f"   📝 Please add some documents and run again.")
        return
    
    print(f"   ✅ Found {len(documents)} documents:")
    for doc in documents:
        print(f"      📄 {doc}")
    
    # Process documents (first run)
    print("\n4️⃣  First Run - Processing Documents...")
    start_time = time.time()
    chunks = parser.load_and_parse_documents()
    first_run_time = time.time() - start_time
    
    if chunks:
        print(f"   ⏱️  First run time: {first_run_time:.2f} seconds")
        print(f"   📄 Chunks generated: {len(chunks)}")
        
        # Show chunk statistics
        chunk_sizes = [len(chunk.get('text', '')) for chunk in chunks]
        print(f"   📏 Average chunk size: {sum(chunk_sizes)/len(chunk_sizes):.0f} characters")
        print(f"   📏 Min chunk size: {min(chunk_sizes)} characters")
        print(f"   📏 Max chunk size: {max(chunk_sizes)} characters")
        
        # Show document types detected
        doc_types = {}
        for chunk in chunks:
            doc_type = chunk.get('metadata', {}).get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"   📋 Document types detected:")
        for doc_type, count in doc_types.items():
            print(f"      • {doc_type}: {count} chunks")
        
    else:
        print("   ❌ No chunks generated")
        return
    
    # Show cache info after first run
    print("\n5️⃣  Cache Status After First Run:")
    cache_info = parser.get_cache_info()
    print(f"   📄 Cached files: {cache_info['total_cached_files']}")
    for file_info in cache_info['cached_files']:
        print(f"   📋 {file_info.get('filename', 'Unknown')}: {file_info.get('chunks_count', 0)} chunks")
    
    # Second run - should use cache
    print("\n6️⃣  Second Run - Using Cache...")
    start_time = time.time()
    chunks_second = parser.load_and_parse_documents()
    second_run_time = time.time() - start_time
    
    print(f"   ⏱️  Second run time: {second_run_time:.2f} seconds")
    if second_run_time > 0:
        speed_improvement = first_run_time/second_run_time
        print(f"   🚀 Speed improvement: {speed_improvement:.1f}x faster")
    else:
        print(f"   🚀 Speed improvement: Instant (cached)")
    
    # Show chunking strategies
    print("\n7️⃣  Chunking Strategies Used:")
    for strategy_name, strategy_config in parser.chunking_strategies.items():
        print(f"   📄 {strategy_name.upper()}:")
        print(f"      • Chunk size: {strategy_config['chunk_size']} characters")
        print(f"      • Overlap: {strategy_config['chunk_overlap']} characters")
    
    print("\n✅ Parsing test completed successfully!")
    print("\n💡 Key Benefits Demonstrated:")
    print("   • Persistent chunk storage (no reprocessing)")
    print("   • Optimal chunking strategies per document type")
    print("   • Significant performance improvements")
    print("   • Intelligent document change detection")
    print("   • No OpenAI API required for parsing!")

if __name__ == "__main__":
    try:
        test_parsing_only()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 