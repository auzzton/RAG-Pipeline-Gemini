#!/usr/bin/env python3
"""
Demo script for the Enhanced Chunking System
Shows persistent storage, optimal chunking strategies, and performance improvements.
"""

import os
import time
from pathlib import Path

from parsing.enhanced_parser import EnhancedParser
from retrieval.embedder import EmbeddingModel
from retrieval.enhanced_retriever import EnhancedVectorStore
from utils.config import DOCS_PATH

def demo_enhanced_chunking():
    """Demonstrate the enhanced chunking system."""
    
    print("🚀 Enhanced Chunking System Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n1️⃣  Initializing Enhanced Components...")
    parser = EnhancedParser()
    embedder = EmbeddingModel()
    vector_store = EnhancedVectorStore(embedder)
    
    # Show initial cache info
    print("\n2️⃣  Initial Cache Status:")
    cache_info = parser.get_cache_info()
    print(f"   📁 Cache directory: {cache_info['cache_directory']}")
    print(f"   📄 Cached files: {cache_info['total_cached_files']}")
    
    # Show initial collection stats
    print("\n3️⃣  Initial Collection Status:")
    stats = vector_store.get_collection_stats()
    print(f"   📊 Total documents: {stats.get('total_documents', 0)}")
    print(f"   📁 Document types: {stats.get('document_types', {})}")
    
    # Process documents (first run)
    print("\n4️⃣  First Run - Processing Documents...")
    start_time = time.time()
    chunks = parser.load_and_parse_documents()
    first_run_time = time.time() - start_time
    
    if chunks:
        print(f"   ⏱️  First run time: {first_run_time:.2f} seconds")
        print(f"   📄 Chunks generated: {len(chunks)}")
        
        # Add to vector store
        result = vector_store.add_documents(chunks)
        print(f"   ✅ Added to vector store: {result['added']} chunks")
    else:
        print("   ❌ No documents found to process")
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
    print(f"   🚀 Speed improvement: {first_run_time/second_run_time:.1f}x faster")
    
    # Show final collection stats
    print("\n7️⃣  Final Collection Statistics:")
    stats = vector_store.get_collection_stats()
    print(f"   📊 Total documents: {stats.get('total_documents', 0)}")
    print(f"   📁 Document types: {stats.get('document_types', {})}")
    print(f"   📚 Sources: {list(stats.get('sources', {}).keys())}")
    print(f"   📏 Average chunk size: {stats.get('avg_chunk_size', 0):.0f} characters")
    
    # Demonstrate chunking strategies
    print("\n8️⃣  Chunking Strategies Used:")
    doc_types = stats.get('document_types', {})
    for doc_type, count in doc_types.items():
        strategy = parser.chunking_strategies.get(doc_type, parser.chunking_strategies["default"])
        print(f"   📋 {doc_type}: {count} chunks (size: {strategy['chunk_size']}, overlap: {strategy['chunk_overlap']})")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Key Benefits Demonstrated:")
    print("   • Persistent chunk storage (no reprocessing)")
    print("   • Optimal chunking strategies per document type")
    print("   • Significant performance improvements")
    print("   • Intelligent document change detection")
    print("   • Comprehensive metadata tracking")

def demo_chunking_strategies():
    """Demonstrate different chunking strategies."""
    
    print("\n🎯 Chunking Strategies Demo")
    print("=" * 40)
    
    parser = EnhancedParser()
    
    print("\n📋 Available Chunking Strategies:")
    for strategy_name, strategy_config in parser.chunking_strategies.items():
        print(f"   📄 {strategy_name.upper()}:")
        print(f"      • Chunk size: {strategy_config['chunk_size']} characters")
        print(f"      • Overlap: {strategy_config['chunk_overlap']} characters")
        print(f"      • Separators: {strategy_config['separators'][:2]}...")
    
    print("\n💡 Strategy Selection Logic:")
    print("   • Legal: Smaller chunks for precise clause matching")
    print("   • Medical: Very small chunks for specific procedures")
    print("   • Technical: Larger chunks for comprehensive context")
    print("   • Financial: Medium chunks for policy details")
    print("   • Default: Balanced approach for general documents")

if __name__ == "__main__":
    try:
        demo_enhanced_chunking()
        demo_chunking_strategies()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have documents in the data/docs directory") 