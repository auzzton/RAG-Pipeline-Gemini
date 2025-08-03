import json
import os
import datetime

from parsing.enhanced_parser import EnhancedParser
from retrieval.embedder import EmbeddingModel
from retrieval.faiss_vector_store import FAISSVectorStore
from generation.enhanced_generator import EnhancedGenerator
from utils.config import DOCS_PATH

def validate_configuration():
    """Validates that all required configuration is in place."""
    print("🔧 Validating configuration...")
    
    # Check if DOCS_PATH exists
    if not os.path.exists(DOCS_PATH):
        print(f"❌ Documents directory not found: {DOCS_PATH}")
        print(f"💡 Creating directory: {DOCS_PATH}")
        os.makedirs(DOCS_PATH, exist_ok=True)
    
    # Check if there are any documents
    if not os.listdir(DOCS_PATH):
        print(f"⚠️  No documents found in {DOCS_PATH}")
        print("💡 Please add PDF or DOCX files to this directory before running queries.")
        return False
    
    # Check for supported file types
    supported_files = [f for f in os.listdir(DOCS_PATH) 
                      if f.lower().endswith(('.pdf', '.docx')) and not f.startswith('~')]
    
    if not supported_files:
        print(f"❌ No supported files (PDF/DOCX) found in {DOCS_PATH}")
        return False
    
    print(f"✅ Found {len(supported_files)} supported documents")
    return True

def setup_pipeline():
    """Initializes and sets up the enhanced RAG pipeline components."""
    print("🚀 Initializing Enhanced RAG Pipeline ---")
    
    # Initialize components
    embedder = EmbeddingModel()
    # Determine embedding dimension from model
    test_embed = embedder.create_embeddings(["test"])
    dim = test_embed.shape[1] if len(test_embed.shape) > 1 else 384
    vector_store = FAISSVectorStore(dim)
    generator = EnhancedGenerator()
    parser = EnhancedParser()
    
    # Get collection statistics
    stats = vector_store.get_collection_stats()
    print(f"📊 Current collection: {stats.get('total_documents', 0)} documents")
    
    # Process documents if needed
    if stats.get('total_documents', 0) == 0:
        print("📄 Vector store is empty. Processing documents...")
        if not os.listdir(DOCS_PATH):
            print(f"⚠️  No documents found in {DOCS_PATH}. Please add policy documents to this directory.")
            return None, None, None

        chunks = parser.load_and_parse_documents()
        if chunks:
            result = vector_store.add_documents(chunks)
            print(f"✅ Pipeline setup complete: {result['added']} chunks added")
        else:
            print("❌ No chunks generated. Pipeline setup failed.")
            return None, None, None
    else:
        print("✅ Using existing processed documents.")
        
        # Show cache info
        cache_info = parser.get_cache_info()
        print(f"📁 Cached documents: {cache_info['total_cached_files']}")

    print("🎯 Enhanced Pipeline Ready!")
    return embedder, vector_store, generator

def log_interaction(query, structured_query, retrieved_chunks, final_response):
    """Logs the full query-response interaction for evaluation/debugging."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_data = {
        "query": query,
        "structured_query": structured_query.model_dump(),
        "retrieved_chunks": retrieved_chunks,
        "final_response": final_response.model_dump()
    }

    with open(f"logs/{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

def main():
    """Main function to run the RAG query process."""
    # Validate configuration first
    if not validate_configuration():
        print("❌ Configuration validation failed. Please check the setup and try again.")
        return
    
    # Setup pipeline
    embedder, vector_store, generator = setup_pipeline()

    if not all([embedder, vector_store, generator]):
        print("❌ Pipeline setup failed. Exiting.")
        return

    print("\n🚀 RAG System Ready! Type 'exit' to quit.")
    print("💡 Example queries:")
    print("   - 'I need knee surgery, I'm 45 years old male from Mumbai'")
    print("   - 'What's covered for dental procedures?'")
    print("   - 'Is heart surgery covered for a 60-year-old female?'")

    while True:
        try:
            query = input("\nEnter your query (or type 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                print("👋 Goodbye!")
                break
            elif query.lower() in ['help', '?', 'h']:
                print("\n📖 Available commands:")
                print("   'help' or '?' - Show this help message")
                print("   'exit' - Quit the application")
                print("   'stats' - Show collection statistics")
                print("   'cache' - Show cache information")
                print("   'reprocess' - Force reprocess all documents")
                print("   'api' - Show API status")
                print("   Any other text - Process as a query")
                print("\n💡 Example queries:")
                print("   - 'I need knee surgery, I'm 45 years old male from Mumbai'")
                print("   - 'What's covered for dental procedures?'")
                print("   - 'Is heart surgery covered for a 60-year-old female?'")
                continue
            elif query.lower() == 'stats':
                try:
                    stats = vector_store.get_collection_stats()
                    print("\n📊 Collection Statistics:")
                    print(f"   📄 Total documents: {stats.get('total_documents', 0)}")
                    print(f"   📁 Document types: {stats.get('document_types', {})}")
                    print(f"   📚 Sources: {list(stats.get('sources', {}).keys())}")
                    print(f"   📏 Avg chunk size: {stats.get('avg_chunk_size', 0):.0f} chars")
                    print(f"   🕒 Last updated: {stats.get('last_updated', 'Unknown')}")
                except Exception as e:
                    print(f"❌ Error getting stats: {e}")
                continue
            elif query.lower() == 'cache':
                try:
                    parser = EnhancedParser()
                    cache_info = parser.get_cache_info()
                    print(f"\n📁 Cache Information:")
                    print(f"   📂 Cache directory: {cache_info['cache_directory']}")
                    print(f"   📄 Cached files: {cache_info['total_cached_files']}")
                    for file_info in cache_info['cached_files'][:5]:  # Show first 5
                        print(f"   📋 {file_info.get('filename', 'Unknown')}: {file_info.get('chunks_count', 0)} chunks")
                    if len(cache_info['cached_files']) > 5:
                        print(f"   ... and {len(cache_info['cached_files']) - 5} more files")
                except Exception as e:
                    print(f"❌ Error getting cache info: {e}")
                continue
            elif query.lower() == 'reprocess':
                try:
                    print("🔄 Force reprocessing all documents...")
                    parser = EnhancedParser()
                    chunks = parser.load_and_parse_documents(force_reprocess=True)
                    if chunks:
                        result = vector_store.add_documents(chunks, force_reprocess=True)
                        print(f"✅ Reprocessing complete: {result['added']} chunks added")
                    else:
                        print("❌ No chunks generated during reprocessing")
                except Exception as e:
                    print(f"❌ Error during reprocessing: {e}")
                continue
            elif query.lower() == 'api':
                try:
                    api_status = generator.get_api_status()
                    print(f"\n🔌 API Status:")
                    print(f"   🎯 Active client: {api_status['active_client']}")
                    print(f"   🤖 OpenAI available: {api_status['openai_available']}")
                    print(f"   🌟 Gemini available: {api_status['gemini_available']}")
                    if api_status['openai_model']:
                        print(f"   📋 OpenAI model: {api_status['openai_model']}")
                    if api_status['gemini_model']:
                        print(f"   📋 Gemini model: {api_status['gemini_model']}")
                except Exception as e:
                    print(f"❌ Error getting API status: {e}")
                continue
            
            if not query:
                print("❌ Please enter a valid query.")
                continue

            print(f"\n🔄 Processing query: '{query}'")

            # Step 1: Structured query extraction
            print("\n📋 Step 1: Parsing query with LLM...")
            try:
                structured_query = generator.extract_structured_query(query)
                print(f"  ✅ Structured Query: {structured_query.model_dump_json(indent=2)}")
            except Exception as e:
                print(f"  ❌ Error in structured query extraction: {str(e)}")
                print("  🔄 Continuing with raw query...")
                continue

            # Step 2: Retrieval
            print("\n🔍 Step 2: Retrieving relevant documents...")
            try:
                retrieved_chunks = vector_store.retrieve_relevant_chunks(query)
                if not retrieved_chunks:
                    print("  ⚠️  No relevant documents found.")
                    print("  💡 Try rephrasing your question or check if relevant documents are loaded.")
                    continue
                print(f"  ✅ Retrieved {len(retrieved_chunks)} chunks.")
            except Exception as e:
                print(f"  ❌ Error in document retrieval: {str(e)}")
                continue

            # Step 3: Response Generation
            print("\n🤖 Step 3: Generating final response with LLM...")
            try:
                final_response = generator.generate_response(structured_query, retrieved_chunks)
            except Exception as e:
                print(f"  ❌ Error in response generation: {str(e)}")
                continue

            # Step 4: Output and Logging
            print("\n📄 --- Final Response ---")
            print(final_response.model_dump_json(indent=4))
            print("📄 ----------------------")

            # Log interaction
            try:
                log_interaction(query, structured_query, retrieved_chunks, final_response)
                print("  📝 Interaction logged successfully.")
            except Exception as e:
                print(f"  ⚠️  Warning: Failed to log interaction: {str(e)}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("🔄 Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()
