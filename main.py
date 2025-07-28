import json
import os
import datetime

from parsing.parser import load_and_parse_documents
from retrieval.embedder import EmbeddingModel
from retrieval.retriever import VectorStore
from generation.generator import Generator
from utils.config import DOCS_PATH

def setup_pipeline():
    """Initializes and sets up the RAG pipeline components."""
    print("--- Initializing RAG Pipeline ---")
    embedder = EmbeddingModel()
    vector_store = VectorStore(embedder)
    generator = Generator()

    # Always clear the collection for a fresh start
    print("Clearing existing vector store collection...")
    vector_store.collection.delete(ids=vector_store.collection.get()["ids"])
    
    if vector_store.collection.count() == 0:
        print("Vector store is empty. Processing documents...")
        if not os.listdir(DOCS_PATH):
            print(f"Warning: No documents found in {DOCS_PATH}. Please add policy documents to this directory.")
            return None, None, None

        chunks = load_and_parse_documents()
        vector_store.add_documents(chunks)
    else:
        print("Vector store already contains documents.")

    print("--- Pipeline Ready ---")
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
    embedder, vector_store, generator = setup_pipeline()

    if not all([embedder, vector_store, generator]):
        print("Pipeline setup failed. Exiting.")
        return

    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        print(f"\nProcessing query: '{query}'")

        # Step 1: Structured query extraction
        print("\nStep 1: Parsing query with LLM...")
        structured_query = generator.extract_structured_query(query)
        print(f"  -> Structured Query: {structured_query.model_dump_json(indent=2)}")

        # Step 2: Retrieval
        print("\nStep 2: Retrieving relevant documents...")
        retrieved_chunks = vector_store.retrieve_relevant_chunks(query)
        if not retrieved_chunks:
            print("  -> No relevant documents found.")
            continue
        print(f"  -> Retrieved {len(retrieved_chunks)} chunks.")

        # Step 3: Response Generation
        print("\nStep 3: Generating final response with LLM...")
        final_response = generator.generate_response(structured_query, retrieved_chunks)

        # Step 4: Output and Logging
        print("\n--- Final Response ---")
        print(final_response.model_dump_json(indent=4))
        print("\n----------------------")

        # Log interaction
        log_interaction(query, structured_query, retrieved_chunks, final_response)

if __name__ == "__main__":
    main()
