#!/usr/bin/env python3
"""
Test script for Gemini API integration.
This tests the enhanced generator with Gemini API.
"""

import os
from generation.enhanced_generator import EnhancedGenerator

def test_gemini_integration():
    """Test the Gemini API integration."""
    
    print("🌟 Testing Gemini API Integration")
    print("=" * 50)
    
    # Initialize the enhanced generator
    print("\n1️⃣  Initializing Enhanced Generator...")
    generator = EnhancedGenerator()
    
    # Check API status
    print("\n2️⃣  API Status:")
    api_status = generator.get_api_status()
    print(f"   🎯 Active client: {api_status['active_client']}")
    print(f"   🤖 OpenAI available: {api_status['openai_available']}")
    print(f"   🌟 Gemini available: {api_status['gemini_available']}")
    
    if api_status['gemini_model']:
        print(f"   📋 Gemini model: {api_status['gemini_model']}")
    
    if not api_status['gemini_available']:
        print("   ❌ Gemini API not available. Please check your GEMINI_API_KEY.")
        return
    
    # Test simple question answering
    print("\n3️⃣  Testing Simple Question Answering...")
    test_question = "What is the capital of France?"
    test_context = [
        {
            "chunk": "Paris is the capital and largest city of France. It is known for its art, fashion, gastronomy and culture.",
            "source": "test_document.txt",
            "confidence": 0.95
        }
    ]
    
    try:
        answer = generator.answer_direct_question(test_question, test_context)
        print(f"   ✅ Question: {test_question}")
        print(f"   🌟 Answer: {answer}")
    except Exception as e:
        print(f"   ❌ Error testing question answering: {e}")
    
    # Test structured query extraction
    print("\n4️⃣  Testing Structured Query Extraction...")
    test_query = "I need knee surgery, I'm 45 years old male from Mumbai"
    
    try:
        structured_query = generator.extract_structured_query(test_query)
        print(f"   ✅ Query: {test_query}")
        print(f"   📋 Structured: {structured_query.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"   ❌ Error testing structured query: {e}")
    
    # Test response generation
    print("\n5️⃣  Testing Response Generation...")
    test_structured_query = structured_query if 'structured_query' in locals() else None
    test_chunks = [
        {
            "chunk": "Knee surgery is covered under our health insurance policy for patients aged 18-65. The coverage includes up to 80% of the total cost.",
            "source": "policy_document.pdf",
            "confidence": 0.92
        },
        {
            "chunk": "Pre-authorization is required for all surgical procedures. Please submit medical reports and doctor's recommendation at least 7 days before the procedure.",
            "source": "policy_document.pdf", 
            "confidence": 0.88
        }
    ]
    
    if test_structured_query:
        try:
            response = generator.generate_response(test_structured_query, test_chunks)
            print(f"   ✅ Generated response:")
            print(f"   📄 Decision: {response.decision}")
            print(f"   💰 Amount: {response.amount}")
            print(f"   📝 Justification: {response.justification[:100]}...")
        except Exception as e:
            print(f"   ❌ Error testing response generation: {e}")
    
    print("\n✅ Gemini integration test completed!")
    print("\n💡 If all tests passed, your Gemini API is working correctly!")

if __name__ == "__main__":
    try:
        test_gemini_integration()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 