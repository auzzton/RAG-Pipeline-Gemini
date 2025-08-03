import google.generativeai as genai
import os

# Set the API key directly
GEMINI_API_KEY = "AIzaSyDtV1QfxjAgJ-HiAkltaveuUCTPW6CdFec"

print("🌟 Testing Gemini API Key Directly")
print("=" * 50)

try:
    # Configure the API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Create a model instance
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Test with a simple prompt
    print("🤖 Testing API with simple prompt...")
    response = model.generate_content("Hello! Please respond with 'Gemini API is working!' if you can see this message.")
    
    print("✅ SUCCESS! Gemini API is working!")
    print(f"📝 Response: {response.text}")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    print("🔍 This means either:")
    print("   - The API key is invalid")
    print("   - There's a network issue")
    print("   - The Gemini service is down") 