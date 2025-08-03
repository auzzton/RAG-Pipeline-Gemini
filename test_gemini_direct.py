import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("🌟 Testing Gemini API Key Directly")
print("=" * 50)

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in environment variables")
    print("💡 Please set GEMINI_API_KEY in your .env file")
    exit(1)

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
