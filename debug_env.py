import os
from dotenv import load_dotenv
print("🔍 Debugging Environment Variables")
print("=" * 50)

# Load .env file
load_dotenv()

# Check if .env file exists
env_file_path = ".env"
if os.path.exists(env_file_path):
    print(f"✅ .env file found at: {os.path.abspath(env_file_path)}")
    with open(env_file_path, 'r') as f:
        content = f.read()
        print(f"📄 .env file content:\n{content}")
else:
    print(f"❌ .env file not found at: {os.path.abspath(env_file_path)}")

# Check environment variables
print("\n🔧 Environment Variables:")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY', 'NOT SET')}")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'NOT SET')}")
print(f"GEMINI_MODEL: {os.getenv('GEMINI_MODEL', 'NOT SET')}")

# Check if keys are valid
gemini_key = os.getenv('GEMINI_API_KEY')
if gemini_key and gemini_key != 'NOT SET':
    print(f"✅ GEMINI_API_KEY is set and valid: {gemini_key[:10]}...")
else:
    print("❌ GEMINI_API_KEY is not set or invalid") 