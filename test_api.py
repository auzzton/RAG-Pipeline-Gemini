import requests

API_URL = "http://127.0.0.1:8000/hackrx/run"
API_KEY = "AIzaSyDtV1QfxjAgJ-HiAkltaveuUCTPW6CdFec"  # Replace with your actual API key from .env
DOCUMENT_PATH = "C:/Projects/Bajaj_new/Bajaj-hackathon/data/docs/bajaj document 5.pdf"  # Absolute path to your document

payload = {
    "documents": "C:/Projects/Bajaj_new/Bajaj-hackathon/data/docs/bajaj document 5.pdf",
    "questions": [
        "What are the expenses that would be covered in the given insurance policy?",
        "Will the policy cover the cost of my Knee surgery?",
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.post(API_URL, json=payload, headers=headers)
print("Status Code:", response.status_code)
print("Response:", response.json())
