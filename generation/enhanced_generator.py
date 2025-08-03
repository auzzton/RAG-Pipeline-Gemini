import json
import os
from typing import List, Dict, Any
from pydantic import ValidationError

from utils.config import OPENAI_API_KEY, GEMINI_API_KEY, GPT_MODEL, GEMINI_MODEL, USE_GEMINI
from generation.schema import StructuredQuery, FinalResponse

class EnhancedGenerator:
    """
    Enhanced generator that supports both OpenAI and Gemini APIs.
    Automatically falls back to available API.
    """
    
    def __init__(self):
        self.use_gemini = USE_GEMINI
        self.openai_client = None
        self.gemini_client = None
        
        # Initialize OpenAI if available
        if OPENAI_API_KEY and OPENAI_API_KEY != "dummy-key-for-parsing-only":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                print(f"✅ OpenAI generator initialized with model: {GPT_MODEL}")
            except Exception as e:
                print(f"⚠️  OpenAI initialization failed: {e}")
        
        # Initialize Gemini if available
        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_client = genai.GenerativeModel(GEMINI_MODEL)
                print(f"✅ Gemini generator initialized with model: {GEMINI_MODEL}")
            except Exception as e:
                print(f"⚠️  Gemini initialization failed: {e}")
        
        # Determine which API to use
        if self.gemini_client and self.use_gemini:
            self.active_client = "gemini"
            print("🎯 Using Gemini API for generation")
        elif self.openai_client:
            self.active_client = "openai"
            print("🎯 Using OpenAI API for generation")
        else:
            self.active_client = None
            print("⚠️  No API clients available. Generation will be limited.")
    
    def _extract_first_json_block(self, text: str) -> str:
        """Extracts the first valid JSON object from text using brace matching."""
        brace_stack = []
        json_start = None

        for i, char in enumerate(text):
            if char == '{':
                if not brace_stack:
                    json_start = i
                brace_stack.append(char)
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and json_start is not None:
                        return text[json_start:i + 1]

        raise ValueError("No valid JSON object found in text.")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        response = self.openai_client.chat.completions.create(
            model=GPT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        return response.choices[0].message.content

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini API."""
        if not self.gemini_client:
            raise Exception("Gemini client not available")
        
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = self.gemini_client.generate_content(full_prompt)
        
        # Extract JSON from response
        response_text = response.text
        
        # Try to extract JSON if it's wrapped in markdown
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        
        return response_text

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call the appropriate API based on availability."""
        if self.active_client == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.active_client == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        else:
            raise Exception("No API clients available")

    def extract_structured_query(self, query: str) -> StructuredQuery:
        """Uses available API to extract structured information from a raw query."""
        system_prompt = "You are an expert at extracting structured information from text and returning it as JSON."

        user_prompt = f"""
        Extract the following fields from the user query: age, gender, medical_procedure, location, policy_duration_months.
        User Query: "{query}"
        Return a valid JSON object with the extracted fields. If a field is not present, its value should be null.
        Example:
        {{
            "age": 46,
            "gender": "male",
            "medical_procedure": "knee surgery",
            "location": "Pune",
            "policy_duration_months": 3
        }}
        Only return the JSON object. Do not include any explanation or narration.
        """

        try:
            raw_output = self._call_api(system_prompt, user_prompt)
            
            try:
                json_block = self._extract_first_json_block(raw_output)
                extracted_json = json.loads(json_block)
                return StructuredQuery(**extracted_json)
            except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as e:
                print(f"❌ Error parsing StructuredQuery: {e}")
                print("🔎 API raw output:\n", raw_output)

                os.makedirs("logs", exist_ok=True)
                with open("logs/structured_query_error.txt", "w", encoding="utf-8") as f:
                    f.write(raw_output)

                return StructuredQuery()  # fallback empty object
                
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return StructuredQuery()  # fallback empty object

    def generate_response(self, structured_query: StructuredQuery, retrieved_chunks: List[Dict[str, Any]]) -> FinalResponse:
        """Generates a final decision and justification based on the query and retrieved context."""
        context = "\n\n---\n\n".join(
            [f"Source: {chunk['source']}\nContent: {chunk['chunk']}" for chunk in retrieved_chunks]
        )
        query_details = json.dumps(structured_query.model_dump(exclude_none=True), indent=2)

        system_prompt = "You are an insurance claim evaluation expert who provides responses in JSON format."

        user_prompt = f"""
        You are an insurance claim evaluation expert. Based on the user's claim details and the provided policy document clauses, please make a decision.

        **User Claim Details:**
        {query_details}

        **Retrieved Policy Clauses:**
        {context}

        **Your Task:**
        Analyze the claim details against the policy clauses and provide a structured response in JSON format.

        **Response Format:**
        {{
            "decision": "Approved" or "Rejected",
            "amount": "The approved amount if applicable, otherwise null",
            "justification": "Detailed explanation referencing specific policy clauses",
            "sources": [
                {{
                    "chunk": "Exact text from source document",
                    "source": "Document filename",
                    "confidence": 0.95
                }}
            ]
        }}

        **Guidelines:**
        1. Be thorough in your analysis
        2. Reference specific policy clauses in your justification
        3. Provide clear reasoning for your decision
        4. Include relevant source chunks with confidence scores
        5. Return only valid JSON

        Return the response as a valid JSON object.
        """

        try:
            raw_output = self._call_api(system_prompt, user_prompt)
            
            try:
                json_block = self._extract_first_json_block(raw_output)
                extracted_json = json.loads(json_block)
                return FinalResponse(**extracted_json)
            except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as e:
                print(f"❌ Error parsing FinalResponse: {e}")
                print("🔎 API raw output:\n", raw_output)

                os.makedirs("logs", exist_ok=True)
                with open("logs/final_response_error.txt", "w", encoding="utf-8") as f:
                    f.write(raw_output)

                # Return fallback response
                return FinalResponse(
                    decision="Unable to process",
                    justification="Error in response generation. Please try again.",
                    sources=[]
                )
                
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return FinalResponse(
                decision="Error",
                justification=f"API call failed: {str(e)}",
                sources=[]
            )

    def answer_direct_question(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Answer a question directly using the retrieved context."""
        context = "\n\n---\n\n".join(
            [f"Source: {chunk['source']}\nContent: {chunk['chunk']}" for chunk in retrieved_chunks]
        )

        system_prompt = "You are a helpful assistant that answers questions based on provided document content."

        user_prompt = f"""
        Based on the following document content, please answer this question: "{question}"

        **Document Content:**
        {context}

        **Instructions:**
        1. Answer the question based only on the provided document content
        2. If the information is not available in the content, say so clearly
        3. Be concise but thorough
        4. Reference specific parts of the document when possible

        Please provide a clear and helpful answer.
        """

        try:
            return self._call_api(system_prompt, user_prompt)
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return f"Unable to generate response: {str(e)}"

    def get_api_status(self) -> Dict[str, Any]:
        """Get the status of available APIs."""
        return {
            "active_client": self.active_client,
            "openai_available": self.openai_client is not None,
            "gemini_available": self.gemini_client is not None,
            "openai_model": GPT_MODEL if self.openai_client else None,
            "gemini_model": GEMINI_MODEL if self.gemini_client else None
        } 