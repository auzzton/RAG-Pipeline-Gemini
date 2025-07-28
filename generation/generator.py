import json
import os
from typing import List, Dict, Any
from pydantic import ValidationError

from openai import OpenAI
from utils.config import OPENAI_API_KEY, GPT_MODEL
from generation.schema import StructuredQuery, FinalResponse


class Generator:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = GPT_MODEL
        print(f"✅ OpenAI generator initialized with model: {self.model}")

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

    def extract_structured_query(self, query: str) -> StructuredQuery:
        """Uses OpenAI to extract structured information from a raw query."""
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

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        raw_output = response.choices[0].message.content

        try:
            json_block = self._extract_first_json_block(raw_output)
            extracted_json = json.loads(json_block)
            return StructuredQuery(**extracted_json)
        except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as e:
            print(f"❌ Error parsing StructuredQuery: {e}")
            print("🔎 LLM raw output:\n", raw_output)

            os.makedirs("logs", exist_ok=True)
            with open("logs/structured_query_error.txt", "w", encoding="utf-8") as f:
                f.write(raw_output)

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
        1. Analyze: Review the user's claim against the policy clauses.
        2. Decide: Approve or Reject.
        3. Justify: Explain with specific clause references.
        4. Amount: Include amount if applicable, or null.
        5. Format: Return a JSON with:
        {{
            "decision": "<Approved/Rejected>",
            "amount": "<e.g., ₹80000 or null>",
            "justification": "<Detailed explanation with clause references>",
            "sources": [
                {{
                    "chunk": "<Text of clause used>",
                    "source": "<source filename>",
                    "confidence": <confidence score>
                }}
            ]
        }}

        Only return the JSON object. Do not include any explanation or narration.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        raw_output = response.choices[0].message.content

        try:
            json_block = self._extract_first_json_block(raw_output)
            response_json = json.loads(json_block)
            return FinalResponse(**response_json)
        except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as e:
            print(f"❌ Error parsing FinalResponse: {e}")
            print("🔎 LLM raw output:\n", raw_output)

            os.makedirs("logs", exist_ok=True)
            with open("logs/final_response_error.txt", "w", encoding="utf-8") as f:
                f.write(raw_output)

            return FinalResponse(
                decision="Error",
                justification=f"Failed to generate a valid response: {str(e)}",
                sources=[]
            )
