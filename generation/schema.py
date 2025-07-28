from pydantic import BaseModel, Field
from typing import List, Optional

class StructuredQuery(BaseModel):
    age: Optional[int] = Field(None, description="Age of the person")
    gender: Optional[str] = Field(None, description="Gender of the person")
    medical_procedure: Optional[str] = Field(None, description="The medical procedure in question")
    location: Optional[str] = Field(None, description="Location where the procedure took place")
    policy_duration_months: Optional[int] = Field(None, description="Duration of the insurance policy in months")

class Source(BaseModel):
    chunk: str = Field(..., description="The exact text chunk from the source document.")
    source: str = Field(..., description="The filename of the source document.")
    confidence: float = Field(..., description="The retrieval confidence score.")

class FinalResponse(BaseModel):
    decision: str = Field(..., description="The final decision, either 'Approved' or 'Rejected'.")
    amount: Optional[str] = Field(None, description="The approved amount, if applicable.")
    justification: str = Field(..., description="A detailed justification for the decision, referencing specific clauses.")
    sources: List[Source] = Field(..., description="A list of source chunks that support the decision.")
