from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from src.state import PatientState

# 1. Define the schema for a single diagnosis
class AdjustedDiagnosis(BaseModel):
    condition: str = Field(description="The name of the medical condition")
    likelihood: str = Field(description="High, Medium, or Low")
    justification: str = Field(description="Clinical reasoning incorporating demographics and guidelines")

# 2. Define the schema for the agent's overall JSON response
class DemographicAdjustmentOutput(BaseModel):
    chain_of_thought: str = Field(description="A verbose, detailed paragraph explaining step-by-step how the physical symptoms match or do not match the RAG guidelines.")
    adjusted_differential: List[AdjustedDiagnosis] = Field(description="List of exactly 3 adjusted potential conditions")
    adjustment_rationale: str = Field(description="Brief summary of why you finalized this list")

DEMOGRAPHIC_PROMPT = """You are an expert Clinical Epidemiologist.
Review the Blinded Differential Diagnosis, the Patient Demographics, and the Retrieved Clinical Guidelines (RAG Context).

Your task is to adjust the differential diagnosis safely. 
You MUST adhere to the RAG Context provided. If the RAG context forbids a psychiatric diagnosis, you must not suggest one.

Patient Demographics: {demographics}
Blinded Differential: {blinded_differential}
Retrieved Guidelines (RAG): {rag_context}

CRITICAL: For your 'chain_of_thought', you must write a highly detailed paragraph explaining your logic BEFORE you output the adjusted list. Do not use placeholders.
"""

def run_demographic_agent(state: PatientState) -> dict:
    print("\n[Agent] Demographic Agent (Epidemiologist) is thinking...")
    
    # 3. Use ChatOllama for structured outputs
    llm = ChatOllama(model="llama3", temperature=0.1)
    
    # 4. Bind the Pydantic schema
    structured_llm = llm.with_structured_output(DemographicAdjustmentOutput)
    
    prompt = PromptTemplate.from_template(DEMOGRAPHIC_PROMPT)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "demographics": state["demographics"],
            "blinded_differential": state["blinded_differential"],
            "rag_context": state.get("rag_context", "No guidelines retrieved.")
        })
        
        # Convert the Pydantic list of objects back into dictionaries
        adjusted_differential = [
            {"condition": d.condition, "likelihood": d.likelihood, "justification": d.justification} 
            for d in result.adjusted_differential
        ]
        
        print("[Agent] Demographic Agent parsed successfully via Pydantic.")
        return {
            "chain_of_thought": result.chain_of_thought,
            "adjusted_differential": adjusted_differential,
            "adjustment_rationale": result.adjustment_rationale
        }
        
    except Exception as e:
        print(f"[Error] Demographic Agent Failed: {e}")
        return {
            "chain_of_thought": "Failed.",
            "adjusted_differential": [],
            "adjustment_rationale": f"System Error: {e}"
        }