from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from src.state import PatientState
from typing import List, Literal

# 1. Define the exact JSON schema we want the LLM to output
class Diagnosis(BaseModel):
    condition: str = Field(description="The name of the medical condition")
    
    # 2. CHANGED: Use Literal to mathematically restrict the output choices
    likelihood: Literal["High", "Medium", "Low"] = Field(
        description="You MUST select exactly one of these words: High, Medium, or Low."
    )
    
    justification: str = Field(description="Brief reasoning based ONLY on the symptoms")

class DifferentialDiagnosis(BaseModel):
    # CHANGED: Added strict mathematical bounds to force exactly 3 items
    diagnoses: List[Diagnosis] = Field(
        min_length=3,
        max_length=3,
        description="A list of exactly 3 potential conditions"
    )

HYPOTHESIS_PROMPT = """You are an objective diagnostic AI. 
Analyze the clinical note containing ONLY symptoms and findings.
Generate a differential diagnosis of 3 conditions purely based on the findings.
You must NOT infer demographic information.

Clinical Note:
{redacted_text}
"""

def run_hypothesis_agent(state: PatientState) -> dict:
    print("\n[Agent] Blinded Hypothesis Agent (LLM) is thinking...")
    
    # 2. Use ChatOllama instead of OllamaLLM to support structured outputs
    llm = ChatOllama(model="llama3", temperature=0.1)
    
    # 3. Bind the Pydantic schema to the LLM
    structured_llm = llm.with_structured_output(DifferentialDiagnosis)
    
    prompt = PromptTemplate.from_template(HYPOTHESIS_PROMPT)
    chain = prompt | structured_llm
    
    try:
        # 4. Invoke the chain. It automatically returns a validated Python object!
        result = chain.invoke({"redacted_text": state["redacted_text"]})
        
        # Convert the Pydantic object back into the dictionary format your Streamlit app expects
        blinded_differential = [
            {"condition": d.condition, "likelihood": d.likelihood, "justification": d.justification} 
            for d in result.diagnoses
        ]
        
        print("[Agent] LLM differential parsed successfully via Pydantic.")
        return {"blinded_differential": blinded_differential}
        
    except Exception as e:
        print(f"[Error] Hypothesis Agent Failed: {e}")
        return {"blinded_differential": []}
