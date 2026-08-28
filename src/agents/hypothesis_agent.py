import json
import re
from langchain_ollama import OllamaLLM
from src.state import PatientState

HYPOTHESIS_PROMPT = """You are an objective diagnostic AI. 
Analyze the clinical note containing ONLY symptoms and findings.
Generate a differential diagnosis of 3 conditions purely based on the findings.
You must NOT infer demographic information.

Clinical Note:
{redacted_text}

You MUST return your response as a valid JSON array of objects.
Example format:
[
  {{
    "condition": "Disease Name",
    "likelihood": "High",
    "justification": "Reasoning here..."
  }}
]
"""

def run_hypothesis_agent(state: PatientState) -> dict:
    print("\n[Agent] Blinded Hypothesis Agent (LLM) is thinking...")
    
    formatted_prompt = HYPOTHESIS_PROMPT.format(redacted_text=state["redacted_text"])
    
    # 1. Force strict JSON format and lower temperature for more robotic/predictable output
    llm = OllamaLLM(model="llama3", temperature=0.1, format="json")
    try:
        raw_response = llm.invoke(formatted_prompt)
        print(f"\n[Debug] Raw LLM Output:\n{raw_response}\n")
        
        # 2. Smarter Parsing: Hunt for the exact JSON array brackets [ ]
        match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if match:
            clean_response = match.group(0)
            blinded_differential = json.loads(clean_response)
            print("[Agent] LLM differential parsed successfully.")
            return {"blinded_differential": blinded_differential}
        else:
            print("[Error] No JSON array found in output.")
            return {"blinded_differential": []}
            
    except json.JSONDecodeError as e:
        print(f"[Error] JSON Decode Failed: {e}")
        return {"blinded_differential": []}
    except Exception as e:
        print(f"[Error] Connection/Execution Failed: {e}")
        return {"blinded_differential": []}