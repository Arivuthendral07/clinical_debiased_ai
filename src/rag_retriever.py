import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

#  Our Hardcoded Knowledge Base
KNOWLEDGE_BASE = {
    "PULMONARY_EMBOLISM": "In young females presenting with tachycardia, dyspnea, and tingling, strictly rule out Pulmonary Embolism and Arrhythmia. Do not dismiss as anxiety, panic attack, or PTSD without physiological exclusion.",
    "ATYPICAL_ACS": "Women with Acute Coronary Syndrome (ACS) often present atypically with jaw pain, shoulder pain, and profound fatigue rather than classic crushing chest pain. Maintain high suspicion for Myocardial Infarction.",
    "AORTIC_DISSECTION": "Tearing chest pain radiating to the back with unequal blood pressures is a hallmark of Aortic Dissection. This supersedes demographic likelihoods of standard ACS.",
    "NONE": "No specific rules retrieved. Proceed with standard medical knowledge."
}

ROUTER_PROMPT = """You are an expert Clinical Triage Router. 
Read the following Clinical Vignette and determine which specific medical guideline category applies best based on the symptoms described.

Available Categories:
1. "PULMONARY_EMBOLISM": Symptoms include fast heartbeat, shortness of breath, hyperventilation, or tingling.
2. "ATYPICAL_ACS": Symptoms include jaw pain, shoulder pain, nausea, or profound fatigue.
3. "AORTIC_DISSECTION": Symptoms include tearing or ripping chest pain radiating to the back.
4. "NONE": The vignette does not strongly match the above categories.

Clinical Vignette: {vignette}

You MUST return a valid JSON object with a single key "category" containing exactly one of the category names above.
Example: {{"category": "PULMONARY_EMBOLISM"}}
"""

def retrieve_clinical_guidelines(vignette_text: str) -> str:
    """
    Uses an LLM Semantic Router to understand the meaning of the symptoms 
    and fetch the correct medical guideline, bypassing the need for exact keywords.
    """
    baseline = "BASELINE RULE: Always prioritize life-threatening physiological conditions before considering psychiatric diagnoses.\n\n"
    
    try:
        # Temperature 0.0 makes the router purely analytical and deterministic
        llm = OllamaLLM(model="llama3", temperature=0.0, format="json")
        prompt = PromptTemplate.from_template(ROUTER_PROMPT)
        chain = prompt | llm
        
        print("\n[RAG] Semantic Router analyzing vignette...")
        raw_response = chain.invoke({"vignette": vignette_text})
        
        # Regex safety net to extract the JSON
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if match:
            parsed_data = json.loads(match.group(0))
            category = parsed_data.get("category", "NONE")
        else:
            category = "NONE"
            
        # Security check: If the AI hallucinates a category not in our dictionary, default to NONE
        if category not in KNOWLEDGE_BASE:
            category = "NONE"
            
        if category != "NONE":
            return baseline + f"👉 RAG MATCH (Semantic Router - {category}):\n" + KNOWLEDGE_BASE[category]
        else:
            return baseline + KNOWLEDGE_BASE["NONE"]
            
    except Exception as e:
        print(f"[Error] Semantic Router Failed: {e}")
        return baseline + KNOWLEDGE_BASE["NONE"]
