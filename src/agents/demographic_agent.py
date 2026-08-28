import json
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

DEMOGRAPHIC_PROMPT = """You are an expert Clinical Epidemiologist.
Review the Blinded Differential Diagnosis, the Patient Demographics, and the Retrieved Clinical Guidelines (RAG Context).

Your task is to adjust the differential diagnosis safely. 
You MUST adhere to the RAG Context provided. If the RAG context forbids a psychiatric diagnosis, you must not suggest one.

Patient Demographics: {demographics}
Blinded Differential: {blinded_differential}
Retrieved Guidelines (RAG): {rag_context}

You MUST return your response as a valid JSON object with exactly three keys:
1. "chain_of_thought": A detailed string explaining step-by-step how you evaluated the physical symptoms against the RAG guidelines BEFORE looking at the demographics.
2. "adjusted_differential": An array of objects, each with "condition", "likelihood", and "justification".
3. "adjustment_rationale": A brief summary of why you finalized this list.
"""

def run_demographic_agent(state):
    llm = OllamaLLM(model="llama3", temperature=0.1, format="json")
    prompt = PromptTemplate.from_template(DEMOGRAPHIC_PROMPT)
    chain = prompt | llm
    
    max_retries = 3  # The AI gets 3 chances to get the JSON right
    
    for attempt in range(max_retries):
        response = chain.invoke({
            "demographics": state["demographics"],
            "blinded_differential": state["blinded_differential"],
            "rag_context": state.get("rag_context", "No guidelines retrieved.")
        })
        
        try:
            parsed = json.loads(response)
            return {
                "adjusted_differential": parsed.get("adjusted_differential", []),
                "adjustment_rationale": parsed.get("adjustment_rationale", "No rationale."),
                "chain_of_thought": parsed.get("chain_of_thought", "No CoT provided.")
            }
        except json.JSONDecodeError:
            print(f"JSON Error on attempt {attempt + 1}. Forcing AI to retry...")
            # If it fails, the loop simply runs again!
            
    # If it fails 3 times in a row, THEN we fail safely.
    return {"adjusted_differential": [], "adjustment_rationale": "System Error: AI failed to format output after 3 attempts.", "chain_of_thought": "Failed."}