from typing import TypedDict, List, Dict


class DiagnosisItem(TypedDict):
    condition: str
    likelihood: str
    justification: str

class PatientState(TypedDict):
    raw_text: str
    redacted_text: str
    demographics: Dict[str, str]
    rag_context: str              
    chain_of_thought: str         
    blinded_differential: List[DiagnosisItem]
    adjusted_differential: List[DiagnosisItem]
    adjustment_rationale: str
    arbiter_verdict: str
    arbiter_analysis: str          