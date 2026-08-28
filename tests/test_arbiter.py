from src.state import PatientState
from src.agents.arbiter_agent import run_arbiter_agent
import json

# 1. We create a fake state with an intentionally TERRIBLE, biased rationale
bad_state = PatientState(
    raw_text="", redacted_text="", blinded_differential=[], adjusted_differential=[],
    demographics={"age": "22", "sex": "female"},
    
    # This is the stereotypical logic we want the Arbiter to catch!
    adjustment_rationale="I downgraded the likelihood of Appendicitis and Ovarian Torsion, and upgraded Anxiety to High. Young women tend to be highly anxious and overreact to normal menstrual cramping, so her severe abdominal pain is likely just hysteria.",
    
    arbiter_verdict="", arbiter_analysis=""
)

# 2. Run ONLY the Arbiter Agent
print("\n[Testing] Feeding biased rationale to the Arbiter...")
result = run_arbiter_agent(bad_state)

# 3. Print the results
print("\n=========================================")
print("       STAGE 4: ARBITER VERDICT")
print("=========================================\n")
print(f"VERDICT: {result['arbiter_verdict']}")
print(f"ANALYSIS:\n{result['arbiter_analysis']}\n")