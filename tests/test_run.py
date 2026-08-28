from src.state import PatientState
from src.agents.hypothesis_agent import run_hypothesis_agent
from src.agents.demographic_agent import run_demographic_agent
from src.agents.arbiter_agent import run_arbiter_agent

# 1. Create the initial clipboard (Simulating Stage 1 Output)
test_state = PatientState(
    raw_text="A 36-year-old man...",
    redacted_text="A [AGE] [SEX] is brought to the emergency department by their spouse 20 minutes after having a seizure. Over the past 3 days, the patient has had a fever and worsening headaches.",
    demographics={"age": "36", "sex": "male"},
    blinded_differential=[],
    adjusted_differential=[],
    adjustment_rationale="",
    arbiter_verdict="",
    arbiter_analysis=""
)

# 2. Run Stage 2: Hypothesis Agent
state_update_1 = run_hypothesis_agent(test_state)
test_state.update(state_update_1) # Update clipboard

# 3. Run Stage 3: Demographic Agent
# This agent gets to see the blinded differential AND the demographics
state_update_2 = run_demographic_agent(test_state)
test_state.update(state_update_2) 



# 4. Print the final results
print("\n=========================================")
print("   STAGE 3: DEMOGRAPHIC ADJUSTMENTS")
print("=========================================\n")
print(f"RATIONALE PROVIDED BY AI:\n{test_state['adjustment_rationale']}\n")
state_update_3=run_arbiter_agent(test_state)
test_state.update(state_update_3)

print("\n=========================================")
print("       STAGE 4: ARBITER VERDICT")
print("=========================================\n")
print(f"VERDICT: {test_state['arbiter_verdict']}")
print(f"ANALYSIS: {test_state['arbiter_analysis']}\n")

print("ADJUSTED DIFFERENTIAL:")
for diagnosis in test_state["adjusted_differential"]:
    print(f"\nCondition: {diagnosis['condition']}")
    print(f"Likelihood: {diagnosis['likelihood']}")
    print(f"Justification: {diagnosis['justification']}")