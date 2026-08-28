import streamlit as st
import pandas as pd
import time
from src.state import PatientState
from src.agents.security_agent import run_security_check
from src.agents.hypothesis_agent import run_hypothesis_agent
from src.agents.demographic_agent import run_demographic_agent
from src.agents.arbiter_agent import run_arbiter_agent
from src.redactor import layer1_regex_redactor, layer2_ner_redactor
from src.database import init_db, log_decision, fetch_all_logs
from src.rag_retriever import retrieve_clinical_guidelines

# Initialize the database on startup
init_db()

st.set_page_config(page_title="Debiased Clinical AI", page_icon="🩺", layout="wide")

# --- SESSION STATE MANAGEMENT ---
if "patient_state" not in st.session_state:
    st.session_state.patient_state = None
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "latency" not in st.session_state:
    st.session_state.latency = 0.0

st.title("🩺 Debiased Clinical Diagnostic Assistant")
st.markdown("Enterprise Multi-Agent Architecture with HITL Auditing & SQLite Persistence.")

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["📋 Clinical Pipeline", "📊 Audit & Compliance Dashboard"])

# --- SIDEBAR: PATIENT INPUT ---
with st.sidebar:
    st.header("Patient Data")
    age = st.text_input("Age", value="48")
    sex = st.selectbox("Sex", ["Female", "Male", "Other"])
    raw_text = st.text_area("Clinical Vignette", height=200, value="A 45-year-old male presents with a headache.")
    
    if st.button("Run Diagnostic Pipeline", type="primary"):
        
        # --- NEW LLM SECURITY BOUNCER ---
        if len(raw_text.split()) > 1000:
            st.sidebar.error("Error: Note is too long. Max 1000 words.")
            st.stop()
            
        with st.spinner("Security scan in progress..."):
            security_result = run_security_check(raw_text)
            if not security_result.get("is_safe", False):
                st.sidebar.error(f"🚨 SECURITY ALERT: Request blocked.\nReason: {security_result.get('reason', 'Malicious intent detected.')}")
                st.stop()
        # ---------------------------------

        with st.spinner("Pipeline running... Auditing AI logic..."):
            start_time = time.time()
            
            # 1. RAG Retrieval & Redaction
            rag_context = retrieve_clinical_guidelines(raw_text)
            step_1_text = layer1_regex_redactor(raw_text)
            redacted_note = layer2_ner_redactor(step_1_text)
            
            # 2. Init State
            state = PatientState(
                raw_text=raw_text, 
                redacted_text=redacted_note, 
                demographics={"age": age, "sex": sex.lower()},
                rag_context=rag_context, 
                chain_of_thought="", 
                blinded_differential=[], 
                adjusted_differential=[], 
                adjustment_rationale="", 
                arbiter_verdict="", 
                arbiter_analysis=""
            )
            
            # 3. Run Agents
            state.update(run_hypothesis_agent(state))
            state.update(run_demographic_agent(state))
            state.update(run_arbiter_agent(state))
            
            end_time = time.time()
            
            st.session_state.latency = round(end_time - start_time, 2)
            st.session_state.patient_state = state
            st.session_state.pipeline_run = True

# --- TAB 1: CLINICAL PIPELINE ---
with tab1:
    if st.session_state.pipeline_run and st.session_state.patient_state:
        state = st.session_state.patient_state
        
        st.caption(f"⏱️ System executed in {st.session_state.latency} seconds.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Blinded Hypothesis")
            for d in state["blinded_differential"]:
                condition = d.get("condition", "Unknown Condition")
                likelihood = d.get("likelihood", "Unknown")
                justification = d.get("justification", "No justification provided by AI.")
                st.info(f"**{condition}** ({likelihood})\n\n{justification}")
                
        with col2:
            st.subheader("2. Adjusted Hypothesis")
            for d in state["adjusted_differential"]:
                condition = d.get("condition", "Unknown Condition")
                likelihood = d.get("likelihood", "Unknown")
                justification = d.get("justification", "No justification provided by AI.")
                st.warning(f"**{condition}** ({likelihood})\n\n{justification}")
                
        st.divider()

        # --- RAG & CoT UI EXPANDER ---
        st.subheader("🧠 Inside the AI's Brain (RAG & CoT)")
        with st.expander("View Ground-Truth Guidelines & AI Chain of Thought", expanded=True):
            st.markdown(f"**📚 Retrieved RAG Guidelines:**\n\n{state.get('rag_context', 'None')}")
            st.markdown(f"**🔗 AI Chain of Thought:**\n\n{state.get('chain_of_thought', 'No CoT generated.')}")
        
        st.divider()
        
        # --- ARBITER UI ---
        st.subheader("🛡️ Stage 4: Arbiter Audit")
        if "FLAGGED" in state["arbiter_verdict"] or "JUSTIFIED" not in state["arbiter_verdict"]:
            st.error(f"🚨 **RED FLAG DETECTED: {state['arbiter_verdict']}**")
            st.warning("⚠️ **CLINICAL WARNING:** The Arbiter detected potential demographic bias. Do not trust the AI.")
            with st.expander("Read the Arbiter's Bias Report", expanded=True):
                st.write(f"**AI Rationale:** {state['adjustment_rationale']}")
                st.write(f"**Arbiter Analysis:** {state['arbiter_analysis']}")
        else:
            st.success(f"✅ **Verdict:** {state['arbiter_verdict']}")
            with st.expander("Read the AI Rationale & Audit"):
                st.write(f"**AI Rationale:** {state['adjustment_rationale']}")
                st.write(f"**Arbiter Analysis:** {state['arbiter_analysis']}")
            
        st.divider()
        
        # --- HITL DECISION LOGGING ---
        st.subheader("👨‍⚕️ Stage 5: Final Decision")
        decision = st.radio("Final Decision:", ["Pending Review...", "Approve & Add to Chart", "Reject - Rely on Blinded List", "Flag for Manual Overwrite"], horizontal=True)
        
        if st.button("Submit to Database"):
            if decision == "Pending Review...":
                st.error("Please make a decision before submitting.")
            else:
                blinded_list = ", ".join([d.get("condition", "Unknown") for d in state["blinded_differential"]])
                adjusted_list = ", ".join([d.get("condition", "Unknown") for d in state["adjusted_differential"]])
                
                try:
                    log_decision(
                        age=state["demographics"]["age"],
                        sex=state["demographics"]["sex"],
                        blinded=blinded_list,
                        adjusted=adjusted_list,
                        verdict=state["arbiter_verdict"],
                        decision=decision,
                        latency=st.session_state.latency
                    )
                    st.success("✅ Decision securely logged to the SQLite database!")
                except Exception as e:
                    st.error(f"Database error (you may need to delete your old clinical_logs_v3.db file): {e}")

    else:
        st.info("👈 Enter patient data in the sidebar and click 'Run Diagnostic Pipeline' to begin.")

# --- TAB 2: AUDIT DASHBOARD ---
with tab2:
    st.header("Hospital AI Governance Dashboard")
    try:
        logs = fetch_all_logs()
        if logs:
            df = pd.DataFrame(logs, columns=["ID", "Timestamp", "Age", "Sex", "Blinded Diagnoses", "Adjusted Diagnoses", "Arbiter Verdict", "Human Decision", "Latency (s)"])
            
            total_runs = len(df)
            rejections = len(df[df["Human Decision"].isin(["Reject - Rely on Blinded List", "Flag for Manual Overwrite"])])
            red_flags = len(df[df["Arbiter Verdict"].str.contains("FLAGGED", na=False)])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total AI Consults", total_runs)
            m2.metric("Human Rejections", rejections)
            m3.metric("Safety Flags Triggered", red_flags)
            
            st.divider()
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("The database is currently empty.")
    except Exception as e:
        st.error(f"Could not load database. Try deleting `clinical_logs.db` and running a new patient. Error: {e}")