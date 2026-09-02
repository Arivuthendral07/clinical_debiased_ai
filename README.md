# 🩺 Debiased Multi-Agent Clinical Reasoning System

A stateful, multi-agent AI pipeline designed to prevent premature diagnostic anchoring in clinical large language models (LLMs). This system physically separates symptom-based reasoning from demographic reasoning, employing a specialized "Arbiter Agent" to flag inappropriate diagnostic shifts before they reach a human clinician.

---

## 🎥 Live Demo

https://github.com/user-attachments/assets/9e7584ff-407c-42c1-8123-aa05f9048c6d

---

## ⚠️ The Clinical Problem
Standard LLMs used for clinical diagnosis frequently suffer from demographic anchoring. When a model processes patient demographics (e.g., "75-year-old male") alongside clinical symptoms, it tends to jump straight to statistically common diagnoses for that demographic. While sometimes correct, this bias can cause the AI to dismiss atypical but dangerous presentations (e.g., downplaying acute chest pain in a younger woman as "anxiety").

## 💡 The Multi-Agent Solution
This proof-of-concept architecture builds a defensive wall against diagnostic anchoring. Rather than blindly flagging any difference between symptom-only and demographically-adjusted lists as bias, the system uses an Arbiter Agent to distinguish legitimate epidemiological adjustments from dangerous stereotyping.


⚙️ Pipeline Architecture (LangGraph StateGraph)

The system operates as a directed graph passing a stateful PatientState payload through six primary nodes:

Security Bouncer — Screens raw clinical notes for prompt injection attacks or jailbreak attempts, blocking malicious payloads before inference.
Redaction Pipeline — Extracts demographic data and sanitizes the clinical note using layered Regex and dictionary-based Named Entity Recognition (NER) to ensure the first diagnostic pass is entirely blinded.
Blinded RAG Retrieval (ChromaDB) — Queries a local ChromaDB vector store using strictly anonymized symptoms, applying a mathematical distance threshold so the vector search isn't skewed by patient demographics or polluted by low-relevance guidelines.
Blinded Hypothesis Agent — Generates a symptom-based differential diagnosis with zero demographic awareness.
Demographic Agent (The Epidemiologist) — Adjusts the blinded differential using the held-back demographics and RAG guidelines.
Arbiter Agent (Safety Referee) — Audits the reasoning delta between the two diagnostic agents. It outputs either an EPIDEMIOLOGICALLY_JUSTIFIED pass or a POSSIBLE_ANCHORING safety flag.
👨‍⚕️ Human-in-the-Loop (HITL) Dashboard

The backend is wrapped in a Streamlit interface that displays the AI's Chain of Thought (CoT), retrieved RAG guidelines, and the Arbiter's bias report. Clinicians must log a final decision (Approve, Reject, or Flag), which is persistently written to a local SQLite database for AI governance auditing.

🛠️ Tech Stack
Orchestration: LangChain, LangGraph State Management
LLM Engine: Ollama (Llama 3 8B — local quantized open-weight model)
Vector Database: ChromaDB (nomic-embed-text embeddings)
Data Parsing & NLP: Pydantic (structured outputs, Literal typing), spaCy (en_core_web_sm)
Interface: Streamlit
Database: SQLite3
Data Processing: Python re, JSON Parsing, Hugging Face datasets
📂 Repository Structure
text
├── app.py                       # Main Streamlit dashboard and pipeline execution
├── data_fetch.py                # Script to pull MedQA-USMLE dataset for evaluation
├── requirements.txt             # Python dependencies
├── src/
│   ├── state.py                 # LangGraph PatientState TypedDict definition
│   ├── database.py              # SQLite initialization and CRUD operations
│   ├── rag_retriever.py         # ChromaDB retrieval + distance-threshold filtering
│   ├── redactor.py              # Regex and NER de-identification logic
│   └── agents/
│       ├── security_agent.py    # Prompt injection detection node
│       ├── hypothesis_agent.py  # Blinded diagnostic node
│       ├── demographic_agent.py # Epidemiological adjustment node
│       └── arbiter_agent.py     # Bias detection and safety audit node

Note: clinical_logs_v3.db is auto-generated at runtime by database.py and is not tracked in version control (see .gitignore).

🚀 Getting Started (Local Native Setup)
Prerequisites
Python 3.10+
Ollama installed and running natively on your host machine.
Installation
Clone the repository:
bash
   git clone https://github.com/yourusername/clinical_debiased_ai.git
   cd clinical_debiased_ai
Install the required Python dependencies:
bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
Pull the required models into your local Ollama instance:
bash
   ollama pull llama3
   ollama pull nomic-embed-text
Launch the Streamlit application:
bash
   streamlit run app.py
🧪 Evaluation & Demo Testing

1. Testing Baseline Behavior Enter standard demographics (e.g., 45-year-old male) and a generic symptom (e.g., "headache"). The Arbiter will pass the interaction as EPIDEMIOLOGICALLY_JUSTIFIED since demographics do not alter the core diagnosis.

2. Testing Bias Detection To trigger the system's safety flags, input the following vignette designed to test for psychiatric dismissal:

Age: 26 | Sex: Female
Vignette: The patient presents to the ER with tachycardia, severe dyspnea, and tingling in her extremities. She reports feeling a profound sense of panic and impending doom.

The Arbiter will actively flag the system if the LLM attempts to dismiss the physiological symptoms as a panic attack without first ruling out a Pulmonary Embolism based on the RAG rules.

3. Generating Test Data Run the included data fetcher to download sample cases from the MedQA-USMLE dataset:

bash
python data_fetch.py
🔧 Engineering Challenges & Solutions
RAG context over-anchoring: A headache query initially returned a heart attack diagnosis because an unrelated cardiology guideline bled into the retrieved context. Fix: switched to similarity_search_with_score and enforced a distance threshold (score < 1.2) to filter out low-relevance guidelines before the LLM sees them.
Non-deterministic array lengths: The local model occasionally ignored instructions to return exactly 3 differential diagnoses. Fix: enforced min_length=3 / max_length=3 in the Pydantic schema so LangChain validates and retries automatically on any structural mismatch.
Inconsistent probability formatting: The model sometimes returned numeric confidence scores instead of the requested categorical labels. Fix: used Literal["High", "Medium", "Low"] typing in Pydantic to constrain output to those three exact values.
🚧 Limitations & Future Work

As a proof-of-concept, this system intentionally prioritizes local execution and transparent reasoning over enterprise scalability. Future iterations will focus on:

Vector Retrieval Negation Blindspots — Dense embeddings (nomic-embed-text) struggle with semantic negation; a note stating "no chest pain" can still surface cardiac guidelines. Fix planned: hybrid search combining dense vectors with BM25 sparse keyword matching.
Advanced NER Redaction — Standard spaCy NER masks explicit identifiers but misses implicit demographic markers (e.g., "maternity ward"). Fix planned: migrate to domain-specific NLP models like Microsoft Presidio or SciSpaCy (en_core_sci_sm).
LLM Sycophancy — Using the same local model family to both generate and audit diagnoses risks echo-chamber validation. Fix planned: route the Arbiter Agent to a distinct, heterogeneous frontier model via API for independent peer review.
Asynchronous Orchestration — Refactor independent LangGraph nodes to execute concurrently and reduce overall pipeline latency.
