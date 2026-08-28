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

### ⚙️ Pipeline Architecture (LangGraph StateGraph)
The system operates as a directed graph passing a stateful `PatientState` payload through six primary nodes:

1. **Security Bouncer** — Screens raw clinical notes for prompt injection attacks or jailbreak attempts, blocking malicious payloads before inference.
2. **Redaction Pipeline** — Extracts demographic data and sanitizes the clinical note using layered Regex and dictionary-based Named Entity Recognition (NER) to ensure the first diagnostic pass is entirely blinded.
3. **Semantic Router (RAG)** — Evaluates the blinded symptoms using a zero-shot LLM classifier to retrieve hardcoded clinical safety rules (e.g., forcing physiological exclusion before psychiatric diagnosis).
4. **Blinded Hypothesis Agent** — Generates a symptom-based differential diagnosis with zero demographic awareness.
5. **Demographic Agent (The Epidemiologist)** — Adjusts the blinded differential using the held-back demographics and RAG guidelines.
6. **Arbiter Agent (Safety Referee)** — Audits the reasoning delta between the two diagnostic agents. It outputs either an `EPIDEMIOLOGICALLY_JUSTIFIED` pass or a `POSSIBLE_ANCHORING` safety flag.

### 👨‍⚕️ Human-in-the-Loop (HITL) Dashboard
The backend is wrapped in a Streamlit interface that displays the AI's Chain of Thought (CoT), retrieved RAG guidelines, and the Arbiter's bias report. Clinicians must log a final decision (Approve, Reject, or Flag), which is persistently written to a local SQLite database for AI governance auditing.

---

## 🛠️ Tech Stack
* **Orchestration:** LangChain, LangGraph State Management
* **LLM Engine:** Ollama (Llama 3 — local quantized open-weight model)
* **Interface:** Streamlit
* **Database:** SQLite3
* **Data Processing:** Python `re`, JSON Parsing, Hugging Face `datasets`

---

## 📂 Repository Structure
```text
├── app.py                       # Main Streamlit dashboard and pipeline execution
├── data_fetch.py                # Script to pull MedQA-USMLE dataset for evaluation
├── requirements.txt             # Python dependencies
├── clinical_logs_v3.db          # Auto-generated SQLite database for audit logging
├── src/
│   ├── state.py                 # LangGraph PatientState TypedDict definition
│   ├── database.py              # SQLite initialization and CRUD operations
│   ├── rag_retriever.py         # Zero-shot semantic router and knowledge base
│   ├── redactor.py              # Regex and NER de-identification logic
│   └── agents/
│       ├── security_agent.py    # Prompt injection detection node
│       ├── hypothesis_agent.py  # Blinded diagnostic node
│       ├── demographic_agent.py # Epidemiological adjustment node
│       └── arbiter_agent.py     # Bias detection and safety audit node
```

---

## 🚀 Getting Started (Local Native Setup)

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running natively on your host machine.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/clinical_debiased_ai.git
   cd clinical_debiased_ai
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Pull the required Llama 3 model into your local Ollama instance:
   ```bash
   ollama pull llama3
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Evaluation & Demo Testing

**1. Testing Baseline Behavior**
Enter standard demographics (e.g., 45-year-old male) and a generic symptom (e.g., "headache"). The Arbiter will pass the interaction as `EPIDEMIOLOGICALLY_JUSTIFIED` since demographics do not alter the core diagnosis.

**2. Testing Bias Detection**
To trigger the system's safety flags, input the following vignette designed to test for psychiatric dismissal:

* **Age:** 26 | **Sex:** Female
* **Vignette:** *The patient presents to the ER with tachycardia, severe dyspnea, and tingling in her extremities. She reports feeling a profound sense of panic and impending doom.*

The Arbiter will actively flag the system if the LLM attempts to dismiss the physiological symptoms as a panic attack without first ruling out a Pulmonary Embolism based on the RAG rules.

**3. Generating Test Data**
Run the included data fetcher to download sample cases from the MedQA-USMLE dataset:

```bash
python data_fetch.py
```

---

## 🚧 Limitations & Future Work

As a proof-of-concept, this system intentionally prioritizes local execution and transparent reasoning over enterprise scalability. Future iterations will focus on:

* **Advanced NER Redaction** — Upgrading from dictionary-based Regex to Microsoft Presidio or spaCy/scispaCy for robust, context-aware data de-identification.
* **Vector Database Integration** — Migrating the semantic RAG router's hardcoded dictionary to a dedicated local vector store (e.g., ChromaDB) to scale the clinical knowledge base.
* **Structured Outputs** — Transitioning from Regex-based JSON extraction to LangChain's native Pydantic parsers to enforce strict schema compliance at the LLM generation layer.
* **Asynchronous Orchestration** — Refactoring independent LangGraph nodes to execute concurrently to reduce overall pipeline latency.

---

