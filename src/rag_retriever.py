import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

KNOWLEDGE_BASE_DOCS = [
    Document(
        page_content="In young females presenting with tachycardia, dyspnea, and tingling, strictly rule out Pulmonary Embolism and Arrhythmia. Do not dismiss as anxiety, panic attack, or PTSD without physiological exclusion.",
        metadata={"category": "PULMONARY_EMBOLISM"}
    ),
    Document(
        page_content="Women with Acute Coronary Syndrome (ACS) often present atypically with jaw pain, shoulder pain, and profound fatigue rather than classic crushing chest pain. Maintain high suspicion for Myocardial Infarction.",
        metadata={"category": "ATYPICAL_ACS"}
    ),
    Document(
        page_content="Tearing chest pain radiating to the back with unequal blood pressures is a hallmark of Aortic Dissection. This supersedes demographic likelihoods of standard ACS.",
        metadata={"category": "AORTIC_DISSECTION"}
    )
]

# Set up local persistence directory so the database saves to your hard drive
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


embeddings = OllamaEmbeddings(model="nomic-embed-text")

def get_vector_store():
    """Initializes or loads the ChromaDB vector database."""
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    print("\n[RAG] Building ChromaDB Vector Store for the first time...")
    return Chroma.from_documents(
        documents=KNOWLEDGE_BASE_DOCS,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

def retrieve_clinical_guidelines(vignette_text: str) -> str:
    """
    Uses ChromaDB similarity search to retrieve the most relevant medical guideline 
    based on the mathematical embedding of the symptoms.
    """
    baseline = "BASELINE RULE: Always prioritize life-threatening physiological conditions before considering psychiatric diagnoses.\n\n"
    
    try:
        vector_store = get_vector_store()
        print("\n[RAG] Querying ChromaDB Vector Store...")
        
        # Retrieve the single most relevant guideline mathematically (k=1)
        results = vector_store.similarity_search_with_score(vignette_text, k=1)
        
        if results:
            best_doc, distance = results[0]
            category = best_doc.metadata.get("category", "UNKNOWN")
            
            # We append a safety prompt to ensure the LLM only applies it if relevant
            return baseline + f"👉 POTENTIAL RAG MATCH (ChromaDB - {category}):\n{best_doc.page_content}\n\n(AI INSTRUCTION: If this guideline does not directly apply to the patient's physical symptoms, ignore it.)"
        
        return baseline + "No specific rules retrieved. Proceed with standard medical knowledge."
            
    except Exception as e:
        print(f"[Error] ChromaDB Retrieval Failed: {e}")
        return baseline + "No specific rules retrieved. Proceed with standard medical knowledge."
