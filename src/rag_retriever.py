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

def retrieve_clinical_guidelines(text: str) -> str:
    print("\n[RAG] Querying ChromaDB Vector Store...")
    try:
        vectorstore = get_vector_store()
        
        # CHANGED: Get scores alongside the results (lower score = closer match)
        results_with_scores = vectorstore.similarity_search_with_score(text, k=2)
        
        valid_guidelines = []
        for doc, score in results_with_scores:
            # ONLY include the guideline if the distance score is strong (e.g., under 1.2)
            # You can tweak this threshold number based on testing!
            if score < 1.2: 
                valid_guidelines.append(doc.page_content)
        
        if not valid_guidelines:
            return "No specific RAG guidelines found for these exact symptoms."
        
        context = "\n".join(valid_guidelines)
        context += "\n(AI INSTRUCTION: Only apply these guidelines if they explicitly match the patient's symptoms.)"
        
        return context
        
    except Exception as e:
        print(f"[Error] RAG Retrieval Failed: {e}")
        return "RAG System offline. Rely on base training data."