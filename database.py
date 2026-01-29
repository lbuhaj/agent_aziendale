import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./db"
UPLOAD_DIR = "./data/cv_uploads"

# Inizializziamo il modello di embedding locale (gratuito)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_db():
    return Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_model
    )

def ingest_cvs():
    """Legge i PDF e li indicizza in ChromaDB"""
    documents = []
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)

    for file in os.listdir(UPLOAD_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(UPLOAD_DIR, file))
            documents.extend(loader.load())
    
    if not documents: return "Nessun PDF trovato."

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    return f"Successo: {len(documents)} PDF indicizzati."