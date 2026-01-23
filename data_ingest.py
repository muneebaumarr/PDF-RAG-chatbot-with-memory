import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# Load & split PDF
# -------------------------
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    documents = splitter.split_documents(pages)
    return documents

# -------------------------
# Create embeddings safely
# -------------------------
def get_embeddings():
    token = st.secrets.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in Streamlit secrets!")
    
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        huggingfacehub_api_token=token,
        model_kwargs={"device": "cpu"}  # Cloud safe
    )

# -------------------------
# Ingest PDF
# -------------------------
def ingest_pdf(pdf_path):
    documents = load_and_split_pdf(pdf_path)
    embeddings = get_embeddings()       # <- call embeddings here
    vectordb = Chroma.from_documents(documents, embedding=embeddings)
    return vectordb
