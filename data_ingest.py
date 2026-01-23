import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------------------------
# HuggingFace Embeddings
# -------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        huggingfacehub_api_token=st.secrets["HF_TOKEN"],
        model_kwargs={"device": "cpu"}  # Use CPU in Streamlit Cloud
    )

# -------------------------
# Load and Split PDF
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
# Ingest PDF and create vector DB
# -------------------------
def ingest_pdf(pdf_path):
    documents = load_and_split_pdf(pdf_path)
    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(documents, embedding=embeddings)
    return vectordb





