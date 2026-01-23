from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def ingest_pdf(pdf_path: str):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80
    )
    docs = splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        huggingfacehub_api_token=os.environ["token"]
    )

    # 4. Store in vector DB
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="db"
    )


    return vectordb
