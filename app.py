import streamlit as st
from operator import itemgetter

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from data_ingest import ingest_pdf
import tempfile

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄 PDF RAG Chatbot with Memory")
st.write(
    "Upload your PDF and ask questions. The bot will answer using the content, "
    "but can also infer logically if something is not explicitly in the PDF."
)

# ------------------------
# Initialize LLM
# ------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=st.secrets["GROK_API_KEY"]
)

# ------------------------
# Session memory
# ------------------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ------------------------
# PDF Upload
# ------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
vectordb = None
retriever = None
chatbot = None

if uploaded_file:
    # Use temporary file to avoid Cloud file issues
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Ingest PDF
    vectordb = ingest_pdf(pdf_path)

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    st.success("✅ PDF processed successfully!")

# ------------------------
# Message trimming and prompt
# ------------------------
trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=llm,
    include_system=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an intelligent resume assistant.

            Rules:
            1. Use the provided context FIRST.
            2. If the answer is not explicitly in the context, infer logically.
            3. If information is missing, say so clearly.
            4. You may use general knowledge to explain or summarize.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ]
)

# ------------------------
# Memory + RAG chain
# ------------------------
if uploaded_file and retriever:
    chain = (
        RunnablePassthrough.assign(
            messages=itemgetter("messages") | trimmer,
            context=itemgetter("question") | retriever
        )
        | prompt
        | llm
    )

    chatbot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages"
    )

# ------------------------
# Chat Interface
# ------------------------
if uploaded_file and chatbot:
    user_input = st.chat_input("Ask something from the PDF...")

    if user_input:
        response = chatbot.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "question": user_input
            },
            config={"configurable": {"session_id": "pdf_chat"}}
        )

        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response.content)
