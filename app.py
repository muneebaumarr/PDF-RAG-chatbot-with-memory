import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from data_ingest import ingest_pdf




load_dotenv()


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["token"] = os.getenv("token")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key = os.getenv("GROK_API_KEY")
)

# Streamlit UI

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄 PDF RAG Chatbot with Memory")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


#session mEmor 

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


#process PDF

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    vectordb = ingest_pdf("temp.pdf")

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    st.success("PDF processed successfully!")

#prompt + trimming 

trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=llm,
    include_system=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        ( "system",
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


# Memory Chain

if uploaded_file:
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


#chat interface 

if uploaded_file:
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
