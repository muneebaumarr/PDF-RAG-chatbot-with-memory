import streamlit as st
import tempfile

from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableWithMessageHistory
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from data_ingest import ingest_pdf


# ============================================================
# Streamlit Page Config
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄"
)

st.title("📄 PDF RAG Chatbot with Memory")

st.write(
    "Upload a PDF and ask questions about it. "
    "The chatbot remembers the conversation and can handle follow-up questions."
)


# ============================================================
# Initialize LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=st.secrets["GROK_API_KEY"],
    temperature=0
)


# ============================================================
# Conversation Memory
# ============================================================

if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:

    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    return st.session_state.store[session_id]


# ============================================================
# PDF Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


retriever = None


if uploaded_file:

    # --------------------------------------------------------
    # Save uploaded PDF temporarily
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name


    # --------------------------------------------------------
    # Process PDF
    # --------------------------------------------------------

    with st.spinner("Processing PDF..."):

        vectordb = ingest_pdf(pdf_path)

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3
            }
        )

    st.success("✅ PDF processed successfully!")


# ============================================================
# QUESTION CONTEXTUALIZATION
# ============================================================
#
# This chain converts follow-up questions into standalone
# questions before sending them to the retriever.
#
# Example:
#
# Previous:
# User: Who is the CEO?
# Bot: John Smith.
#
# Current:
# User: How old is he?
#
# Becomes:
# "How old is John Smith?"
#
# ============================================================

contextualize_prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are a question contextualization assistant.

Your job is to rewrite the user's latest question into a
standalone question that can be understood without the
conversation history.

Use the conversation history to resolve references such as:

- he
- she
- it
- they
- this
- that
- the previous section
- the person mentioned earlier

If the user's question is already standalone, return it
unchanged.

DO NOT answer the question.

Return ONLY the rewritten standalone question.
"""
        ),

        MessagesPlaceholder(
            variable_name="messages"
        ),

        (
            "human",
            "{question}"
        )

    ]
)


contextualize_chain = contextualize_prompt | llm


# ============================================================
# ANSWER PROMPT
# ============================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are an intelligent PDF assistant.

Your job is to answer the user's question using the
provided PDF context and conversation history.

Rules:

1. Use the PDF context as your primary source.

2. If the answer is explicitly available in the PDF,
   answer based on the PDF.

3. If the answer is not explicitly available in the PDF,
   say that it is not directly mentioned.

4. You may use general knowledge or logical reasoning
   when helpful, but clearly distinguish it from information
   found in the PDF.

5. Never claim that something is present in the PDF if it
   is not supported by the retrieved context.

6. Use the conversation history to understand follow-up
   questions.

7. Give clear and concise answers.

8. If the user asks for an explanation, explain the concept
   in simple language.
"""
        ),

        MessagesPlaceholder(
            variable_name="messages"
        ),

        (
            "human",
            """
PDF Context:

{context}

Current Question:

{question}
"""
        )

    ]
)


# ============================================================
# RAG CHAIN
# ============================================================

chatbot = None


if retriever:

    def retrieve_documents(inputs):

        # ----------------------------------------------------
        # Get conversation history
        # ----------------------------------------------------

        messages = inputs["messages"]

        # ----------------------------------------------------
        # Get current user question
        # ----------------------------------------------------

        question = inputs["question"]

        # ----------------------------------------------------
        # Rewrite question using conversation history
        # ----------------------------------------------------

        standalone_question = contextualize_chain.invoke(
            {
                "messages": messages,
                "question": question
            }
        )

        standalone_question = standalone_question.content

        # ----------------------------------------------------
        # Retrieve relevant PDF documents
        # ----------------------------------------------------

        documents = retriever.invoke(
            standalone_question
        )

        return documents


    # --------------------------------------------------------
    # Build the RAG chain
    # --------------------------------------------------------

    chain = (

        RunnablePassthrough.assign(
            context=retrieve_documents
        )

        | answer_prompt

        | llm

    )


    # --------------------------------------------------------
    # Add conversation memory
    # --------------------------------------------------------

    chatbot = RunnableWithMessageHistory(

        chain,

        get_session_history,

        input_messages_key="messages",

        history_messages_key="messages"

    )


# ============================================================
# CHAT INTERFACE
# ============================================================

if uploaded_file and chatbot:

    user_input = st.chat_input(
        "Ask something about the PDF..."
    )


    if user_input:

        # ----------------------------------------------------
        # Invoke chatbot
        # ----------------------------------------------------

        response = chatbot.invoke(

            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ],

                "question": user_input
            },

            config={
                "configurable": {
                    "session_id": "pdf_chat"
                }
            }

        )


        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        st.chat_message(
            "user"
        ).write(
            user_input
        )


        # ----------------------------------------------------
        # Display assistant response
        # ----------------------------------------------------

        st.chat_message(
            "assistant"
        ).write(
            response.content
        )

else:

    st.info(
        "👆 Upload a PDF to start chatting."
    )

