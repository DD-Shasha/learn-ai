# app/main.py
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("💬 Chat with Your TXT Files")

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not set. Please set it in your .env file or environment.")
    st.stop()

uploaded_files = st.file_uploader("📄 Upload one or more .txt files", type="txt", accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        all_text += file.read().decode("utf-8") + "\n"

    # Split and embed text
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(all_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), retriever=retriever)

    query = st.text_input("❓ Ask something about the uploaded text files:")

    if query:
        with st.spinner("🤖 Thinking..."):
            answer = qa_chain.run(query)
            st.success(answer)
else:
    st.info("👆 Upload `.txt` files to start chatting.")
