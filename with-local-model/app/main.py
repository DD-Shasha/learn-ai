# app/main.py
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import torch

# Load models (will download on first run)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device=0 if torch.cuda.is_available() else -1)
qa_pipeline = pipeline("text-generation", model="google/flan-t5-small", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device=0 if torch.cuda.is_available() else -1)

st.set_page_config(page_title="RAG Chat (Open Source)", layout="wide")
st.title("🧠 Chat with TXT Files (Offline Mode)")

uploaded_files = st.file_uploader("📄 Upload `.txt` files", type="txt", accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        all_text += file.read().decode("utf-8") + "\n"

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(all_text)

    # Prepare documents for FAISS
    docs = [Document(page_content=chunk) for chunk in chunks]
    texts = [doc.page_content for doc in docs]
    # embeddings = embedder.encode(texts, convert_to_tensor=False)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    # Build FAISS
    # vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)

    # Chat UI
    user_query = st.text_input("Ask something:")

    if user_query:
        # Embed query and retrieve relevant chunks
        query_vec = embedder.encode([user_query])[0]
        docs_and_scores = vectorstore.similarity_search_by_vector(query_vec, k=3)
        context = "\n\n".join([doc.page_content for doc in docs_and_scores])

        prompt = f"""You are a helpful assistant. Answer the question based only on the context below.

Context:
{context}

Question:
{user_query}

Answer:"""

        # Generate answer
        with st.spinner("🧠 Generating..."):
            output = qa_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
            st.success(output[0]['generated_text'].split("Answer:")[-1].strip())
