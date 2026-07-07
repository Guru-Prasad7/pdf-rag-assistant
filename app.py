import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import tempfile
import os

st.set_page_config(page_title="Chat with PDF - RAG Assistant", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions about it — powered by RAG (LangChain + FAISS + Groq)")

# API key input
api_key = st.text_input("Enter your Groq API Key", type="password")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load and split
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Embeddings + Vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)

        os.remove(tmp_path)

    st.success(f"PDF processed! {len(chunks)} chunks created. Ask me anything about it.")

    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Thinking..."):
            results = vector_store.similarity_search(question, k=2)
            context = "\n\n".join([doc.page_content for doc in results])

            prompt = f"""Answer the question based only on the context below. Be concise.

Context:
{context}

Question: {question}

Answer:"""

            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            st.write("### Answer")
            st.write(response.choices[0].message.content)

            with st.expander("View retrieved context"):
                st.write(context)
else:
    st.info("Enter your Groq API key and upload a PDF to get started.")
