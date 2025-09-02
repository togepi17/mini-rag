# frontend.py
import streamlit as st
import requests
from typing import List
import pandas as pd

st.set_page_config(page_title="Mini-RAG App", page_icon="🤖", layout="wide")

st.title("Mini-RAG App")

st.markdown(
    """
    This app uses a **hybrid pipeline**:
    - **RAG**: Retrieves relevant documents with FAISS and answers questions.
    - **LLM-only fallback**: If no relevant docs, the LLM answers directly.
    - **File injection**: Upload PDFs or TXT files to add context for future queries.
    - **Metadata**: Shows metadata for each document/context.
    """
)

# -------------------------
# Section 1: File Upload / Injection
# -------------------------
st.subheader("1️⃣ Upload File to Inject Context")
uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file", type=["pdf", "txt"], accept_multiple_files=False
)

if uploaded_file:
    if st.button("Inject File"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.read())}
            response = requests.post("http://localhost:8000/upload", files=files)
            if response.status_code == 200:
                st.success(f"File '{uploaded_file.name}' injected successfully!")

                # Show metadata returned by backend
                metadata = response.json().get("metadata", {})
                if metadata:
                    st.subheader("🗂️ File Metadata")
                    for key, value in metadata.items():
                        st.markdown(f"**{key}:** {value}")
            else:
                st.error(f"Error injecting file: {response.json()}")
        except Exception as e:
            st.error(f"Error injecting file: {e}")

# -------------------------
# Section 2: Ask a Question
# -------------------------
st.subheader("2️⃣ Ask a Question")
user_input = st.text_input("Enter your question:")

if st.button("Ask with RAG"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        try:
            payload = {"question": user_input, "top_k": 3}
            response = requests.post("http://localhost:8000/process", json=payload)
            if response.status_code == 200:
                result = response.json()
                contexts: List[dict] = result.get("contexts", [])
                answer: str = result.get("answer", "")

                # Display contexts
                if contexts:
                    st.subheader("📄 Retrieved Contexts with Metadata")
                    for c in contexts:
                        st.markdown(f"**[{c.get('index', '?')}]** {c.get('text', '')}")
                        metadata = c.get("metadata", {})
                        if metadata:
                            md_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                            st.markdown(f"*Metadata:* {md_str}")

                    st.subheader("🤖 Answer (RAG pipeline)")
                else:
                    st.subheader("🤖 Answer (LLM fallback, no retrieved context)")

                # Display answer
                st.markdown(f"**{answer}**")

                # Display sources with metadata
                sources: List[dict] = result.get("sources", [])
                if sources:
                    st.subheader("🔗 Sources with Metadata")
                    for s in sources:
                        snippet = s.get("snippet", "")
                        st.markdown(f"Source {s.get('id', '?')}: {snippet[:400]}{'...' if len(snippet)>400 else ''}")
                        source_metadata = s.get("metadata", {})
                        if source_metadata:
                            md_str = ", ".join([f"{k}: {v}" for k, v in source_metadata.items()])
                            st.markdown(f"*Metadata:* {md_str}")

                # Optionally save all retrieved contexts + metadata to CSV
                if contexts:
                    save_df = pd.DataFrame([
                        {"index": c.get("index"), "text": c.get("text"), **c.get("metadata", {})} for c in contexts
                    ])
                    save_df.to_csv("retrieved_contexts_metadata.csv", index=False)
                    st.success("Retrieved contexts and metadata saved to 'retrieved_contexts_metadata.csv'")

            else:
                st.error(f"⚠️ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"🚨 Could not connect to backend: {e}")
