# frontend.py
import streamlit as st
import requests

st.set_page_config(page_title="Mini-RAG App", page_icon="🤖")

st.title("Mini-RAG App")
st.markdown(
    """
    This app uses a **hybrid pipeline**:
    - **RAG**: Uses FAISS to retrieve relevant documents and answer questions.
    - **LLM-only fallback**: If no relevant documents, the LLM answers directly.
    - **Injection**: Upload PDF or TXT files to add context dynamically.
    """
)

# -------------------------------
# Section 1: Upload documents
# -------------------------------
st.subheader("Step 1: Upload context documents (PDF or TXT)")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

if uploaded_file:
    if st.button("Inject Context"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post("http://localhost:8000/upload", files=files)
            if response.status_code == 200:
                st.success(f"File injected successfully: {uploaded_file.name}")
            else:
                st.error(f"⚠️ Error injecting file: {response.text}")
        except Exception as e:
            st.error(f"🚨 Could not inject file: {e}")

st.markdown("---")

# -------------------------------
# Section 2: Ask a question
# -------------------------------
st.subheader("Step 2: Ask a question")
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
                contexts = result.get("contexts", [])
                answer = result.get("answer", "")

                # Display retrieved contexts
                if contexts:
                    st.subheader("Retrieved Contexts:")
                    for c in contexts:
                        st.markdown(f"[{c['index']}] {c['text']}")
                    st.subheader("Answer (RAG pipeline):")
                else:
                    st.subheader("Answer (LLM fallback, no retrieved context):")

                st.markdown(f"**🤖 Bot:** {answer}")

                # Display sources if available
                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources:")
                    for s in sources:
                        st.markdown(f"Source {s['id']}: {s['snippet']}")

            else:
                st.error(f"⚠️ Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"🚨 Could not connect to backend: {e}")

st.markdown("---")

# -------------------------------
# Section 3: Optional LLM test
# -------------------------------
if st.button("Test LLM Backend"):
    try:
        response = requests.get("http://localhost:8000/llm-test")
        if response.status_code == 200:
            result = response.json()
            st.markdown(f"**LLM Test Response:** {result.get('output', '')}")
        else:
            st.error(f"⚠️ Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"🚨 Could not connect to backend: {e}")