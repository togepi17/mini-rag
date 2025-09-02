# backend/app.py
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Optional imports
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    HuggingFaceEmbeddings = None
    FAISS = None

# PDF reading
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
print("[env] Loaded OPENAI_API_KEY:", "SET" if OPENAI_API_KEY else "NOT SET")

# Create OpenAI client
client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("[env] OpenAI client created successfully.")
    except Exception as e:
        print("[env] Failed to create OpenAI client:", e)

# FastAPI app
app = FastAPI(title="Mini-RAG (FAISS + OpenAI)")

# Vectorstore and embeddings
_vectorstore = None
_embeddings = None

def ensure_vectorstore_initialized():
    global _vectorstore, _embeddings
    if _vectorstore is not None:
        return
    if HuggingFaceEmbeddings is None or FAISS is None:
        raise RuntimeError("langchain_community or FAISS not installed.")
    print("[vectorstore] Initializing embeddings and FAISS index...")
    _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [
        "RAG stands for Retrieval-Augmented Generation.",
        "FastAPI is a modern Python web framework.",
        "FAISS is a library for efficient similarity search."
    ]
    _vectorstore = FAISS.from_texts(docs, _embeddings)
    print(f"[vectorstore] Initialized. Indexed {len(docs)} docs.")

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Mini-RAG is running!"}


@app.get("/search")
def search(query: str):
    ensure_vectorstore_initialized()
    results = _vectorstore.similarity_search(query, k=1)
    top_text = results[0].page_content if results else ""
    return {"query": query, "result": top_text}


@app.get("/llm-test")
def llm_test():
    if client is None:
        return JSONResponse(status_code=400, content={"ok": False, "error": "OpenAI client not configured."})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a concise assistant."},
                  {"role": "user", "content": "Reply with just: OK"}],
        temperature=0.0,
    )
    try:
        text = resp.choices[0].message.content.strip()
    except Exception:
        text = str(resp)
    return {"ok": True, "model": "gpt-4o-mini", "output": text}


# -------------------------
# RAG pipeline
# -------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/process")
def process(req: QueryRequest):
    ensure_vectorstore_initialized()
    if client is None:
        return JSONResponse(status_code=400, content={"error": "OpenAI client not configured."})

    hits = _vectorstore.similarity_search(req.question, k=req.top_k)
    contexts = []
    sources = []

    for i, h in enumerate(hits):
        text = getattr(h, "page_content", str(h))
        contexts.append(text)
        sources.append({"id": i + 1, "snippet": text[:400]})

    if not contexts:
        # fallback to LLM-only answer
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": req.question}],
            temperature=0.0,
            max_tokens=512
        )
        try:
            answer = resp.choices[0].message.content.strip()
        except Exception:
            answer = str(resp)
        return {"question": req.question, "contexts": [], "answer": answer, "sources": []}

    numbered_context = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    system_msg = (
        "You are a helpful, precise assistant. Answer the user's question using ONLY the provided contexts. "
        "Cite the source(s) inline by using [n]. If the answer is not contained, say you don't know."
    )
    user_prompt = f"Contexts:\n{numbered_context}\n\nQuestion: {req.question}\nAnswer concisely with inline citations."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=512
    )
    try:
        answer = resp.choices[0].message.content.strip()
    except Exception:
        answer = str(resp)

    return {
        "question": req.question,
        "contexts": [{"index": i+1, "text": c} for i, c in enumerate(contexts)],
        "answer": answer,
        "sources": sources
    }

# -------------------------
# File upload / injection
# -------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".pdf"):
            if PdfReader is None:
                return {"error": "PyPDF2 not installed. Run pip install PyPDF2"}
            reader = PdfReader(file.file)  # use .file, not raw bytes
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

        elif file.filename.endswith(".txt"):
            text = (await file.read()).decode("utf-8")

        else:
            return {"error": "Unsupported file type. Only PDF or TXT allowed."}

        ensure_vectorstore_initialized()
        _vectorstore.add_texts([text])

        return {"success": True, "filename": file.filename, "text_snippet": text[:200]}

    except Exception as e:
        return {"error": str(e)}

