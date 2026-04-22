"""
app.py — Streamlit UI for the MAS Regulatory Q&A RAG system
Run with: streamlit run src/app.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from retriever import build_qa_chain, query

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAS Regulatory Q&A",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 MAS Regulatory Q&A")
st.caption("Ask questions about Singapore MAS financial regulations — answers grounded in official documents")

# ── Sidebar: info + settings ─────────────────────────────────────────────────
with st.sidebar:
    st.header("About this system")
    st.markdown("""
    **How it works:**
    1. Your question is converted to an embedding vector
    2. Top-3 most relevant document chunks are retrieved from FAISS
    3. Those chunks + your question are sent to OpenRouter
    4. Answer is generated citing the actual source documents
    
    **Tech stack:**
    - LangChain (orchestration)
    - Sentence Transformers (embeddings)  
    - ChromaDB (vector store)
    - Gemini 1.5 Flash (LLM, free tier)
    - Streamlit (UI)
    """)
    
    k_val = st.slider("Chunks to retrieve (k)", min_value=1, max_value=6, value=3)
    st.caption("Higher k = more context, slower response")

# ── Load QA chain (cached so it doesn't reload on every interaction) ─────────
@st.cache_resource
def get_chain(k):
    with st.spinner("Loading RAG chain..."):
        return build_qa_chain(k=k)

# ── Example questions ─────────────────────────────────────────────────────────
st.subheader("Try these questions")
example_questions = [
    "What are the responsibilities of insurers in product development and pricing?",
    "What payout features are prohibited under MAS Notice 302?",
    "What disclosures must be made for settlement options?",
    "When must insurers obtain MAS approval before launching a product?",
]

cols = st.columns(2)
clicked_question = None
for i, q in enumerate(example_questions):
    with cols[i % 2]:
        if st.button(q, use_container_width=True):
            clicked_question = q

# ── Main Q&A interface ────────────────────────────────────────────────────────
st.divider()

user_question = st.text_input(
    "Ask a question about MAS regulations:",
    value=clicked_question or "",
    placeholder="e.g. What are the requirements for digital payment token services?"
)

if st.button("Get Answer", type="primary", disabled=not user_question):
    qa_chain = get_chain(k_val)
    
    with st.spinner("Searching documents and generating answer..."):
        result = query(user_question, qa_chain)
    
    # Display answer
    st.subheader("Answer")
    st.markdown(result["answer"])
    
    # Display sources
    if result["sources"]:
        st.subheader("Source documents used")
        for i, source in enumerate(result["sources"]):
            with st.expander(f"Source {i+1}: {source['file']} — Page {source['page']}"):
                st.text(source["content"])
    else:
        st.warning("No source documents were retrieved.")

# ── Chat history (bonus feature) ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if user_question and st.session_state.get("last_q") != user_question:
    st.session_state.history.append(user_question)
    st.session_state.last_q = user_question

if st.session_state.history:
    with st.expander("Question history"):
        for past_q in reversed(st.session_state.history[-5:]):
            st.markdown(f"- {past_q}")
