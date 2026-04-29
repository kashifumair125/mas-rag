---
title: MAS RAG
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.37.1"
app_file: src/app.py
pinned: false
---

# MAS Regulatory Q&A — RAG System

> Ask questions about Singapore MAS financial regulations in plain English. Get grounded answers with exact source citations from official MAS documents.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.2-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red) ![FAISS](https://img.shields.io/badge/FAISS-vector--store-orange)

## Live Demo
- 🚀 HuggingFace Spaces: https://huggingface.co/spaces/kashii1/mas-rag
- ☁️ GCP Cloud Run: https://mas-rag-26912506161.asia-southeast1.run.app

---

## What it does

Most LLMs hallucinate answers about specific regulations. This system retrieves exact chunks from official MAS PDFs before generating any answer — so every response is grounded in real documents with page citations.

**Example question:** *"What cyber hygiene practices are mandatory for financial institutions?"*

**What the system does:**
1. Converts your question into an embedding vector (semantic search)
2. Retrieves the top-3 most relevant chunks from the FAISS vector database
3. Sends those chunks + your question to the LLM
4. Returns a structured answer with exact source document and page number

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | LangChain 0.2 (LCEL) | Industry standard for RAG pipelines |
| Vector store | FAISS | Fast similarity search, runs locally |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | Free, no API cost, 384 dimensions |
| LLM | OpenRouter (Llama 3 / auto-routing) | Free tier, OpenAI-compatible API |
| UI | Streamlit | Rapid deployment, clean interface |
| PDF loading | PyPDF | Extract text from MAS circulars |

---

## Documents Indexed

| Document | Coverage |
|----------|----------|
| MAS Notice 626 (2024) | AML/CFT requirements for banks — customer due diligence, wire transfers, suspicious transaction reporting |
| MAS Notice FSM-N14 (2024) | Cyber hygiene — administrative accounts, security patches, MFA requirements |
| MAS Notice 302 (2024) | Insurance product development and pricing requirements |
| PSN09 (2026) | Payment services — specified matters and forms |

---

## Architecture

```
User question
     │
     ▼
Sentence Transformer (embed question)
     │
     ▼
FAISS similarity search → top-k chunks
     │
     ▼
RAG prompt (chunks + question)
     │
     ▼
OpenRouter LLM → answer with citations
     │
     ▼
Streamlit UI (answer + source expanders)
```

---

## Run Locally

**Prerequisites:** Python 3.11+, Git

```bash
# 1. Clone
git clone https://github.com/kashifumair125/mas-rag.git
cd mas-rag

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. API key
# Get free key from openrouter.ai
echo "OPENROUTER_API_KEY=your_key_here" > .env

# 5. Add your PDFs
# Put MAS PDF documents in data/pdfs/
# Download from mas.gov.sg/regulation/notices

# 6. Build vector database (run once)
python src/ingest.py

# 7. Launch
streamlit run src/app.py
```

App opens at **http://localhost:8501**

---

## Project Structure

```
mas-rag/
├── src/
│   ├── app.py              # Streamlit UI
│   ├── ingest.py           # PDF loading, chunking, FAISS indexing
│   └── retriever.py        # RAG chain (LCEL pipeline)
├── data/
│   └── pdfs/               # MAS regulatory documents
├── faiss_db/               # Vector store (auto-generated, gitignored)
├── tests/
│   └── test_retriever.py   # Unit tests
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
├── Dockerfile              # Container definition
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Chunk size: 500 chars with 50 overlap** — prevents information loss at boundaries while keeping chunks focused enough for precise retrieval.

**FAISS over ChromaDB** — zero C++ dependency issues on Windows/Python 3.13, faster indexing for small-medium corpora.

**LCEL pipeline over RetrievalQA** — uses modern LangChain Expression Language, avoids deprecated chains, better composability.

**all-MiniLM-L6-v2 embeddings** — 80MB model, runs on CPU, 384 dimensions. Sufficient quality for regulatory document retrieval without API cost.

---

## What I Learned Building This

- RAG architecture: chunking strategy, embedding model selection, similarity search tuning
- LangChain LCEL pipelines vs legacy chains
- FAISS vector indexing and persistence
- Dependency management with Python 3.13 compatibility issues
- Streamlit caching with `@st.cache_resource` for LLM chains

---

## Roadmap

- [ ] Deploy to HuggingFace Spaces
- [ ] Docker containerization
- [ ] GitHub Actions CI/CD
- [ ] GCP Cloud Run deployment
- [ ] RAGAs evaluation metrics
- [ ] Add more MAS documents (target: 20+ circulars)
- [ ] MLflow experiment tracking for chunk size experiments

---
<img width="1900" height="877" alt="image" src="https://github.com/user-attachments/assets/d6458f18-a0af-4826-870e-ae9d765816b3" />
<img width="1918" height="868" alt="image" src="https://github.com/user-attachments/assets/8306fe01-d8a1-40e4-a9cf-0e6fafcc3067" />

---

## Author

**Umair Kashif** — MCA, Manipal University  

[GitHub](https://github.com/kashifumair125) | [LinkedIn](https://linkedin.com/in/kashifumair125)

--- 
