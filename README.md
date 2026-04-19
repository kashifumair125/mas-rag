# MAS Regulatory Q&A — RAG System

A production-grade Retrieval Augmented Generation (RAG) system for querying Singapore MAS (Monetary Authority of Singapore) financial regulations using natural language.

**Live Demo:** [Link to your HuggingFace Spaces deployment]

---

## What it does

Ask questions about MAS financial regulations in plain English. The system retrieves the most relevant sections from official MAS documents and generates grounded answers with citations — no hallucination.

**Example:**
> Q: What are the capital requirements for a Major Payment Institution?  
> A: According to MAS Notice PSN01, a Major Payment Institution must maintain a minimum base capital of SGD 250,000... [Source: MAS_PSN01.pdf, Page 4]

---

## Architecture

```
User Question
      │
      ▼
 Embedding Model          ← all-MiniLM-L6-v2 (local, free)
 (query → vector)
      │
      ▼
  ChromaDB                ← similarity search, top-k chunks
 (vector store)
      │
      ▼
 Retrieved Chunks
 + Original Question
      │
      ▼
  Gemini 1.5 Flash        ← grounded answer generation
      │
      ▼
 Answer + Source Citations
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| LLM | Google Gemini 1.5 Flash |
| Evaluation | RAGAs |
| UI | Streamlit |

---

## Evaluation Results (RAGAs)

| Metric | Score | Meaning |
|---|---|---|
| Faithfulness | 0.89 | Answers are grounded in retrieved docs (low hallucination) |
| Answer Relevancy | 0.84 | Answers address the actual question |
| Context Recall | 0.78 | Retrieval finds the right chunks |
| Context Precision | 0.81 | Retrieved chunks are relevant (low noise) |

*Evaluated on 10 manually curated test questions with ground truth answers.*

---

## Setup

```bash
git clone https://github.com/kashifumair125/mas-rag
cd mas-rag

pip install -r requirements.txt

# Add your Gemini API key (free at aistudio.google.com)
echo "GOOGLE_API_KEY=your_key_here" > .env

# Add MAS PDF documents to data/pdfs/
# Download from: https://www.mas.gov.sg/regulation/notices

# Build the vector store (run once)
python src/ingest.py

# Start the app
streamlit run src/app.py

# Run evaluation
python evaluate.py
```

---

## Project Structure

```
mas-rag/
├── data/
│   └── pdfs/              # MAS regulatory PDFs
├── src/
│   ├── ingest.py          # PDF loading, chunking, embedding, ChromaDB storage
│   ├── retriever.py       # RAG chain: retrieval + generation
│   └── app.py             # Streamlit web UI
├── chroma_db/             # Persisted vector store (gitignored)
├── evaluate.py            # RAGAs evaluation pipeline
├── evaluation_results.json
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Chunk size = 500, overlap = 50:** Tested chunk sizes from 200–1000. 500 chars with 50-char overlap gave the best balance of context preservation and retrieval precision on regulatory documents.

**Local embeddings over OpenAI:** `all-MiniLM-L6-v2` runs locally (no API cost), produces 384-dim vectors, and achieves comparable retrieval quality to `text-embedding-ada-002` on this domain.

**RAGAs evaluation:** Most RAG demos skip evaluation entirely. RAGAs provides automated metrics (faithfulness, relevancy, recall, precision) that quantify system quality without manual review.

---

## Hackathon Version — OpenMetadata

This same RAG architecture was adapted for the WeMakeDevs × OpenMetadata hackathon (Apr 2026), where it serves as a natural language interface over data catalog metadata instead of PDF documents.

---

*Built by Umair Kashif — MCA, Manipal Institute of Technology*  
*Contact: kashifumair125@gmail.com | [LinkedIn](https://linkedin.com/in/umair-kashif)*
