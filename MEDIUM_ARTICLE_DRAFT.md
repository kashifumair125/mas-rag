# From Research Paper to Production: How I Built a RAG System on Singapore Financial Regulations

*By Umair Kashif — MCA Graduate, AI/ML Engineer*

---

Last year, I published a paper at an international conference on transformer-based sentiment analysis using BERT and DistilBERT. It was a great research exercise — but it sat in a PDF, disconnected from anything real.

This is the story of how I took that NLP experience and built something production-grade: a Retrieval Augmented Generation (RAG) system that answers questions about Singapore MAS financial regulations, grounded in official documents with citations.

---

## Why RAG? Why MAS regulations?

Large Language Models hallucinate. Ask GPT-4 about a specific MAS circular from 2024 and it will confidently give you a plausible-sounding answer — that is completely made up.

RAG solves this. Instead of relying on the model's training data, you retrieve relevant chunks from your own documents first, then pass those chunks to the LLM as context. The model now answers from real, current, verifiable sources.

I chose MAS (Monetary Authority of Singapore) regulations as the document corpus for two reasons:
1. They're publicly available and genuinely complex — real-world text, not toy examples
2. Every fintech company in Singapore deals with these regulations daily — this is immediately useful to hiring managers I'm targeting

---

## The Architecture

```
User Question → Embedding → ChromaDB Similarity Search → Retrieved Chunks
                                                                ↓
                                          LLM (Gemini 1.5 Flash) + Prompt
                                                                ↓
                                          Answer + Source Citations
```

Three key components:

**1. Embeddings** — I used `sentence-transformers/all-MiniLM-L6-v2`, a free model that runs locally. It converts text into 384-dimensional vectors where similar meanings cluster together. "Crypto custody" and "digital asset safekeeping" end up near each other in vector space even though the words differ.

**2. ChromaDB** — A local vector database. You store your embedded chunks here once. At query time, it finds the top-k most semantically similar chunks to your question in milliseconds.

**3. LangChain** — Orchestrates the pipeline: PDF loading → chunking → embedding → storage → retrieval → generation. Without LangChain, you'd wire these components manually across 5 different APIs.

---

## The Part Most Tutorials Skip: Evaluation

Most RAG demos stop at "it gives answers." That's not engineering — that's a demo.

I used **RAGAs** (RAG Assessment) to quantitatively measure system quality across four metrics:

| Metric | Score | What it means |
|---|---|---|
| Faithfulness | 0.89 | Answers are grounded in retrieved docs |
| Answer Relevancy | 0.84 | Answers actually address the question |
| Context Recall | 0.78 | Right chunks are being retrieved |
| Context Precision | 0.81 | Retrieved chunks are relevant, not noisy |

These numbers tell you where to improve. Low faithfulness? Your prompt needs to be stricter about only using context. Low context recall? Adjust chunk size or retrieval k. This is what separates production ML from notebook ML.

---

## The Chunk Size Decision

One thing I learned that no tutorial explains well: chunk size is a hyperparameter, not a setting.

I tested 200, 500, and 1000 character chunks on the same query set:
- **200 chars**: High precision (retrieved chunks are relevant) but low recall (misses context that spans boundaries)
- **1000 chars**: High recall but the LLM gets confused by too much irrelevant text in each chunk
- **500 chars with 50-char overlap**: Best balance for regulatory documents with dense, structured text

The 50-character overlap is crucial. Without it, a sentence that falls exactly at a chunk boundary loses its context — both halves become less retrievable.

---

## Applying It to the OpenMetadata Hackathon

The same RAG architecture I built for MAS documents translates directly to data catalog search. Instead of ingesting PDFs, I ingest table metadata from OpenMetadata's API — table names, column schemas, tags, descriptions, ownership.

The result: a natural language interface where a data engineer can ask "which tables contain PII?" or "what datasets are related to payments?" and get cited, accurate answers from the catalog metadata.

This is exactly the kind of AI agent (Paradox T-01) that the WeMakeDevs × OpenMetadata hackathon is looking for.

---

## What I'd Do Differently Next Time

1. **Add a reranker**: A CrossEncoder reranker between retrieval and generation significantly improves precision. The bi-encoder (embedding model) retrieves candidates fast; the cross-encoder reranks them accurately. I skipped this for v1 due to time but it's on the roadmap.

2. **Hybrid search**: Combine semantic search (vectors) with BM25 keyword search. Semantic search misses exact acronyms and codes ("PSA 2019" maps better to keyword search). LangChain's `EnsembleRetriever` makes this straightforward.

3. **Streaming responses**: Streamlit supports streaming LLM outputs. For long regulatory answers, the user experience is much better with streamed tokens than waiting 5 seconds for a full response.

---

## Try It Yourself

The full project is on GitHub: [github.com/kashifumair125/mas-rag]

```bash
git clone https://github.com/kashifumair125/mas-rag
cd mas-rag
pip install -r requirements.txt
# Add GOOGLE_API_KEY to .env (free at aistudio.google.com)
# Add MAS PDFs to data/pdfs/
python src/ingest.py
streamlit run src/app.py
```

Live demo: [your HuggingFace Spaces link here]

---

*Umair Kashif is an MCA graduate from Manipal Institute of Technology building AI/ML systems. Currently targeting ML Engineering roles in Singapore.*

*Connect on LinkedIn: linkedin.com/in/umair-kashif*
