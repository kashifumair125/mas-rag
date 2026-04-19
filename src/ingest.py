"""
ingest.py — Load PDFs, chunk them, embed, store in FAISS
Run this ONCE to build your vector store from your PDF documents.
"""

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
PDF_DIR    = Path("data/pdfs")
FAISS_DIR  = "./faiss_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50


def load_pdfs(pdf_dir: Path) -> list:
    documents = []
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}. Add your MAS PDFs there.")
        return documents

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
        documents.extend(docs)

    print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDFs")
    return documents


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def build_vectorstore(chunks: list):
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Embedding chunks and storing in FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"Saved FAISS index to {FAISS_DIR}")
    return vectorstore


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_pdfs(PDF_DIR)
    if not documents:
        return

    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)

    # Quick sanity check
    print("\nSanity check — querying vectorstore...")
    results = vectorstore.similarity_search("financial regulations", k=2)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1} (from {doc.metadata.get('source_file', 'unknown')}):")
        print(doc.page_content[:200] + "...")

    print("\nIngest complete. Run: streamlit run src/app.py")


if __name__ == "__main__":
    main()