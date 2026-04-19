"""
openmetadata_agent.py — Hackathon version: RAG over OpenMetadata catalog
Instead of PDFs, this ingests metadata from OpenMetadata's API.
Paradox T-01: Natural language interface for data discovery.

Setup:
  1. Run OpenMetadata locally: docker run -p 8585:8585 openmetadata/server
  2. Set OPENMETADATA_HOST in .env
  3. Run: python openmetadata_agent.py

What it does: User asks "Which tables have PII data?" or
"What datasets are related to payments?" → RAG retrieves relevant
metadata → LLM gives a clear, cited answer.
"""

import os
import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

OPENMETADATA_HOST = os.getenv("OPENMETADATA_HOST", "http://localhost:8585")
CHROMA_DIR = "./chroma_openmetadata"
EMBED_MODEL = "all-MiniLM-L6-v2"


def fetch_openmetadata_tables(limit: int = 100) -> list[Document]:
    """
    Fetch table metadata from OpenMetadata API.
    Converts each table's metadata into a LangChain Document.
    """
    documents = []
    
    try:
        # OpenMetadata REST API — get all tables
        url = f"{OPENMETADATA_HOST}/api/v1/tables"
        params = {"limit": limit, "fields": "columns,tags,description,owner"}
        headers = {"Authorization": f"Bearer {os.getenv('OM_JWT_TOKEN', '')}"}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for table in data.get("data", []):
            # Build a rich text description of each table for embedding
            table_name = table.get("fullyQualifiedName", table.get("name", "unknown"))
            description = table.get("description", "No description available")
            
            # Format columns info
            columns = table.get("columns", [])
            col_text = ", ".join([
                f"{c['name']} ({c.get('dataType', '?')})"
                for c in columns[:20]  # cap at 20 columns
            ])
            
            # Format tags (PII, sensitive, etc.)
            tags = table.get("tags", [])
            tag_names = ", ".join([t.get("tagFQN", "") for t in tags]) or "none"
            
            # Combine into searchable text
            content = f"""Table: {table_name}
Description: {description}
Columns: {col_text}
Tags: {tag_names}
Owner: {table.get('owner', {}).get('name', 'unassigned')}"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "table_name": table_name,
                    "source": "openmetadata",
                    "table_id": table.get("id", "")
                }
            )
            documents.append(doc)
    
    except requests.exceptions.ConnectionError:
        print("Cannot connect to OpenMetadata. Using sample data for demo.")
        # Demo documents so the app still works without OpenMetadata running
        documents = _get_sample_documents()
    
    print(f"Fetched {len(documents)} table documents from OpenMetadata")
    return documents


def _get_sample_documents() -> list[Document]:
    """Sample data for when OpenMetadata isn't running (demo mode)."""
    samples = [
        ("payments.transactions", "Records all payment transactions. Contains customer PII.", 
         "transaction_id, customer_id, amount, currency, timestamp, card_last4", "PII, Financial"),
        ("customers.profiles", "Customer demographic and contact information.",
         "customer_id, name, email, phone, address, date_of_birth", "PII, Sensitive"),
        ("risk.fraud_signals", "Computed fraud risk scores per transaction.",
         "transaction_id, risk_score, signal_type, model_version, created_at", "Internal"),
        ("products.catalog", "Product listings and pricing information.",
         "product_id, name, category, price, stock_count, last_updated", "Public"),
    ]
    docs = []
    for name, desc, cols, tags in samples:
        docs.append(Document(
            page_content=f"Table: {name}\nDescription: {desc}\nColumns: {cols}\nTags: {tags}",
            metadata={"table_name": name, "source": "sample"}
        ))
    return docs


def build_metadata_rag():
    """Build the RAG chain over OpenMetadata catalog."""
    
    # 1. Fetch metadata as documents
    documents = fetch_openmetadata_tables()
    
    # 2. Chunk (metadata docs are short, so small chunks work fine)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)
    
    # 3. Embed + store
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    
    # 4. LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    # 5. Custom prompt for metadata Q&A
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a data catalog assistant. Use the OpenMetadata catalog information below to answer questions about available datasets, their schemas, tags, and ownership.

If you find tables with PII or sensitive tags, always mention this clearly.
If the answer is not in the catalog, say so.

Catalog context:
{context}

Question: {question}

Answer (mention specific table names and their attributes):"""
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


# ── Simple CLI demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building OpenMetadata RAG agent...")
    chain = build_metadata_rag()
    
    test_questions = [
        "Which tables contain PII data?",
        "What datasets are related to payments or transactions?",
        "Which tables have no description or owner assigned?",
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        result = chain.invoke({"query": q})
        print(f"A: {result['result']}")
        print(f"Sources: {[d.metadata['table_name'] for d in result['source_documents']]}")
