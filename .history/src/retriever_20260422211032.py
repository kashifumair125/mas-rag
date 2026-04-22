"""
retriever.py — RAG chain using FAISS + OpenRouter via LCEL pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

FAISS_DIR   = "./faiss_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert on Singapore MAS financial regulations.

Use ONLY the following context to answer the question.
If the answer is not in the context, say "I cannot find this in the provided documents."
Always mention which document your answer comes from.

Context:
{context}

Question: {question}

Answer (with source references):"""
)


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def build_qa_chain(k: int = 3):
    llm = ChatOpenAI(
        model="openrouter/auto",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        temperature=0.1,
        max_tokens=1024
    )

    vectorstore = load_vectorstore()

    # ✅ FIX 1: Increase k to retrieve more chunks for better coverage
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # MMR reduces redundant chunks
        search_kwargs={"k": k, "fetch_k": k * 3}
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            # ✅ FIX 2: Include source info inside the context so LLM can cite it
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def query(question: str, qa_chain=None) -> dict:
    if qa_chain is None:
        qa_chain = build_qa_chain()

    chain = qa_chain["chain"]
    retriever = qa_chain["retriever"]

    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    sources = []
    for doc in source_docs:
        # ✅ FIX 3: Correct metadata key is "source" not "source_file"
        raw_source = doc.metadata.get("source", "unknown")
        # Clean up path to just show filename
        file_name = raw_source.replace("\\", "/").split("/")[-1]
        sources.append({
            "content": doc.page_content[:500],  # Show more content
            "file": file_name,
            "page": doc.metadata.get("page", "?")
        })

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    result = query("What payout features are prohibited under MAS Notice 302?")
    print("Answer:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['file']} (page {s['page']})")