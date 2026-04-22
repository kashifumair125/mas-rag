"""
retriever.py — RAG chain using FAISS + OpenRouter via LCEL pipeline
No deprecated RetrievalQA — uses modern LangChain Expression Language
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
    openai_api_key=os.environ.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
    temperature=0.1,
    max_tokens=1024
)

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
        sources.append({
            "content": doc.page_content[:300],
            "file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", "?")
        })

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    result = query("What are the AML/CFT requirements for banks?")
    print("Answer:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['file']} (page {s['page']})")