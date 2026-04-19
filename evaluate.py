"""
evaluate.py — RAGAs evaluation for your RAG system
This is what makes your project stand out. Most people skip evaluation entirely.

RAGAs metrics:
  - Faithfulness: Is the answer supported by the retrieved chunks? (no hallucination)
  - Answer Relevancy: Does the answer actually address the question?
  - Context Recall: Did retrieval find the right chunks?
  - Context Precision: Were the retrieved chunks relevant?

Run with: python evaluate.py
"""

import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from retriever import build_qa_chain, load_retriever


# ── Test questions with ground truth answers ──────────────────────────────────
# In a real project, a domain expert writes these.
# For your project, write 5-10 based on your actual MAS PDFs.
TEST_SET = [
    {
        "question": "What is the minimum base capital requirement for a Major Payment Institution?",
        "ground_truth": "A Major Payment Institution must maintain a minimum base capital of SGD 250,000."
    },
    {
        "question": "What AML obligations apply to digital payment token service providers?",
        "ground_truth": "Digital payment token service providers must conduct customer due diligence, monitor transactions, and report suspicious transactions to the Suspicious Transaction Reporting Office."
    },
    {
        "question": "How does MAS define a digital payment token?",
        "ground_truth": "A digital payment token is a cryptographically secured digital representation of value that can be transferred, stored or traded electronically and is used as a medium of exchange."
    },
]


def run_evaluation():
    """Run RAGAs evaluation on the test set."""
    print("Building RAG chain...")
    qa_chain = build_qa_chain(k=3)
    retriever = load_retriever(k=3)
    
    # Collect outputs from RAG system
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    print(f"\nRunning {len(TEST_SET)} test questions...")
    for i, test_case in enumerate(TEST_SET):
        q = test_case["question"]
        print(f"  [{i+1}/{len(TEST_SET)}] {q[:60]}...")
        
        # Get RAG answer
        result = qa_chain.invoke({"query": q})
        answer = result["result"]
        
        # Get retrieved contexts
        docs = retriever.get_relevant_documents(q)
        context_list = [doc.page_content for doc in docs]
        
        questions.append(q)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(test_case["ground_truth"])
    
    # Build HuggingFace Dataset (required by RAGAs)
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    print("\nRunning RAGAs evaluation...")
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]
    )
    
    # Display results
    print("\n" + "="*50)
    print("RAGAs Evaluation Results")
    print("="*50)
    scores = results.to_pandas()
    
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        score = scores[metric].mean()
        status = "GOOD" if score > 0.7 else "NEEDS IMPROVEMENT"
        print(f"  {metric:<25} {score:.3f}   [{status}]")
    
    print("="*50)
    
    # Save results to JSON for your README/portfolio
    output = {
        "metrics": {
            "faithfulness": float(scores["faithfulness"].mean()),
            "answer_relevancy": float(scores["answer_relevancy"].mean()),
            "context_recall": float(scores["context_recall"].mean()),
            "context_precision": float(scores["context_precision"].mean()),
        },
        "num_test_cases": len(TEST_SET),
        "model": "gemini-1.5-flash",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 500,
        "k": 3,
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to evaluation_results.json")
    print("Add these scores to your README — it shows you measure quality, not just build demos.")
    
    return output


if __name__ == "__main__":
    run_evaluation()
