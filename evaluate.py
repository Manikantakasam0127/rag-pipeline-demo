# RAG evaluation using RAGAS
# added this after noticing inconsistent answers
# TODO: add LangSmith integration later

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

def evaluate_rag(questions, answers, contexts, ground_truths):
    """
    Evaluate RAG pipeline quality using RAGAS metrics.
    faithfulness - is the answer grounded in the context?
    answer_relevancy - is the answer relevant to the question?
    context_precision - is the retrieved context precise?
    context_recall - did we retrieve all relevant context?
    """
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    # running all 4 metrics - faithfulness matters most for our use case
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )

    return results

def print_scores(results):
    print("\n--- RAG Evaluation Results ---")
    print(f"Faithfulness:      {results['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['context_precision']:.3f}")
    print(f"Context Recall:    {results['context_recall']:.3f}")
    print("------------------------------\n")
