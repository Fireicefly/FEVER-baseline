"""
Official FEVER Score computation.

The FEVER score is the official metric for the FEVER shared task.
It is stricter than label accuracy as it also evaluates evidence retrieval.
"""

from typing import List, Tuple


def compute_fever_score(
    predictions: List[int],
    labels: List[int],
    predicted_evidence_list: List[List[Tuple[str, int, str]]],
    gold_evidence_list: List[List[List]]
) -> float:
    """
    Compute the official FEVER score.

    The FEVER score is stricter than accuracy:
    - For SUPPORTS/REFUTES: Correct only if label is correct AND at least one
      evidence sentence is correct
    - For NOT ENOUGH INFO: Correct only if label is correct

    Args:
        predictions: Predicted labels (indices)
        labels: True labels (indices)
        predicted_evidence_list: List of predicted evidence for each claim
                                 Each item is a list of (page_id, sent_id, text) tuples
        gold_evidence_list: List of gold evidence from dataset
                           Each item is the 'evidence' field from FEVER data

    Returns:
        FEVER score (float between 0 and 1)
    """
    correct = 0
    total = len(predictions)

    # Label indices: 0=SUPPORTS, 1=REFUTES, 2=NOT ENOUGH INFO
    NOT_ENOUGH_INFO_IDX = 2

    for pred, label, pred_evidence, gold_evidence in zip(
        predictions, labels, predicted_evidence_list, gold_evidence_list
    ):
        # Check if label prediction is correct
        if pred != label:
            continue

        # For NOT ENOUGH INFO, only label needs to be correct
        if label == NOT_ENOUGH_INFO_IDX:
            correct += 1
            continue

        # For SUPPORTS and REFUTES, need at least one correct evidence sentence
        # Extract predicted (page_id, sent_id) pairs
        predicted_pairs = set()
        for page_id, sent_id, _ in pred_evidence:
            if page_id and sent_id is not None:
                predicted_pairs.add((page_id, sent_id))

        # Extract gold (page_id, sent_id) pairs
        gold_pairs = set()
        if gold_evidence:
            for evidence_set in gold_evidence:
                for evidence in evidence_set:
                    if len(evidence) >= 4:
                        page_id = evidence[2]
                        sent_id = evidence[3]
                        if page_id and sent_id is not None:
                            gold_pairs.add((page_id, sent_id))

        # Check if there's at least one overlap
        if predicted_pairs & gold_pairs:  # Set intersection
            correct += 1

    return correct / total if total > 0 else 0.0


def compute_label_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    Compute label accuracy (ignoring evidence).

    Args:
        predictions: Predicted labels
        labels: True labels

    Returns:
        Accuracy as float between 0 and 1
    """
    correct = sum([p == l for p, l in zip(predictions, labels)])
    return correct / len(labels) if len(labels) > 0 else 0.0


def compute_evidence_precision_recall(
    predicted_evidence_list: List[List[Tuple[str, int, str]]],
    gold_evidence_list: List[List[List]]
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for evidence retrieval.

    Args:
        predicted_evidence_list: List of predicted evidence
        gold_evidence_list: List of gold evidence

    Returns:
        (precision, recall, f1) tuple
    """
    total_predicted = 0
    total_gold = 0
    total_correct = 0

    for pred_evidence, gold_evidence in zip(predicted_evidence_list, gold_evidence_list):
        # Extract predicted pairs
        predicted_pairs = set()
        for page_id, sent_id, _ in pred_evidence:
            if page_id and sent_id is not None:
                predicted_pairs.add((page_id, sent_id))

        # Extract gold pairs
        gold_pairs = set()
        if gold_evidence:
            for evidence_set in gold_evidence:
                for evidence in evidence_set:
                    if len(evidence) >= 4:
                        page_id = evidence[2]
                        sent_id = evidence[3]
                        if page_id and sent_id is not None:
                            gold_pairs.add((page_id, sent_id))

        # Count
        total_predicted += len(predicted_pairs)
        total_gold += len(gold_pairs)
        total_correct += len(predicted_pairs & gold_pairs)

    precision = total_correct / total_predicted if total_predicted > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
