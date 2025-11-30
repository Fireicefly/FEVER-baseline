"""
Test FEVER scorer implementation.
"""

from fever_scorer import compute_fever_score, compute_evidence_precision_recall, compute_label_accuracy


def test_fever_score():
    """Test FEVER score computation."""
    print("="*80)
    print("Testing FEVER Score Implementation")
    print("="*80)

    # Test case 1: Perfect predictions (label + evidence)
    print("\nTest 1: Perfect predictions")
    predictions = [0, 1, 2]  # SUPPORTS, REFUTES, NOT ENOUGH INFO
    labels = [0, 1, 2]
    predicted_evidence = [
        [("Page1", 0, "text1")],
        [("Page2", 1, "text2")],
        [("Page3", 2, "text3")]
    ]
    gold_evidence = [
        [[[None, None, "Page1", 0]]],
        [[[None, None, "Page2", 1]]],
        [[[None, None, "Page3", 2]]]
    ]

    score = compute_fever_score(predictions, labels, predicted_evidence, gold_evidence)
    accuracy = compute_label_accuracy(predictions, labels)

    print(f"  Label Accuracy: {accuracy * 100:.2f}%")
    print(f"  FEVER Score: {score * 100:.2f}%")
    assert score == 1.0, f"Expected 100%, got {score * 100}%"
    print("  [OK] Perfect predictions work correctly")

    # Test case 2: Correct labels, wrong evidence for SUPPORTS/REFUTES
    print("\nTest 2: Correct labels, wrong evidence")
    predictions = [0, 1, 2]  # SUPPORTS, REFUTES, NOT ENOUGH INFO
    labels = [0, 1, 2]
    predicted_evidence = [
        [("WrongPage", 99, "wrong text")],  # Wrong evidence
        [("WrongPage2", 98, "wrong text2")],  # Wrong evidence
        [("Page3", 2, "text3")]  # Doesn't matter for NEI
    ]
    gold_evidence = [
        [[[None, None, "Page1", 0]]],
        [[[None, None, "Page2", 1]]],
        [[[None, None, "Page3", 2]]]
    ]

    score = compute_fever_score(predictions, labels, predicted_evidence, gold_evidence)
    accuracy = compute_label_accuracy(predictions, labels)

    print(f"  Label Accuracy: {accuracy * 100:.2f}%")
    print(f"  FEVER Score: {score * 100:.2f}%")
    # Only NOT ENOUGH INFO should be counted as correct (1/3 = 33.33%)
    expected = 1.0 / 3.0
    assert abs(score - expected) < 0.01, f"Expected {expected * 100:.2f}%, got {score * 100:.2f}%"
    print("  [OK] Wrong evidence correctly penalized for SUPPORTS/REFUTES")

    # Test case 3: Wrong labels
    print("\nTest 3: Wrong labels")
    predictions = [1, 2, 0]  # All wrong
    labels = [0, 1, 2]
    predicted_evidence = [
        [("Page1", 0, "text1")],
        [("Page2", 1, "text2")],
        [("Page3", 2, "text3")]
    ]
    gold_evidence = [
        [[[None, None, "Page1", 0]]],
        [[[None, None, "Page2", 1]]],
        [[[None, None, "Page3", 2]]]
    ]

    score = compute_fever_score(predictions, labels, predicted_evidence, gold_evidence)
    accuracy = compute_label_accuracy(predictions, labels)

    print(f"  Label Accuracy: {accuracy * 100:.2f}%")
    print(f"  FEVER Score: {score * 100:.2f}%")
    assert score == 0.0, f"Expected 0%, got {score * 100}%"
    print("  [OK] Wrong labels correctly give 0 FEVER score")

    # Test case 4: Partial evidence match (one correct, one wrong)
    print("\nTest 4: Partial evidence match")
    predictions = [0, 0]  # Both SUPPORTS
    labels = [0, 0]
    predicted_evidence = [
        [("Page1", 0, "text1"), ("Page1", 1, "text2")],  # One correct, one extra
        [("WrongPage", 0, "text1"), ("Page2", 1, "text2")]  # One wrong, one correct
    ]
    gold_evidence = [
        [[[None, None, "Page1", 0]]],
        [[[None, None, "Page2", 1]]]
    ]

    score = compute_fever_score(predictions, labels, predicted_evidence, gold_evidence)

    print(f"  FEVER Score: {score * 100:.2f}%")
    # Both should be correct (at least one evidence matches)
    assert score == 1.0, f"Expected 100%, got {score * 100}%"
    print("  [OK] Partial evidence match works correctly")

    # Test evidence precision/recall
    print("\nTest 5: Evidence Precision/Recall")
    predicted_evidence = [
        [("Page1", 0, "text1"), ("Page1", 1, "text2")],  # 1 correct, 1 extra
        [("Page2", 1, "text2")]  # 1 correct
    ]
    gold_evidence = [
        [[[None, None, "Page1", 0]], [[None, None, "Page1", 2]]],  # 2 gold (we got 1)
        [[[None, None, "Page2", 1]], [[None, None, "Page2", 2]]]  # 2 gold (we got 1)
    ]

    precision, recall, f1 = compute_evidence_precision_recall(predicted_evidence, gold_evidence)

    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall: {recall * 100:.2f}%")
    print(f"  F1: {f1 * 100:.2f}%")

    # 3 predicted total, 2 correct -> precision = 2/3
    # 4 gold total, 2 correct -> recall = 2/4 = 1/2
    expected_precision = 2.0 / 3.0
    expected_recall = 2.0 / 4.0
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)

    assert abs(precision - expected_precision) < 0.01, f"Expected precision {expected_precision:.2f}, got {precision:.2f}"
    assert abs(recall - expected_recall) < 0.01, f"Expected recall {expected_recall:.2f}, got {recall:.2f}"
    assert abs(f1 - expected_f1) < 0.01, f"Expected F1 {expected_f1:.2f}, got {f1:.2f}"
    print("  [OK] Evidence precision/recall computed correctly")

    print("\n" + "="*80)
    print("[SUCCESS] All FEVER scorer tests passed!")
    print("="*80)
    print("\nFEVER Score Implementation Summary:")
    print("  - Label Accuracy: Measures only label prediction correctness")
    print("  - FEVER Score: Requires correct label + evidence (for SUPPORTS/REFUTES)")
    print("  - Evidence P/R/F1: Measures evidence retrieval quality")
    print("="*80)


if __name__ == "__main__":
    test_fever_score()
