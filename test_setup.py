"""
Quick test to verify the implementation is working correctly.

This script tests the core components without requiring data download.
"""

import torch
import numpy as np

import config
from tokenizer import Vocabulary, SimpleTokenizer
from model import DecomposableAttention, FEVERModel


def test_tokenizer():
    """Test tokenizer."""
    print("Testing tokenizer...")

    tokenizer = SimpleTokenizer()
    text = "The Eiffel Tower is located in Paris, France."
    tokens = tokenizer.tokenize(text)

    print(f"  Text: {text}")
    print(f"  Tokens: {tokens}")
    assert len(tokens) > 0
    print("  [OK] Tokenizer works!")


def test_vocabulary():
    """Test vocabulary."""
    print("\nTesting vocabulary...")

    vocab = Vocabulary(max_vocab_size=100)

    texts = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "Paris is the capital of France."
    ]

    vocab.build_vocab(texts)

    encoded = vocab.encode("The cat sat", max_length=10)
    decoded = vocab.decode(encoded)

    print(f"  Original: The cat sat")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    print(f"  Vocab size: {len(vocab)}")
    assert len(vocab) > 2  # At least PAD and UNK
    print("  [OK] Vocabulary works!")


def test_model():
    """Test model."""
    print("\nTesting Decomposable Attention model...")

    vocab_size = 1000
    batch_size = 4
    claim_len = 10
    evidence_len = 20

    model = DecomposableAttention(
        vocab_size=vocab_size,
        embedding_dim=300,
        projection_dim=200,
        hidden_dim=200,
        num_classes=3,
        dropout=0.2
    )

    # Create dummy data
    claims = torch.randint(0, vocab_size, (batch_size, claim_len))
    evidence = torch.randint(0, vocab_size, (batch_size, evidence_len))
    claim_masks = torch.ones(batch_size, claim_len, dtype=torch.long)
    evidence_masks = torch.ones(batch_size, evidence_len, dtype=torch.long)

    # Forward pass
    logits = model(evidence, claims, evidence_masks, claim_masks)

    print(f"  Input shapes:")
    print(f"    Claims: {claims.shape}")
    print(f"    Evidence: {evidence.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, 3)")

    assert logits.shape == (batch_size, 3)
    print("  [OK] Model forward pass works!")


def test_fever_model():
    """Test complete FEVER model."""
    print("\nTesting FEVER model...")

    vocab_size = 1000
    batch_size = 2
    claim_len = 15
    evidence_len = 30

    model = FEVERModel(
        vocab_size=vocab_size,
        embedding_dim=300,
        projection_dim=200,
        hidden_dim=200,
        num_classes=3,
        dropout=0.2
    )

    # Create dummy data
    claims = torch.randint(0, vocab_size, (batch_size, claim_len))
    evidence = torch.randint(0, vocab_size, (batch_size, evidence_len))
    claim_masks = torch.ones(batch_size, claim_len, dtype=torch.long)
    evidence_masks = torch.ones(batch_size, evidence_len, dtype=torch.long)

    # Forward pass
    logits = model(claims, evidence, claim_masks, evidence_masks)

    # Get predictions
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)

    print(f"  Logits: {logits}")
    print(f"  Probabilities: {probs}")
    print(f"  Predictions: {predictions}")
    print(f"  Prediction labels: {[['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'][p] for p in predictions]}")

    assert logits.shape == (batch_size, 3)
    assert predictions.shape == (batch_size,)
    print("  [OK] FEVER model works!")


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    print(f"  DOWNLOAD_WIKIPEDIA: {config.DOWNLOAD_WIKIPEDIA}")
    print(f"  NUM_DOCS_RETRIEVED: {config.NUM_DOCS_RETRIEVED}")
    print(f"  NUM_SENTENCES_RETRIEVED: {config.NUM_SENTENCES_RETRIEVED}")
    print(f"  EMBEDDING_DIM: {config.EMBEDDING_DIM}")
    print(f"  PROJECTION_DIM: {config.PROJECTION_DIM}")
    print(f"  HIDDEN_DIM: {config.HIDDEN_DIM}")
    print(f"  BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"  LEARNING_RATE: {config.LEARNING_RATE}")

    assert config.NUM_DOCS_RETRIEVED == 5
    assert config.NUM_SENTENCES_RETRIEVED == 5
    assert config.EMBEDDING_DIM == 300
    assert config.PROJECTION_DIM == 200
    print("  [OK] Configuration is correct!")


def main():
    """Run all tests."""
    print("="*80)
    print("FEVER Baseline - Setup Test")
    print("="*80)

    try:
        test_config()
        test_tokenizer()
        test_vocabulary()
        test_model()
        test_fever_model()

        print("\n" + "="*80)
        print("[SUCCESS] All tests passed!")
        print("="*80)
        print("\nYour FEVER baseline implementation is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python run.py download' to download data")
        print("  2. Run 'python run.py train' to train the model")
        print("  3. Run 'python run.py evaluate' to evaluate the model")
        print("\nOr run 'python run.py all' to do everything at once.")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("[ERROR] Test failed!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
