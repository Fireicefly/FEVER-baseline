"""
Evaluation script for FEVER baseline model.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from sklearn.metrics import classification_report, confusion_matrix

import config
from data_loader import FEVERDataLoader
from wiki_processor import WikipediaProcessor, MockWikipediaProcessor
from retrieval import EvidenceRetrieval, OracleEvidenceRetrieval
from tokenizer import Vocabulary
from dataset import FEVERDataset, collate_fn
from model import FEVERModel
from fever_scorer import compute_fever_score, compute_evidence_precision_recall


def prepare_evidence_text(evidence_sentences):
    """Concatenate evidence sentences into a single text."""
    if not evidence_sentences:
        return "No evidence found."

    texts = [sent_text for _, _, sent_text in evidence_sentences]
    return ' '.join(texts)


def save_cached_evidence(evidence_list, evidence_structured_list, cache_path):
    """
    Save retrieved evidence to cache file.
    
    Args:
        evidence_list: List of concatenated evidence text strings
        evidence_structured_list: List of lists of (page_id, sent_id, sent_text) tuples
        cache_path: Path to save cache file
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {
        'evidence_text': evidence_list,
        'evidence_structured': evidence_structured_list
    }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f)
    print(f"Evidence cached to {cache_path}")


def load_cached_evidence(cache_path):
    """
    Load cached evidence if it exists.
    
    Returns:
        Tuple of (evidence_text_list, evidence_structured_list) or (None, None)
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle both old and new cache formats
        if isinstance(data, dict) and 'evidence_text' in data:
            # New format with structured evidence
            return data['evidence_text'], data['evidence_structured']
        else:
            # Old format (just text strings) - return None for structured
            return data, None
    return None, None


def predict(model, dataloader, device):
    """
    Make predictions on a dataset.

    Returns:
        predictions, labels, ids
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Predicting")

        for batch in progress_bar:
            claims = batch['claim'].to(device)
            evidence = batch['evidence'].to(device)
            claim_masks = batch['claim_mask'].to(device)
            evidence_masks = batch['evidence_mask'].to(device)
            labels = batch['label']
            ids = batch['id']

            logits = model(claims, evidence, claim_masks, evidence_masks)
            _, predicted = torch.max(logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_ids.extend(ids)

    return all_predictions, all_labels, all_ids


def evaluate_model(model_path: str, data_file: str, output_file: str = None):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        data_file: Path to evaluation data
        output_file: Optional path to save predictions
    """
    print("="*80)
    print("FEVER Baseline Evaluation (2018)")
    print("="*80)

    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load vocabulary
    vocab_path = f"{config.MODEL_DIR}/vocab.pkl"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Train the model first.")

    vocabulary = Vocabulary.load(vocab_path)

    # Load data
    print(f"\nLoading evaluation data from {data_file}...")
    eval_data = FEVERDataLoader.load_dataset(data_file)
    print(f"Loaded {len(eval_data)} examples.")

    # Initialize Wikipedia processor
    if config.DOWNLOAD_WIKIPEDIA and os.path.exists(config.WIKI_DIR):
        print("\nLoading Wikipedia pages...")
        wiki_processor = WikipediaProcessor()
        wiki_processor.load_wikipedia_pages()

        evidence_retriever = EvidenceRetrieval(wiki_processor, use_drqa=False)
        print(f"\nBuilding/loading TF-IDF index (cache: {config.MODEL_DIR})...")
        evidence_retriever.build_index(cache_dir=config.MODEL_DIR)
    else:
        print("\nUsing mock Wikipedia processor (oracle mode)...")
        wiki_processor = MockWikipediaProcessor()
        evidence_retriever = OracleEvidenceRetrieval(wiki_processor)

    # Retrieve evidence (with caching support)
    print("\nRetrieving evidence...")
    
    # Determine cache filename based on data file
    data_basename = os.path.basename(data_file).replace('.jsonl', '')
    eval_evidence_cache = f"{config.MODEL_DIR}/{data_basename}_evidence_cache.json"
    
    # Try to load cached evidence first
    cached_evidence_text, cached_evidence_structured = load_cached_evidence(eval_evidence_cache)
    
    if cached_evidence_text is not None and len(cached_evidence_text) == len(eval_data):
        print(f"✓ Loaded cached evidence for {len(cached_evidence_text)} examples from {eval_evidence_cache}")
        eval_evidence = cached_evidence_text
        
        if cached_evidence_structured is not None:
            # New cache format with structured evidence
            predicted_evidence_list = cached_evidence_structured
            print("  ✓ Structured evidence loaded. FEVER score computation will be accurate.")
        else:
            # Old cache format without structured evidence
            predicted_evidence_list = []
            print("  Warning: Old cache format detected. FEVER score will be 0%. Re-run to update cache.")
    else:
        if cached_evidence_text is not None:
            print(f"  Cache size mismatch: {len(cached_evidence_text)} cached vs {len(eval_data)} examples. Re-retrieving...")
        
        eval_evidence = []
        predicted_evidence_list = []  # Store for FEVER score computation

        if isinstance(evidence_retriever, OracleEvidenceRetrieval):
            # Oracle mode: must retrieve individually (needs gold evidence per claim)
            for item in tqdm(eval_data, desc="Evidence retrieval"):
                evidence = evidence_retriever.retrieve_evidence(
                    item['claim'],
                    item.get('evidence', [])
                )
                evidence_text = prepare_evidence_text(evidence)
                eval_evidence.append(evidence_text)
                predicted_evidence_list.append(evidence)
        else:
            # Multi-threaded batch retrieval (>10x faster!)
            print("Using multi-threaded batch retrieval...")
            eval_claims = [item['claim'] for item in eval_data]

            # Process in batches with progress bar
            batch_size = 1000  # Process 1000 claims at a time
            for i in tqdm(range(0, len(eval_claims), batch_size), desc="Evidence batches"):
                batch_claims = eval_claims[i:i+batch_size]
                batch_evidence = evidence_retriever.retrieve_evidence_batch(batch_claims)

                for evidence in batch_evidence:
                    evidence_text = prepare_evidence_text(evidence)
                    eval_evidence.append(evidence_text)
                    predicted_evidence_list.append(evidence)
        
        # Cache both text and structured evidence for future runs
        save_cached_evidence(eval_evidence, predicted_evidence_list, eval_evidence_cache)

    # Create dataset
    print("\nCreating dataset...")
    eval_dataset = FEVERDataset(eval_data, eval_evidence, vocabulary)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Load model
    print("\nLoading model...")
    model = FEVERModel(
        vocab_size=len(vocabulary),
        embedding_dim=config.EMBEDDING_DIM,
        projection_dim=config.PROJECTION_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded from {model_path}")

    # Make predictions
    print("\nMaking predictions...")
    predictions, labels, ids = predict(model, eval_loader, device)

    # Calculate metrics
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)

    accuracy = 100.0 * sum([p == l for p, l in zip(predictions, labels)]) / len(labels)
    print(f"\nLabel Accuracy: {accuracy:.2f}%")

    # Compute FEVER score
    gold_evidence_list = [item.get('evidence', []) for item in eval_data]
    fever_score = compute_fever_score(predictions, labels, predicted_evidence_list, gold_evidence_list)
    print(f"FEVER Score: {fever_score * 100:.2f}%")

    # Compute evidence precision/recall
    precision, recall, f1 = compute_evidence_precision_recall(predicted_evidence_list, gold_evidence_list)
    print(f"\nEvidence Retrieval:")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall: {recall * 100:.2f}%")
    print(f"  F1: {f1 * 100:.2f}%")

    print("\nNote:")
    print("  - Label Accuracy: Percentage of correct label predictions (ignoring evidence)")
    print("  - FEVER Score: Official metric - Percentage of correct (label + evidence) predictions")
    print("    * For SUPPORTS/REFUTES: Requires correct label AND at least one correct evidence sentence")
    print("    * For NOT ENOUGH INFO: Only requires correct label")
    print("  - Evidence Precision/Recall: Measures quality of retrieved evidence sentences")

    print("\nClassification Report:")
    print(classification_report(
        labels,
        predictions,
        target_names=FEVERDataLoader.LABEL_NAMES,
        digits=4
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)

    # Save predictions if requested
    if output_file:
        print(f"\nSaving predictions to {output_file}...")
        results = []
        for i, (pred, label, claim_id) in enumerate(zip(predictions, labels, ids)):
            results.append({
                'id': claim_id,
                'predicted_label': FEVERDataLoader.LABEL_NAMES[pred],
                'true_label': FEVERDataLoader.LABEL_NAMES[label],
                'claim': eval_data[i]['claim']
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"Predictions saved!")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FEVER baseline model")
    parser.add_argument(
        '--model',
        type=str,
        default=f"{config.MODEL_DIR}/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        '--data',
        type=str,
        default=config.DEV_FILE,
        help="Path to evaluation data"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Path to save predictions (optional)"
    )

    args = parser.parse_args()

    evaluate_model(args.model, args.data, args.output)
