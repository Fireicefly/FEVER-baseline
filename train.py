"""
Training script for FEVER baseline model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

import config
from data_loader import FEVERDataLoader
from wiki_processor import WikipediaProcessor, MockWikipediaProcessor
from retrieval import EvidenceRetrieval, OracleEvidenceRetrieval
from tokenizer import Vocabulary
from dataset import FEVERDataset, collate_fn
from model import FEVERModel


def prepare_evidence_text(evidence_sentences):
    """
    Concatenate evidence sentences into a single text.

    Args:
        evidence_sentences: List of (page_id, sent_id, sent_text) tuples

    Returns:
        Concatenated evidence text
    """
    if not evidence_sentences:
        return "No evidence found."

    texts = [sent_text for _, _, sent_text in evidence_sentences]
    return ' '.join(texts)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        claims = batch['claim'].to(device)
        evidence = batch['evidence'].to(device)
        claim_masks = batch['claim_mask'].to(device)
        evidence_masks = batch['evidence_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits = model(claims, evidence, claim_masks, evidence_masks)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")

        for batch in progress_bar:
            claims = batch['claim'].to(device)
            evidence = batch['evidence'].to(device)
            claim_masks = batch['claim_mask'].to(device)
            evidence_masks = batch['evidence_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(claims, evidence, claim_masks, evidence_masks)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    """Main training function."""
    print("="*80)
    print("FEVER Baseline Training (2018)")
    print("="*80)

    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading training data...")
    train_data = FEVERDataLoader.load_dataset(config.TRAIN_FILE)
    print(f"Loaded {len(train_data)} training examples.")

    print("\nLoading dev data...")
    dev_data = FEVERDataLoader.load_dataset(config.DEV_FILE)
    print(f"Loaded {len(dev_data)} dev examples.")

    # Initialize Wikipedia processor
    if config.DOWNLOAD_WIKIPEDIA and os.path.exists(config.WIKI_DIR):
        print("\nLoading Wikipedia pages...")
        wiki_processor = WikipediaProcessor()
        wiki_processor.load_wikipedia_pages()

        # Use full evidence retrieval
        evidence_retriever = EvidenceRetrieval(wiki_processor)
        evidence_retriever.build_index()
    else:
        print("\nUsing mock Wikipedia processor (oracle mode)...")
        wiki_processor = MockWikipediaProcessor()

        # Use oracle evidence retrieval
        evidence_retriever = OracleEvidenceRetrieval(wiki_processor)

    # Retrieve evidence for all training examples
    print("\nRetrieving evidence for training data...")
    train_evidence = []
    for item in tqdm(train_data, desc="Train evidence"):
        if isinstance(evidence_retriever, OracleEvidenceRetrieval):
            evidence = evidence_retriever.retrieve_evidence(
                item['claim'],
                item.get('evidence', [])
            )
        else:
            evidence = evidence_retriever.retrieve_evidence(item['claim'])

        evidence_text = prepare_evidence_text(evidence)
        train_evidence.append(evidence_text)

    print("\nRetrieving evidence for dev data...")
    dev_evidence = []
    for item in tqdm(dev_data, desc="Dev evidence"):
        if isinstance(evidence_retriever, OracleEvidenceRetrieval):
            evidence = evidence_retriever.retrieve_evidence(
                item['claim'],
                item.get('evidence', [])
            )
        else:
            evidence = evidence_retriever.retrieve_evidence(item['claim'])

        evidence_text = prepare_evidence_text(evidence)
        dev_evidence.append(evidence_text)

    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab_path = f"{config.MODEL_DIR}/vocab.pkl"

    if os.path.exists(vocab_path):
        vocabulary = Vocabulary.load(vocab_path)
    else:
        vocabulary = Vocabulary(max_vocab_size=50000)

        all_texts = []
        for item in train_data:
            all_texts.append(item['claim'])
        for evidence_text in train_evidence:
            all_texts.append(evidence_text)

        vocabulary.build_vocab(all_texts)

        os.makedirs(config.MODEL_DIR, exist_ok=True)
        vocabulary.save(vocab_path)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = FEVERDataset(train_data, train_evidence, vocabulary)
    dev_dataset = FEVERDataset(dev_data, dev_evidence, vocabulary)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Initialize model
    print("\nInitializing model...")
    model = FEVERModel(
        vocab_size=len(vocabulary),
        embedding_dim=config.EMBEDDING_DIM,
        projection_dim=config.PROJECTION_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    print("\nStarting training...")
    best_dev_acc = 0.0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*80}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.2f}%")

        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            model_path = f"{config.MODEL_DIR}/best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_acc': dev_acc,
            }, model_path)
            print(f"Best model saved to {model_path}")

    print("\n" + "="*80)
    print(f"Training complete! Best dev accuracy: {best_dev_acc:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
