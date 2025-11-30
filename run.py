"""
Main entry point for FEVER baseline.

This script provides a simple interface to download data, train, and evaluate.
"""

import argparse
import os
import sys

import config


def download_data():
    """Download FEVER dataset."""
    from download_data import download_fever_data
    download_fever_data()


def train_model():
    """Train the FEVER baseline model."""
    from train import main as train_main
    train_main()


def evaluate_model(model_path=None, data_path=None, output_path=None):
    """Evaluate the FEVER baseline model."""
    from evaluate import evaluate_model

    if model_path is None:
        model_path = f"{config.MODEL_DIR}/best_model.pt"
    if data_path is None:
        data_path = config.DEV_FILE

    evaluate_model(model_path, data_path, output_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="FEVER Baseline 2018 - Fact Extraction and VERification"
    )

    parser.add_argument(
        'action',
        choices=['download', 'train', 'evaluate', 'all'],
        help="Action to perform"
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Path to model checkpoint (for evaluation)"
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help="Path to evaluation data (for evaluation)"
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Path to save predictions (for evaluation)"
    )

    args = parser.parse_args()

    print("="*80)
    print("FEVER Baseline 2018")
    print("Fact Extraction and VERification")
    print("="*80)
    print()
    print("Configuration:")
    print(f"  - Download Wikipedia: {config.DOWNLOAD_WIKIPEDIA}")
    print(f"  - Num Docs Retrieved: {config.NUM_DOCS_RETRIEVED}")
    print(f"  - Num Sentences Retrieved: {config.NUM_SENTENCES_RETRIEVED}")
    print(f"  - Embedding Dim: {config.EMBEDDING_DIM}")
    print(f"  - Projection Dim: {config.PROJECTION_DIM}")
    print(f"  - Hidden Dim: {config.HIDDEN_DIM}")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Learning Rate: {config.LEARNING_RATE}")
    print(f"  - Epochs: {config.NUM_EPOCHS}")
    print("="*80)
    print()

    if args.action == 'download':
        download_data()

    elif args.action == 'train':
        # Check if data exists
        if not os.path.exists(config.TRAIN_FILE):
            print("Training data not found. Downloading...")
            download_data()

        train_model()

    elif args.action == 'evaluate':
        evaluate_model(args.model, args.data, args.output)

    elif args.action == 'all':
        # Download, train, and evaluate
        if not os.path.exists(config.TRAIN_FILE):
            print("Step 1: Downloading data...")
            download_data()
        else:
            print("Step 1: Data already downloaded, skipping.")

        print("\nStep 2: Training model...")
        train_model()

        print("\nStep 3: Evaluating model...")
        evaluate_model()


if __name__ == "__main__":
    main()
