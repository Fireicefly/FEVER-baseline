"""
PyTorch Dataset for FEVER.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
from tokenizer import Vocabulary, create_mask


class FEVERDataset(Dataset):
    """PyTorch Dataset for FEVER claims with evidence."""

    def __init__(
        self,
        data: List[Dict],
        evidence_list: List[List[str]],
        vocabulary: Vocabulary,
        max_claim_length: int = 50,
        max_evidence_length: int = 200
    ):
        """
        Initialize FEVER dataset.

        Args:
            data: List of claim dictionaries
            evidence_list: List of evidence strings for each claim
            vocabulary: Vocabulary instance
            max_claim_length: Maximum claim length
            max_evidence_length: Maximum evidence length
        """
        self.data = data
        self.evidence_list = evidence_list
        self.vocabulary = vocabulary
        self.max_claim_length = max_claim_length
        self.max_evidence_length = max_evidence_length

    def __len__(self):
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single item.

        Returns:
            Dictionary with claim, evidence, masks, and label
        """
        item = self.data[idx]
        claim_text = item['claim']
        label = item['label_id']
        evidence_text = self.evidence_list[idx]

        # Encode claim and evidence
        claim_indices = self.vocabulary.encode(claim_text, self.max_claim_length)
        evidence_indices = self.vocabulary.encode(evidence_text, self.max_evidence_length)

        # Create masks
        claim_mask = create_mask(claim_indices)
        evidence_mask = create_mask(evidence_indices)

        return {
            'claim': torch.tensor(claim_indices, dtype=torch.long),
            'evidence': torch.tensor(evidence_indices, dtype=torch.long),
            'claim_mask': torch.tensor(claim_mask, dtype=torch.long),
            'evidence_mask': torch.tensor(evidence_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'id': item['id']
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched tensors
    """
    claims = torch.stack([item['claim'] for item in batch])
    evidence = torch.stack([item['evidence'] for item in batch])
    claim_masks = torch.stack([item['claim_mask'] for item in batch])
    evidence_masks = torch.stack([item['evidence_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    ids = [item['id'] for item in batch]

    return {
        'claim': claims,
        'evidence': evidence,
        'claim_mask': claim_masks,
        'evidence_mask': evidence_masks,
        'label': labels,
        'id': ids
    }
