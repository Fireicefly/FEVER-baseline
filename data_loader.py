"""
Data loading utilities for FEVER dataset.
"""

import json
from typing import List, Dict, Any


class FEVERDataLoader:
    """Load and parse FEVER dataset JSONL files."""

    LABEL_MAP = {
        "SUPPORTS": 0,
        "REFUTES": 1,
        "NOT ENOUGH INFO": 2
    }

    LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Load a JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    @classmethod
    def load_dataset(cls, file_path: str, include_evidence: bool = True) -> List[Dict[str, Any]]:
        """
        Load FEVER dataset and parse claims.

        Args:
            file_path: Path to JSONL file
            include_evidence: Whether to include evidence in the output

        Returns:
            List of parsed claim dictionaries
        """
        raw_data = cls.load_jsonl(file_path)
        parsed_data = []

        for item in raw_data:
            claim_id = item.get('id')
            claim_text = item.get('claim', '')
            label = item.get('label', 'NOT ENOUGH INFO')

            parsed_item = {
                'id': claim_id,
                'claim': claim_text,
                'label': label,
                'label_id': cls.LABEL_MAP.get(label, 2)
            }

            if include_evidence and 'evidence' in item:
                # Parse evidence: list of evidence sets
                # Each evidence set is a list of [annotation_id, evidence_id, page_title, sent_id]
                evidence_sets = item['evidence']
                parsed_item['evidence'] = evidence_sets

            parsed_data.append(parsed_item)

        return parsed_data

    @staticmethod
    def extract_evidence_pages(evidence_sets: List[List[List]]) -> List[str]:
        """
        Extract unique page titles from evidence sets.

        Args:
            evidence_sets: Evidence from FEVER dataset

        Returns:
            List of unique page titles
        """
        pages = set()
        for evidence_set in evidence_sets:
            for evidence in evidence_set:
                if len(evidence) >= 3:
                    page_title = evidence[2]
                    if page_title:
                        pages.add(page_title)
        return list(pages)

    @staticmethod
    def extract_evidence_sentences(evidence_sets: List[List[List]]) -> List[tuple]:
        """
        Extract (page_title, sentence_id) pairs from evidence sets.

        Args:
            evidence_sets: Evidence from FEVER dataset

        Returns:
            List of (page_title, sentence_id) tuples
        """
        sentences = set()
        for evidence_set in evidence_sets:
            for evidence in evidence_set:
                if len(evidence) >= 4:
                    page_title = evidence[2]
                    sent_id = evidence[3]
                    if page_title and sent_id is not None:
                        sentences.add((page_title, sent_id))
        return list(sentences)
