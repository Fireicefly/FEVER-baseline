"""
Tokenizer and vocabulary builder for FEVER.
"""

import re
from collections import Counter
from typing import List, Dict
import pickle
import os


class SimpleTokenizer:
    """Simple word-level tokenizer."""

    def __init__(self):
        """Initialize tokenizer."""
        self.word_re = re.compile(r'\b\w+\b')

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        text = text.lower()
        tokens = self.word_re.findall(text)
        return tokens


class Vocabulary:
    """Vocabulary for mapping words to indices."""

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    def __init__(self, max_vocab_size: int = 50000):
        """
        Initialize vocabulary.

        Args:
            max_vocab_size: Maximum vocabulary size
        """
        self.max_vocab_size = max_vocab_size
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.word_counts = Counter()
        self.tokenizer = SimpleTokenizer()

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of text strings
        """
        print("Building vocabulary...")

        # Count word frequencies
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            self.word_counts.update(tokens)

        # Get most common words
        most_common = self.word_counts.most_common(self.max_vocab_size - 2)

        # Build word2idx and idx2word
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary built with {len(self.word2idx)} words.")

    def encode(self, text: str, max_length: int = None) -> List[int]:
        """
        Encode text to token indices.

        Args:
            text: Input text
            max_length: Maximum sequence length (truncate if longer, pad if shorter)

        Returns:
            List of token indices
        """
        tokens = self.tokenizer.tokenize(text)
        indices = [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]) for token in tokens]

        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices = indices + [self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices))

        return indices

    def decode(self, indices: List[int]) -> str:
        """
        Decode token indices to text.

        Args:
            indices: List of token indices

        Returns:
            Decoded text
        """
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        return ' '.join(words)

    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'max_vocab_size': self.max_vocab_size
            }, f)
        print(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        vocab = cls(max_vocab_size=data['max_vocab_size'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_counts = data['word_counts']

        print(f"Vocabulary loaded from {path}")
        return vocab

    def __len__(self):
        """Return vocabulary size."""
        return len(self.word2idx)


def create_mask(indices: List[int]) -> List[int]:
    """
    Create padding mask for a sequence.

    Args:
        indices: Token indices

    Returns:
        Mask (1 for real tokens, 0 for padding)
    """
    return [1 if idx != 0 else 0 for idx in indices]
