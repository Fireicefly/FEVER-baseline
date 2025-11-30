"""
Decomposable Attention Model for Natural Language Inference.

Based on Parikh et al. (2016): "A Decomposable Attention Model for Natural Language Inference"
Adapted for FEVER fact verification task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class FeedForward(nn.Module):
    """Feed-forward neural network with ReLU activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize feed-forward network.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super(FeedForward, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class DecomposableAttention(nn.Module):
    """
    Decomposable Attention Model for NLI.

    Three-step process: Attend, Compare, Aggregate
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        projection_dim: int = 200,
        hidden_dim: int = 200,
        num_classes: int = 3,
        dropout: float = 0.2,
        pretrained_embeddings=None
    ):
        """
        Initialize Decomposable Attention model.

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension (e.g., 300 for GloVe)
            projection_dim: Projection dimension (e.g., 200)
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (3 for FEVER)
            dropout: Dropout rate
            pretrained_embeddings: Pretrained embedding matrix (optional)
        """
        super(DecomposableAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Fix embeddings as in the paper

        # Projection layer (project embeddings from 300 to 200 dimensions)
        self.projection = nn.Linear(embedding_dim, projection_dim)

        # F: Attend function (soft alignment)
        self.attend_f = FeedForward(
            input_dim=projection_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=config.NUM_LAYERS,
            dropout=dropout
        )

        # G: Compare function (element-wise comparison)
        self.compare_g = FeedForward(
            input_dim=projection_dim * 2,  # Concatenate aligned pairs
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=config.NUM_LAYERS,
            dropout=dropout
        )

        # H: Aggregate function (final classification)
        self.aggregate_h = FeedForward(
            input_dim=hidden_dim * 2,  # Concatenate v1 and v2
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=config.NUM_LAYERS,
            dropout=dropout
        )

    def forward(self, premise, hypothesis, premise_mask=None, hypothesis_mask=None):
        """
        Forward pass through the Decomposable Attention model.

        Args:
            premise: Premise token IDs [batch_size, premise_len]
            hypothesis: Hypothesis token IDs [batch_size, hypothesis_len]
            premise_mask: Premise padding mask [batch_size, premise_len]
            hypothesis_mask: Hypothesis padding mask [batch_size, hypothesis_len]

        Returns:
            Logits for each class [batch_size, num_classes]
        """
        batch_size = premise.size(0)

        # Embed and project
        premise_embedded = self.embedding(premise)  # [batch, premise_len, embed_dim]
        hypothesis_embedded = self.embedding(hypothesis)  # [batch, hyp_len, embed_dim]

        premise_proj = self.projection(premise_embedded)  # [batch, premise_len, proj_dim]
        hypothesis_proj = self.projection(hypothesis_embedded)  # [batch, hyp_len, proj_dim]

        # Step 1: Attend
        # Compute soft alignment scores
        premise_attended = self.attend_f(premise_proj)  # [batch, premise_len, hidden]
        hypothesis_attended = self.attend_f(hypothesis_proj)  # [batch, hyp_len, hidden]

        # Compute unnormalized attention scores: e_ij = F(a_i)^T F(b_j)
        scores = torch.bmm(premise_attended, hypothesis_attended.transpose(1, 2))
        # [batch, premise_len, hyp_len]

        # Apply masks if provided
        if premise_mask is not None:
            scores = scores.masked_fill(premise_mask.unsqueeze(2) == 0, -1e9)
        if hypothesis_mask is not None:
            scores = scores.masked_fill(hypothesis_mask.unsqueeze(1) == 0, -1e9)

        # Normalize attention weights
        alpha = F.softmax(scores, dim=2)  # [batch, premise_len, hyp_len]
        beta = F.softmax(scores, dim=1)  # [batch, premise_len, hyp_len]

        # Compute aligned representations
        # β_i = Σ_j exp(e_ij) * b_j / Σ_j exp(e_ij)
        premise_aligned = torch.bmm(alpha, hypothesis_proj)  # [batch, premise_len, proj_dim]

        # α_j = Σ_i exp(e_ij) * a_i / Σ_i exp(e_ij)
        hypothesis_aligned = torch.bmm(beta.transpose(1, 2), premise_proj)  # [batch, hyp_len, proj_dim]

        # Step 2: Compare
        # Concatenate original and aligned representations
        premise_combined = torch.cat([premise_proj, premise_aligned], dim=2)  # [batch, premise_len, 2*proj_dim]
        hypothesis_combined = torch.cat([hypothesis_proj, hypothesis_aligned], dim=2)  # [batch, hyp_len, 2*proj_dim]

        # Apply compare function element-wise
        premise_compared = self.compare_g(premise_combined)  # [batch, premise_len, hidden]
        hypothesis_compared = self.compare_g(hypothesis_combined)  # [batch, hyp_len, hidden]

        # Step 3: Aggregate
        # Sum over sequence dimensions
        if premise_mask is not None:
            premise_compared = premise_compared * premise_mask.unsqueeze(2).float()
        if hypothesis_mask is not None:
            hypothesis_compared = hypothesis_compared * hypothesis_mask.unsqueeze(2).float()

        v1 = premise_compared.sum(dim=1)  # [batch, hidden]
        v2 = hypothesis_compared.sum(dim=1)  # [batch, hidden]

        # Concatenate and classify
        v = torch.cat([v1, v2], dim=1)  # [batch, 2*hidden]
        logits = self.aggregate_h(v)  # [batch, num_classes]

        return logits


class FEVERModel(nn.Module):
    """
    Complete FEVER model combining evidence retrieval and textual entailment.

    This is a wrapper that uses the Decomposable Attention model for classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        projection_dim: int = 200,
        hidden_dim: int = 200,
        num_classes: int = 3,
        dropout: float = 0.2,
        pretrained_embeddings=None
    ):
        """
        Initialize FEVER model.

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            projection_dim: Projection dimension
            hidden_dim: Hidden dimension
            num_classes: Number of classes (3: SUPPORTS, REFUTES, NOT ENOUGH INFO)
            dropout: Dropout rate
            pretrained_embeddings: Pretrained embedding matrix
        """
        super(FEVERModel, self).__init__()

        self.da_model = DecomposableAttention(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings
        )

    def forward(self, claim, evidence, claim_mask=None, evidence_mask=None):
        """
        Forward pass.

        Args:
            claim: Claim token IDs [batch_size, claim_len]
            evidence: Evidence token IDs [batch_size, evidence_len]
            claim_mask: Claim padding mask [batch_size, claim_len]
            evidence_mask: Evidence padding mask [batch_size, evidence_len]

        Returns:
            Logits [batch_size, num_classes]
        """
        return self.da_model(
            premise=evidence,
            hypothesis=claim,
            premise_mask=evidence_mask,
            hypothesis_mask=claim_mask
        )
