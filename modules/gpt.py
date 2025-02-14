"""
modules/gpt.py

This file provides the GPTConfig class for storing model hyperparameters,
and the GPT model class itself. Currently a minimal GPT-like model is implemented.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTConfig:
    """
    A configuration class for the GPT model.

    :param vocab_size: The size of the vocabulary.
    :param block_size: The maximum sequence length (context length).
    :param n_layer: Number of transformer layers.
    :param n_head: Number of attention heads.
    :param n_embd: Embedding dimension.
    :param dropout: Dropout rate.
    :param bias: Whether to include bias in linear layers.
    """
    def __init__(
        self, 
        vocab_size, 
        block_size, 
        n_layer, 
        n_head, 
        n_embd, 
        dropout=0.1, 
        bias=False
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


class GPT(nn.Module):
    """
    A minimal GPT-like model consisting of an embedding layer and a linear layer.
    This is primarily for demonstration purposes.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # Token embedding layer
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        # Language modeling head (linear projection to vocab)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        :param idx: Tensor of shape (batch_size, sequence_length) with token indices.
        :param targets: Optional tensor of the same shape for computing cross-entropy loss.
        :return: (logits, loss) tuple. 'logits' is (batch_size, sequence_length, vocab_size).
                 'loss' is a scalar if 'targets' is provided, otherwise None.
        """
        b, t = idx.size()
        # Convert token indices to embeddings
        token_emb = self.token_embedding_table(idx)
        # Project embeddings onto vocab space
        logits = self.lm_head(token_emb)

        loss = None
        if targets is not None:
            # Flatten logits and targets for cross entropy
            logits_view = logits.view(b * t, -1)
            targets_view = targets.view(b * t)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates tokens one by one, appending each new token to 'idx'. 
        Continues until 'max_new_tokens' have been generated.

        :param idx: Initial token indices of shape (batch_size, sequence_length).
        :param max_new_tokens: Number of new tokens to generate.
        :param temperature: Softmax temperature for sampling.
        :param top_k: Top-k cutoff for sampling. If None, no cutoff is applied.
        :return: A tensor of shape (batch_size, sequence_length + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            if idx.size(1) == 0:
                raise ValueError("Input sequence is empty. Provide at least one token.")
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)

            # Scale logits by temperature
            logits = logits[:, -1, :] / temperature

            # If top_k is set, zero out everything except the top_k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                top_value = v[:, -1].unsqueeze(-1)
                logits[logits < top_value] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the new token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def crop_block_size(self, block_size):
        """
        Adjust the model's internal block size (for context length).
        This is useful if you resume training with a smaller block_size than before.
        """
        self.config.block_size = block_size
        self.block_size = block_size
