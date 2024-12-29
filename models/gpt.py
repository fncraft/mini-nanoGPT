import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    """
    Configuration class for the GPT model that defines essential hyperparameters including:
    - vocab_size: size of the vocabulary
    - block_size: maximum context length
    - number of layers, attention heads, embedding dimension, dropout rate,
          and whether to use bias terms in linear layers
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

##############################################################################
# A minimal GPT-like model example
##############################################################################

class GPT(nn.Module):
    """
    A minimalist GPT implementation that contains only the essential components:
    - token_embedding_table: Maps token indices to vectors
    - lm_head: Projects the output vectors back to vocabulary size for next token prediction
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # token_embedding_table: [vocab_size, n_embd]
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # lm_head: [n_embd, vocab_size]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model
        
        Args:
            idx: [batch_size, time] - Input token indices
            targets: [batch_size, time] - Target tokens for training (optional) 
        """
        b, t = idx.size()
        # Convert token indices into embeddings
        token_emb = self.token_embedding_table(idx)
        # Project embeddings back to vocabulary space
        logits = self.lm_head(token_emb)
    
        # Calculate cross entropy loss if targets are provided 
        loss = None
        if targets is not None:
            logits_view = logits.view(b * t, -1)    # Flatten to shape: [b*t, vocab_size]
            targets_view = targets.view(b * t)       # Flatten to shape: [b*t]
            loss = F.cross_entropy(logits_view, targets_view)
    
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates new tokens sequentially based on the input context (prompt).
        
        Args:
            idx: Input token sequence
            max_new_tokens: Number of tokens to generate
            temperature: Controls randomness in generation. Higher values produce more diverse outputs
            top_k: If set, only keeps the top k tokens from probability distribution
        """
        for _ in range(max_new_tokens):
            if idx.size(1) == 0:
                raise ValueError("Input sequence is empty. Please provide at least one token.")
                
            # Only use the last block_size tokens as context
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
    
            # Get logits from the last timestep and apply temperature scaling
            logits = logits[:, -1, :] / temperature
    
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                top_value = v[:, -1].unsqueeze(-1)
                logits[logits < top_value] = -float('Inf')
    
            # Convert logits to probability distribution
            probs = F.softmax(logits, dim=-1)
            # Sample next token based on the probabilities
            idx_next = torch.multinomial(probs, num_samples=1) 
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
    
        return idx

    def crop_block_size(self, block_size):
        """
        Crops the block size to the specified length.
        """
        self.config.block_size = block_size
        self.block_size = block_size
