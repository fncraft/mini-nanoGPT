"""
trainer/optim.py

This file provides functions related to optimizer configuration.
Currently, it has the `configure_optimizers` function which returns an AdamW 
optimizer for training the GPT model.
"""

import torch
from torch.optim import AdamW


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """
    Creates and returns an AdamW optimizer for the model's parameters, 
    ignoring those that do not require gradients.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer
