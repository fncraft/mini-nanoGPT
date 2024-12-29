from torch.optim import AdamW

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """
        Creates and returns an AdamW optimizer for the model, optimizing only parameters that require gradients.
        
        Args:
            model: PyTorch model to be optimized
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate for optimization
            betas: Tuple of (beta1, beta2) for AdamW optimizer
            device_type: Target device ('cuda' or 'cpu') - reserved for future extensions
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer
