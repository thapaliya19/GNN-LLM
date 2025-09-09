import torch
import torch.nn as nn

def embed_text(text: str, model: nn.Module) -> torch.Tensor:
    """Encode text using sentence-bert model"""
    return model.encode(text)