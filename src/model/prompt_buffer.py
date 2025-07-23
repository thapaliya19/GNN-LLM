from typing import List, Callable, Tuple
import torch.nn.functional as F

import torch


class PromptBuffer:
    """
    Fixed‐size FIFO buffer of text_descriptions.
    get_k(text, k, embed_fn) returns the top‐k buffer items
    by cosine similarity to `text`, using `embed_fn(text)->Tensor`.
    """

    def __init__(self, max_size: int = 10):
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        self.max_size = max_size
        self.buffer: List[Tuple[str, torch.Tensor]] = []  # (text, emb)
        print(f"PromptBuffer initialized with max_size={max_size}")

    def add(self, text: str, emb: torch.Tensor):
        """Add new text+embedding; evict oldest if over capacity."""
        self.buffer.append((text, emb.detach().cpu()))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_k(self, query_text: str, query_emb: torch.Tensor, k: int) -> List[List[str]]:
        """
        Return top-k buffer entries by cosine similarity between query_emb and stored emb.
        Each demo is (orig, refined) but here orig==refined==stored text.
        """
        # compute cosine similarities
        sims = []
        for text, emb in self.buffer:
            emb = emb.to(query_emb.device)
            cos = F.cosine_similarity(query_emb.unsqueeze(0), emb.unsqueeze(0)).item()
            sims.append((cos, text))
        # select top-k by similarity
        sims.sort(key=lambda x: x[0], reverse=True)
        top = sims[:k]
        return [t for _, t in top]