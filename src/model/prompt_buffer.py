from typing import List, Tuple
import torch
import torch.nn.functional as F

class PromptBuffer:
    """
    Fixed‑size FIFO buffer storing (node_idx, text, text_emb).
    """

    def __init__(self, max_size: int = 10):
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = max_size
        # each entry: (node_idx: int, text: str, text_emb: Tensor on CPU)
        self.buffer: List[Tuple[int, str, torch.Tensor]] = []
        print(f"PromptBuffer initialized (max_size={max_size})")

    def add(self, node_idx: int, text: str, text_emb: torch.Tensor):
        """Add new entry; evict oldest if over capacity."""
        self.buffer.append((node_idx, text, text_emb.detach().cpu()))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_k_text(self, query_emb: torch.Tensor, k: int) -> List[str]:
        """Top‑k by cosine similarity in text embedding space."""
        sims = []
        for nid, txt, emb in self.buffer:
            sim = F.cosine_similarity(
                query_emb.unsqueeze(0),
                emb.to(query_emb.device).unsqueeze(0),
                dim=1
            ).item()
            sims.append((nid, sim, txt))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [(nid,sim, txt) for nid, sim,txt in sims[:k]]

    def get_k_struct(self, query_idx: int, struct_embs: torch.Tensor, k: int) -> List[str]:
        """Top‑k by cosine similarity in structural embedding space."""
        q_emb = struct_embs[query_idx].to(torch.float32)
        sims = []
        for n_idx, txt, _ in self.buffer:
            s_emb = struct_embs[n_idx].to(q_emb.device)
            sim = F.cosine_similarity(q_emb.unsqueeze(0), s_emb.unsqueeze(0), dim=1).item()
            sims.append((n_idx, sim, txt))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [(nid,sim, txt) for nid, sim, txt in sims[:k]]
