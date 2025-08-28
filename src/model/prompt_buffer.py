# --- prompt_buffer.py ---
from typing import List, Tuple, NamedTuple
import torch
import torch.nn.functional as F

class BufferEntry(NamedTuple):
    node_idx: int
    label: int
    text: str
    text_emb: torch.Tensor  # CPU

class BufferItemReturned(NamedTuple):
    node_idx: int
    score: float
    cos_t: float
    cos_s: float
    label: int
    text_desc: str

class PromptBuffer:
    def __init__(self, max_size: int = 10):
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = max_size
        self.buffer: List[BufferEntry] = []
        print(f"PromptBuffer initialized (max_size={max_size})")

    def add(self, node_idx: int, label: int, text: str, text_emb: torch.Tensor):
        entry = BufferEntry(node_idx=node_idx, label=int(label), text=text, text_emb=text_emb.detach().cpu())
        self.buffer.append(entry)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def class_hist(self):
        from collections import Counter
        return Counter([e.label for e in self.buffer])

    # existing get_k_text / get_k_struct / get_k_combined:
    # change loops from (nid, txt, emb) to (entry) and access entry.*
    def get_k_text(self, query_emb: torch.Tensor, k: int):
        sims = []
        for e in self.buffer:
            sim = F.cosine_similarity(query_emb.unsqueeze(0), e.text_emb.to(query_emb.device).unsqueeze(0), dim=1).item()
            sims.append((e.node_idx, sim, e.text, e.label))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [(nid, sim, txt) for (nid, sim, txt, _) in sims[:k]]

    def get_k_struct(self, query_idx: int, struct_embs: torch.Tensor, k: int):
        q = struct_embs[query_idx].to(torch.float32)
        sims = []
        for e in self.buffer:
            s = struct_embs[e.node_idx].to(q.device)
            sim = F.cosine_similarity(q.unsqueeze(0), s.unsqueeze(0), dim=1).item()
            sims.append((e.node_idx, sim, e.text, e.label))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [(nid, sim, txt) for (nid, sim, txt, _) in sims[:k]]

    def get_k_combined(self, query_idx: int, text_emb: torch.Tensor, struct_embs: torch.Tensor, k: int, buffer_alpha: float):
        q_struct = struct_embs[query_idx].to(text_emb.device).to(torch.float32)
        sims = []
        for e in self.buffer:
            cos_t = F.cosine_similarity(text_emb.unsqueeze(0), e.text_emb.to(text_emb.device).unsqueeze(0), dim=1).item()
            cos_s = F.cosine_similarity(q_struct.unsqueeze(0), struct_embs[e.node_idx].to(text_emb.device).unsqueeze(0), dim=1).item()
            score = buffer_alpha * cos_t + (1 - buffer_alpha) * cos_s

            sims.append(BufferItemReturned(node_idx=e.node_idx, score=score, cos_t=cos_t, cos_s=cos_s, label=e.label, text_desc=e.text))
        sims.sort(key=lambda x: x.score, reverse=True)
        return sims[:k]
