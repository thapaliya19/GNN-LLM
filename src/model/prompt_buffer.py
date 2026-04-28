# src/model/prompt_buffer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional
import torch
import torch.nn.functional as F


@dataclass
class BufferItem:
    node_idx: int
    label: int
    text_desc: str
    text_emb: torch.Tensor
    struct_emb: Optional[torch.Tensor] = None


@dataclass
class ReturnedItem:
    node_idx: int
    label: int
    text_desc: str

BufferItemReturned = ReturnedItem


class PromptBuffer:
    """
    Stores few-shot demo candidates and retrieves top-k with configurable similarity.

    sim_method:
      - 'text_only'      : cosine(text_q, text_i)
      - 'struct_only'    : cosine(struct_q, struct_i)
      - 'weighted_sum'   : α*cos(text) + (1-α)*cos(struct)
      - 'max'            : max(cos(text), cos(struct))
      - 'product'        : cos(text) * cos(struct)
      - 'concat_cosine'  : cosine([√α*t, √(1-α)*s] , [√α*T, √(1-α)*S])
      - 'random'         : sample k random items from the buffer (respecting masks)
    """

    def __init__(
        self,
        items: Sequence[BufferItem],
        device: torch.device | str = "cpu",
        sim_method: str = "weighted_sum",
        embed_alpha: float = 1.0,   # reserved for symmetry/future use
    ):
        self.device = torch.device(device)
        self.sim_method = sim_method.lower()
        self.embed_alpha = float(embed_alpha)

        # scalar/meta
        self.texts: List[str] = [it.text_desc for it in items]
        self.node_idx = torch.tensor([int(it.node_idx) for it in items], dtype=torch.long)
        self.labels   = torch.tensor([int(it.label)    for it in items], dtype=torch.long)

        # embeddings (L2-normalized)
        T = torch.stack([it.text_emb.detach().to(torch.float32) for it in items], dim=0) if items else torch.empty(0,0)
        self.T = F.normalize(T.to(self.device), p=2, dim=1) if T.numel() else T.to(self.device)

        if items and any(it.struct_emb is None for it in items):
            self.S = None
        elif items:
            S = torch.stack([it.struct_emb.detach().to(torch.float32) for it in items], dim=0)
            self.S = F.normalize(S.to(self.device), p=2, dim=1)
        else:
            self.S = None

        # cache for concat scheme
        self._concat_cache_alpha: Optional[float] = None
        self._concat_cache_tensor: Optional[torch.Tensor] = None

    # ---------- mutation helpers ----------
    def _invalidate_cache(self):
        self._concat_cache_alpha = None
        self._concat_cache_tensor = None

    def add(self, *, node_idx: int, label: int, text_desc: str,
            text_emb: torch.Tensor, struct_emb: Optional[torch.Tensor] = None):
        """
        Append a single item to the buffer (keeps tensors normalized).
        """
        self.texts.append(str(text_desc))
        self.node_idx = torch.cat([self.node_idx, torch.tensor([int(node_idx)], dtype=torch.long, device=self.node_idx.device)])
        self.labels   = torch.cat([self.labels,   torch.tensor([int(label)],   dtype=torch.long, device=self.labels.device)])

        t = F.normalize(text_emb.detach().to(torch.float32).to(self.device).unsqueeze(0), p=2, dim=1)  # [1,Dt]
        if self.T.numel() == 0:
            self.T = t
        else:
            # pad if first time had 0 dims
            if self.T.dim()==2 and self.T.shape[1]==0:
                self.T = t
            else:
                self.T = torch.cat([self.T, t], dim=0)

        if struct_emb is not None:
            s = F.normalize(struct_emb.detach().to(torch.float32).to(self.device).unsqueeze(0), p=2, dim=1)  # [1,Ds]
            if self.S is None:
                self.S = s
            else:
                self.S = torch.cat([self.S, s], dim=0)
        elif self.S is not None:
            # keep lengths consistent: if S exists we must append a row (copy last row as neutral fallback)
            raise ValueError("struct_emb is required because buffer already stores structural embeddings.")
        self._invalidate_cache()

    def add_many(self, items: Sequence[BufferItem]):
        for it in items:
            self.add(node_idx=it.node_idx, label=it.label, text_desc=it.text_desc,
                     text_emb=it.text_emb, struct_emb=it.struct_emb)

    # ---------- config ----------
    def set_method(self, method: str):
        self.sim_method = method.lower()
        self._invalidate_cache()

    @classmethod
    def from_tensors(
        cls,
        node_idx: Sequence[int],
        labels: Sequence[int],
        text_descs: Sequence[str],
        text_embs: torch.Tensor,                 # [M, Dt]
        struct_embs: Optional[torch.Tensor] = None,  # [M, Ds] or None
        **kwargs,
    ) -> "PromptBuffer":
        items: List[BufferItem] = []
        for i in range(len(node_idx)):
            items.append(BufferItem(
                node_idx=int(node_idx[i]),
                label=int(labels[i]),
                text_desc=str(text_descs[i]),
                text_emb=text_embs[i],
                struct_emb=None if struct_embs is None else struct_embs[i],
            ))
        return cls(items, **kwargs)

    # ---------- retrieval ----------
    def _ensure_concat_cache(self, alpha: float):
        if self.S is None:
            raise ValueError("concat_cosine requires structural embeddings in the buffer.")
        if self._concat_cache_tensor is not None and self._concat_cache_alpha == alpha:
            return
        w_t = alpha ** 0.5
        w_s = (1.0 - alpha) ** 0.5
        Z = torch.cat([w_t * self.T, w_s * self.S], dim=1)      # [M, Dt+Ds]
        self._concat_cache_tensor = F.normalize(Z, p=2, dim=1)
        self._concat_cache_alpha = alpha

    @torch.no_grad()
    def get_k_combined(
        self,
        query_idx: int,
        text_emb: torch.Tensor,                      # [Dt]
        struct_embs: Optional[torch.Tensor],         # [N, Ds] or None
        k: int = 5,
        buffer_alpha: float = 0.5,
        sim_method: Optional[str] = None,
        exclude_same_label: bool = False,
    ) -> List[BufferItemReturned]:
        method = (sim_method or self.sim_method).lower()
        alpha = float(buffer_alpha)

        q_t = F.normalize(text_emb.detach().to(torch.float32).to(self.device).unsqueeze(0), p=2, dim=1)  # [1,Dt]
        q_s = None
        if struct_embs is not None:
            q_s = F.normalize(struct_embs[int(query_idx)].detach().to(torch.float32).to(self.device).unsqueeze(0), p=2, dim=1)

        sim_t = (q_t @ self.T.T).squeeze(0) if self.T.numel() else torch.empty(0, device=self.device)

        sim_s = None
        if self.S is not None and q_s is not None:
            sim_s = (q_s @ self.S.T).squeeze(0)

        if method == "text_only" or sim_s is None:
            sims = sim_t
        elif method == "struct_only":
            sims = sim_s
        elif method == "weighted_sum":
            sims = alpha * sim_t + (1.0 - alpha) * sim_s
        elif method == "max":
            sims = torch.maximum(sim_t, sim_s)
        elif method == "product":
            sims = sim_t * sim_s
        elif method == "concat_cosine":
            if q_s is None:
                raise ValueError("concat_cosine requires a structural query embedding.")
            self._ensure_concat_cache(alpha)
            Z = self._concat_cache_tensor
            w_t = alpha ** 0.5
            w_s = (1.0 - alpha) ** 0.5
            zq = torch.cat([w_t * q_t, w_s * q_s], dim=1)
            zq = F.normalize(zq, p=2, dim=1)
            sims = (zq @ Z.T).squeeze(0)
        elif method == "random":
            # We won’t actually use sims for 'random', but keep a placeholder
            sims = sim_t
        else:
            raise ValueError(f"Unknown sim_method '{method}'")

        # mask self
        if (self.node_idx == int(query_idx)).any():
            mask_self = (self.node_idx != int(query_idx))
        else:
            mask_self = torch.ones_like(self.node_idx, dtype=torch.bool)

        if exclude_same_label and (self.node_idx == int(query_idx)).any():
            q_label = self.labels[self.node_idx == int(query_idx)][0]
            mask_label = (self.labels != q_label)
            mask = mask_self & mask_label
        else:
            mask = mask_self

        if method == "random":
            # sample k random valid indices from mask
            valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0 or k <= 0:
                return []

            k_eff = int(min(k, valid_idx.numel()))
            perm = torch.randperm(valid_idx.numel(), device=self.device)[:k_eff]
            topk_idx = valid_idx[perm]
        else:
            sims_masked = sims.masked_fill(~mask, float("-inf"))
            k_eff = int(min(k, sims_masked.numel()))
            if k_eff == 0:
                return []
            _, topk_idx = torch.topk(sims_masked, k=k_eff)

        out: List[BufferItemReturned] = []
        for j in topk_idx.tolist():
            out.append(ReturnedItem(
                node_idx=int(self.node_idx[j].item()),
                label=int(self.labels[j].item()),
                text_desc=self.texts[j],
            ))
        return out
