# ----- retrieval_helpers.py -----
import torch
import torch.nn.functional as F

def mmr_select(query_vec, cand_mat, k, lambda_mmr=0.7):
    """
    Maximal Marginal Relevance selection on cosine similarity.
    query_vec: [D]      (torch.Tensor)
    cand_mat:  [N, D]   (torch.Tensor)
    Returns: list of indices (length k)
    """
    sim_q = F.cosine_similarity(cand_mat, query_vec.unsqueeze(0), dim=1)  # [N]
    selected = []
    candidates = torch.arange(cand_mat.size(0))
    while len(selected) < k and candidates.numel() > 0:
        if len(selected) == 0:
            # pick most relevant first
            i = torch.argmax(sim_q[candidates]).item()
            selected.append(candidates[i].item())
            candidates = torch.cat([candidates[:i], candidates[i+1:]])
        else:
            S = torch.stack([cand_mat[idx] for idx in selected], dim=0)      # [|S|, D]
            sim_div = torch.max(torch.matmul(cand_mat[candidates], S.T), dim=1).values # [|C|]
            # normalize both terms
            rel = sim_q[candidates]
            mmr = lambda_mmr * rel - (1 - lambda_mmr) * sim_div
            i = torch.argmax(mmr).item()
            selected.append(candidates[i].item())
            candidates = torch.cat([candidates[:i], candidates[i+1:]])
    return selected

def get_k_retrieval_mmr(query_idx, R, y, k, pos_ratio=0.5, lambda_mmr=0.7):
    """
    Class-balanced + MMR demos from retriever space.
    R: [N, D] retriever embeddings (CPU tensor)
    y: [N] long labels (CPU)
    """
    qv = R[query_idx]
    yq = int(y[query_idx])
    pos_mask = (y == yq)
    neg_mask = ~pos_mask

    # split budget
    k_pos = max(1, int(k * pos_ratio))
    k_neg = max(1, k - k_pos)

    # candidate pools (exclude query itself)
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
    pos_idx = pos_idx[pos_idx != query_idx]
    neg_idx = torch.nonzero(neg_mask, as_tuple=False).view(-1)

    pos_sel = []
    if pos_idx.numel() > 0:
        pos_sel = mmr_select(qv, R[pos_idx], min(k_pos, pos_idx.numel()), lambda_mmr=lambda_mmr)
        pos_sel = pos_idx[pos_sel].tolist()

    neg_sel = []
    if neg_idx.numel() > 0:
        neg_sel = mmr_select(qv, R[neg_idx], min(k_neg, neg_idx.numel()), lambda_mmr=lambda_mmr)
        neg_sel = neg_idx[neg_sel].tolist()

    sel = pos_sel + neg_sel
    return sel[:k]


import math
import torch

def _softmax_weights(scores: torch.Tensor, tau: float) -> torch.Tensor:
    s = (scores / max(tau, 1e-6)).float()
    s = s - s.max()  # numeric stability
    w = torch.exp(s)
    return w / (w.sum() + 1e-12)

def _nucleus_filter(weights: torch.Tensor, p: float) -> torch.Tensor:
    # return mask of the smallest prefix whose cumulative mass >= p
    w_sorted, idx = torch.sort(weights, descending=True)
    csum = torch.cumsum(w_sorted, dim=0)
    keep = csum <= p
    if not torch.any(keep):
        keep = torch.ones_like(w_sorted, dtype=torch.bool)
    last = torch.nonzero(keep, as_tuple=False).view(-1)
    if last.numel() > 0:
        last_idx = last[-1].item()
    else:
        last_idx = 0
    mask_sorted = torch.zeros_like(w_sorted, dtype=torch.bool)
    mask_sorted[: last_idx + 1] = True
    # invert sort
    mask = torch.zeros_like(weights, dtype=torch.bool)
    mask[idx] = mask_sorted
    return mask

def _sample_without_replacement(weights: torch.Tensor, k: int) -> list[int]:
    # Simple Gumbel-top-k on weights
    eps = 1e-20
    g = -torch.log(-torch.log(torch.rand_like(weights).clamp(min=eps)))
    keys = torch.log(weights.clamp(min=eps)) + g
    topk = torch.topk(keys, k=min(k, weights.numel()), largest=True).indices
    return topk.tolist()

def probabilistic_select(scores: torch.Tensor, k: int, tau: float, mode: str, top_p: float = 0.9) -> list[int]:
    """
    scores: [N] (higher is better). Returns selected indices.
    mode ∈ {"softmax", "nucleus"}.
    """
    w = _softmax_weights(scores, tau)
    if mode == "nucleus":
        mask = _nucleus_filter(w, top_p)
        w = w * mask
        w = w / (w.sum() + 1e-12)
    idx = _sample_without_replacement(w, k)
    return idx

def stochastic_mmr_scores(q_vec: torch.Tensor, cand_mat: torch.Tensor, selected: list[int], lam: float = 0.7):
    """
    Compute MMR scores for candidates given selected set; returns [N] scores.
    """
    sim_q = torch.nn.functional.cosine_similarity(cand_mat, q_vec.unsqueeze(0), dim=1)  # [N]
    if len(selected) == 0:
        return sim_q
    S = cand_mat[selected]  # [|S|, D]
    # redundancy as max similarity to already selected
    red = torch.max(torch.nn.functional.normalize(cand_mat) @ torch.nn.functional.normalize(S).T, dim=1).values
    return lam * sim_q - (1.0 - lam) * red

def probabilistic_mmr(q_vec: torch.Tensor, cand_mat: torch.Tensor, k: int, tau: float, lam: float = 0.7) -> list[int]:
    """
    Stochastic MMR: iteratively sample 1 item with softmax over MMR scores.
    """
    N = cand_mat.size(0)
    all_idx = list(range(N))
    chosen = []
    while len(chosen) < min(k, N):
        mmr = stochastic_mmr_scores(q_vec, cand_mat, chosen, lam)
        w = _softmax_weights(mmr, tau)
        pick = _sample_without_replacement(w, 1)[0]
        if pick not in chosen:
            chosen.append(pick)
    return chosen
