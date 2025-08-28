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
