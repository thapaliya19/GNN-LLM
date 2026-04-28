"""
Neighbor selection strategies for including 1-hop graph neighbor
information in LLM prompts. Based on insights from Huang et al., AAAI 2024.
"""

import math
import random
from typing import List, Optional

import numpy as np


def _cosine_similarities(target_emb: np.ndarray, neighbor_embs: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single target and multiple neighbors."""
    target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-10)
    norms = np.linalg.norm(neighbor_embs, axis=1, keepdims=True) + 1e-10
    neighbor_normed = neighbor_embs / norms
    return neighbor_normed @ target_norm


def select_neighbors(
    target_idx: int,
    one_hop_neighbors: List[int],
    embeddings: np.ndarray,
    k: int = 5,
    strategy: str = "random",
    mmr_lambda: float = 0.7,
) -> List[int]:
    """Select k neighbors from 1-hop neighborhood using specified strategy.

    Args:
        target_idx: Index of the target node.
        one_hop_neighbors: List of 1-hop neighbor node indices.
        embeddings: [N, D] precomputed embeddings for all nodes.
        k: Number of neighbors to select.
        strategy: One of "random", "similar", "diverse", "mixed".
        mmr_lambda: Lambda for MMR (only used with "diverse" strategy).

    Returns:
        List of selected neighbor node indices.
    """
    if len(one_hop_neighbors) == 0:
        return []
    if len(one_hop_neighbors) <= k:
        return list(one_hop_neighbors)

    neighbors = list(one_hop_neighbors)

    if strategy == "random":
        return random.sample(neighbors, k)

    target_emb = embeddings[target_idx]
    neighbor_embs = embeddings[neighbors]
    sims = _cosine_similarities(target_emb, neighbor_embs)

    if strategy == "similar":
        top_k_idx = np.argsort(sims)[::-1][:k]
        return [neighbors[i] for i in top_k_idx]

    if strategy == "diverse":
        # MMR: Maximal Marginal Relevance
        selected_indices = []
        candidate_mask = np.ones(len(neighbors), dtype=bool)

        # Start with most similar
        first = int(np.argmax(sims))
        selected_indices.append(first)
        candidate_mask[first] = False

        for _ in range(k - 1):
            if not candidate_mask.any():
                break
            cand_idx = np.where(candidate_mask)[0]
            # Relevance: sim to target
            relevance = sims[cand_idx]
            # Redundancy: max sim to any already selected
            selected_embs = neighbor_embs[selected_indices]
            cand_embs = neighbor_embs[cand_idx]
            cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-10
            sel_norms = np.linalg.norm(selected_embs, axis=1, keepdims=True) + 1e-10
            sim_matrix = (cand_embs / cand_norms) @ (selected_embs / sel_norms).T
            max_redundancy = sim_matrix.max(axis=1)
            # MMR score
            mmr_scores = mmr_lambda * relevance - (1 - mmr_lambda) * max_redundancy
            best = cand_idx[int(np.argmax(mmr_scores))]
            selected_indices.append(best)
            candidate_mask[best] = False

        return [neighbors[i] for i in selected_indices]

    if strategy == "mixed":
        # Half most similar, half most dissimilar
        n_similar = math.ceil(k / 2)
        n_dissimilar = k - n_similar

        sorted_idx = np.argsort(sims)[::-1]
        similar_idx = sorted_idx[:n_similar].tolist()

        # Most dissimilar from remaining
        remaining_idx = sorted_idx[n_similar:]
        # Reverse to get least similar first
        dissimilar_idx = remaining_idx[::-1][:n_dissimilar].tolist()

        selected = similar_idx + dissimilar_idx
        return [neighbors[i] for i in selected]

    raise ValueError(f"Unknown neighbor selection strategy: {strategy}")


def _truncate_sentences(text: str, max_sentences: int = 2) -> str:
    """Truncate text to first max_sentences sentences."""
    if not text:
        return text
    parts = text.split('. ')
    if len(parts) <= max_sentences:
        return text
    truncated = '. '.join(parts[:max_sentences])
    if not truncated.endswith('.'):
        truncated += '.'
    return truncated


def build_neighbor_context_text(
    target_idx: int,
    selected_neighbor_indices: List[int],
    node_texts: List[str],
    topo_descriptions: List[str],
    is_text_rich: bool,
    max_sentences: int = 2,
    keyphrases_by_node: Optional[dict] = None,
    neighbor_format: str = "raw",
) -> str:
    """Format selected neighbors into a prompt section.

    For text-rich (Cora/Pubmed): uses node text descriptions (raw or keyphrases).
    For text-free (airports): uses topological feature summaries.

    Args:
        target_idx: Target node index (unused but kept for API consistency).
        selected_neighbor_indices: List of neighbor node indices to include.
        node_texts: All node text descriptions.
        topo_descriptions: All topological descriptions.
        is_text_rich: Whether the dataset has meaningful text (citation vs airport).
        max_sentences: Max sentences to keep per neighbor description.
        keyphrases_by_node: Dict mapping node_id -> list of keyphrases (optional).
        neighbor_format: "raw" for truncated text, "keyphrases" for extracted keyphrases.

    Returns:
        Formatted string of neighbor descriptions.
    """
    if not selected_neighbor_indices:
        return "No connected neighbors found."

    from .keyphrases import format_keyphrases

    lines = []
    for i, nid in enumerate(selected_neighbor_indices, 1):
        if is_text_rich and neighbor_format == "keyphrases" and keyphrases_by_node:
            kws = keyphrases_by_node.get(nid, [])
            lines.append(f"Connected node {i} keywords: {format_keyphrases(kws)}")
        elif is_text_rich:
            text = node_texts[nid] if nid < len(node_texts) else ""
            text = _truncate_sentences(text, max_sentences)
            lines.append(f"Connected node {i}: {text}")
        else:
            text = topo_descriptions[nid] if nid < len(topo_descriptions) else ""
            text = _truncate_sentences(text, max_sentences)
            lines.append(f"Connected node {i}: {text}")

    return "\n".join(lines)


def build_one_hop_from_edge_index(edge_index, num_nodes: int) -> dict:
    """Build 1-hop neighbor dict from PyG edge_index tensor.

    Args:
        edge_index: [2, E] tensor of edges.
        num_nodes: Total number of nodes.

    Returns:
        Dict mapping node_idx -> list of neighbor indices.
    """
    one_hop = {i: [] for i in range(num_nodes)}
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for s, d in zip(src, dst):
        one_hop[int(s)].append(int(d))
    # Deduplicate
    for k in one_hop:
        one_hop[k] = list(set(one_hop[k]))
    return one_hop
