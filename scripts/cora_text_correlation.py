#!/usr/bin/env python3
"""
cora_struct_text_correlation.py

Load Cora via src.data.data.get_data, compute Node2Vec structural embeddings
and SBERT text embeddings, then compute Pearson correlation between structural
and text similarity.
"""

import os
import argparse
import torch
import numpy as np

from src.data.data import get_data
from src.utils.consts import SBERT_MODELS
from torch_geometric.nn import Node2Vec
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer


def compute_struct_embs(edge_index, args):
    """
    Train Node2Vec on given edge_index and return a [N × D] numpy array.
    """
    device = torch.device('cpu')
    node2vec = Node2Vec(
        edge_index,
        embedding_dim=args.struct_dim,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        num_negative_samples=args.neg_samples,
        p=args.p,
        q=args.q,
        sparse=True,
    ).to(device)

    loader = node2vec.loader(batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Node2Vec] Epoch {epoch}/{args.epochs} — loss {total_loss:.4f}")

    return node2vec.embedding.weight.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Compute structural vs. text similarity on Cora"
    )
    # paths & dataset
    parser.add_argument(
        "--data-path",
        required=True,
        help="Root folder containing dataset/cora/cora.pt",
    )
    parser.add_argument(
        "--desc-csv",
        required=True,
        help="Path to your Cora descriptions TSV (with columns node_idx, answer)",
    )
    parser.add_argument(
        "--topo-csv",
        required=True,
        help="Path to your topo_features CSV (with column topo_features)",
    )
    parser.add_argument(
        "--emb-model",
        default="minilm",
        help="Key for SBERT model in SBERT_MODELS",
    )
    # Node2Vec params
    parser.add_argument("--struct-dim",    type=int,   default=64)
    parser.add_argument("--walk-length",   type=int,   default=20)
    parser.add_argument("--context-size",  type=int,   default=10)
    parser.add_argument("--walks-per-node",type=int,   default=10)
    parser.add_argument("--neg-samples",   type=int,   default=1)
    parser.add_argument("--p",             type=float, default=1.0)
    parser.add_argument("--q",             type=float, default=1.0)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch-size",    type=int,   default=128)
    # SBERT params
    parser.add_argument(
        "--sbert-batch",
        type=int,
        default=64,
        help="Batch size for SBERT encoding",
    )

    args = parser.parse_args()

    # Build params dict for get_data
    params = {
        "data_path": args.data_path,
        "desc_csv": args.desc_csv,
        "topo_csv": args.topo_csv,
        "emb_model": args.emb_model,
    }

    # 1) Load Cora
    data = get_data(params, "cora")
    print(f"[Data] Loaded Cora: {data.num_nodes} nodes, {data.num_edges} edges")

    # 2) Structural embeddings via Node2Vec
    S = compute_struct_embs(data.edge_index, args)     # shape [N, D_struct]

    # 3) Text embeddings via SBERT
    print("[SBERT] Encoding text descriptions…")
    sbert = SentenceTransformer(SBERT_MODELS[args.emb_model])
    T = sbert.encode(
        data.text_descriptions,
        batch_size=args.sbert_batch,
        show_progress_bar=True,
    )
    T = np.asarray(T, dtype=np.float32)               # shape [N, D_text]

    # 4) Pairwise cosine similarities
    print("[Compute] Pairwise cosine similarities…")
    struct_sim = 1.0 - pairwise_distances(S, metric="cosine")
    text_sim   = 1.0 - pairwise_distances(T, metric="cosine")

    # 5) Flatten upper triangle & correlate
    iu = np.triu_indices(data.num_nodes, k=1)
    struct_flat = struct_sim[iu]
    text_flat   = text_sim[iu]

    r, pval = pearsonr(struct_flat, text_flat)
    print("\n=== Structural vs. Text Similarity (Cora) ===")
    print(f"Pearson r = {r:.4f}, p-value = {pval:.3e}")

    # 6) Top-10 structurally similar pairs & their text sim
    print("\nTop 10 pairs by structural similarity:")
    topk = 10
    order = np.argsort(-struct_flat)[:topk]
    for rank, idx in enumerate(order, 1):
        i, j = iu[0][idx], iu[1][idx]
        print(
            f"{rank:2d}. Nodes ({i},{j}) — struct={struct_flat[idx]:.4f}, "
            f"text={text_flat[idx]:.4f}"
        )
        print(f"     → \"{data.text_descriptions[i]}\"")
        print(f"     → \"{data.text_descriptions[j]}\"\n")


if __name__ == "__main__":
    main()
