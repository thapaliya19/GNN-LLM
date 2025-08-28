#!/usr/bin/env python3
"""
struct_text_homophily.py

Compute whether topologically similar nodes also have similar text descriptions
by correlating Node2Vec cosine‐similarity with SBERT‐based text cosine‐similarity.
"""

import os
import argparse

import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.datasets import Airports
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import Node2Vec

from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr


def compute_struct_embs(data, args):
    """Train Node2Vec on `data` and return an [N × D] numpy array."""
    device = torch.device('cpu')
    node2vec = Node2Vec(
        data.edge_index,
        embedding_dim=args.struct_dim,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        num_negative_samples=args.neg_samples,
        p=args.p, q=args.q,
        sparse=True
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


def compute_text_embs(texts, model_name="all-MiniLM-L6-v2", batch_size=128):
    """Encode a list of text descriptions into an [N × D] numpy array via SBERT."""
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Compare structural vs. text similarity in Airports graphs"
    )
    parser.add_argument("--dataset", choices=["brazil","europe","usa", "cora"], required=True,
                        help="Which Airports split to load")
    parser.add_argument("--data-root", default="datasets",
                        help="Root folder containing `dataset/airports`")
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
    parser.add_argument("--sbert-model",   type=str,
                        default="all-MiniLM-L12-v2",
                        help="SentenceTransformer model for text embeddings")
    parser.add_argument("--sbert-batch",   type=int, default=64)
    parser.add_argument("--desc-csv", type=str, default="data/airports_descriptions.tsv",
                        help="Path to CSV file with text descriptions (tab-separated)")
    args = parser.parse_args()

    # 1) Load graph
    path = os.path.join(args.data_root, "dataset", "airports")
    data = Airports(root=path, name=args.dataset)[0]
    data = ToUndirected()(data)
    print(f"[Data] Loaded {args.dataset}: {data.num_nodes} nodes, {data.num_edges} edges")

    # 2) Read or build text descriptions: assumes you have data.text_descriptions
    #    If you store descriptions in a CSV, load here instead.
    #    We'll try to read `text_descriptions` attribute; if missing, abort.
    try:
            desc_csv = args.desc_csv
            df = pd.read_csv(desc_csv, sep='\t')
            data.text_descriptions = df['answer'].tolist()
            texts = data.text_descriptions  # list of length N
    except AttributeError:
        raise RuntimeError("Dataset object has no `text_descriptions`. "
                           "Please attach your node text list as `data.text_descriptions`.")

    # 3) Compute embeddings
    S = compute_struct_embs(data, args)                         # [N, D_struct]
    T = compute_text_embs(texts, args.sbert_model, args.sbert_batch)  # [N, D_text]

    # 4) Compute cosine-similarity matrices
    print("[Compute] pairwise cosine-similarities...")
    struct_sim = 1.0 - pairwise_distances(S, metric="cosine")
    text_sim   = 1.0 - pairwise_distances(T, metric="cosine")

    # 5) Flatten upper-triangle and compute Pearson r
    N = S.shape[0]
    iu = np.triu_indices(N, k=1)
    struct_flat = struct_sim[iu]
    text_flat   = text_sim[iu]

    r, pval = pearsonr(struct_flat, text_flat)
    print()
    print("=== Structural vs. Text Similarity ===")
    print(f"Pearson r = {r:.3f}, p-value = {pval:.3e}")

    # 6) Print top-10 highly-structurally-similar pairs and their text similarity
    topk = 10
    order = np.argsort(-struct_flat)[:topk]
    print(f"\nTop {topk} pairs by structural similarity:")
    for rank, idx in enumerate(order, 1):
        i, j = iu[0][idx], iu[1][idx]
        print(f" {rank:2d}. Nodes ({i},{j}) — struct={struct_flat[idx]:.4f}, text={text_flat[idx]:.4f}")
        print(f"     → \"{texts[i]}\"")
        print(f"     → \"{texts[j]}\"")

if __name__ == "__main__":
    main()
