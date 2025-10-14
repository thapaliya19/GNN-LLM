# src/embeddings/structural.py
from __future__ import annotations
import torch
from torch.optim import SparseAdam
from torch_geometric.nn import Node2Vec

# ---------- shared training loop for Node2Vec-like models ----------
def _train_node2vec_like(edge_index: torch.Tensor,
                         num_nodes: int,
                         *,
                         embedding_dim: int = 128,
                         walk_length: int = 20,
                         context_size: int = 10,
                         walks_per_node: int = 10,
                         p: float = 1.0,
                         q: float = 1.0,
                         epochs: int = 20,
                         batch_size: int = 128,
                         lr: float = 1e-3,
                         device: str | torch.device = "cpu",
                         sparse: bool = True) -> torch.Tensor:
    """
    Trains a skip-gram model on random walks (Node2Vec). Setting p=q=1.0 recovers DeepWalk behavior.
    Returns a [num_nodes, embedding_dim] float32 tensor on CPU.
    """
    dev = torch.device(device)
    edge_index = edge_index.to(dev)
    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p, q=q,
        num_negative_samples=1,
        sparse=sparse,
    ).to(dev)

    loader = model.loader(batch_size=batch_size, shuffle=True)
    opt = SparseAdam(list(model.parameters()), lr=lr)

    model.train()
    for _ in range(int(epochs)):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            opt.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            opt.step()
            total_loss += float(loss)
        # optional: print(f"[struct] epoch loss {total_loss:.4f}")

    with torch.no_grad():
        embs = model.embedding.weight.detach().cpu().to(torch.float32)
    return embs


# ---------- existing function (kept for backward compatibility) ----------
def train_node2vec(data, params) -> torch.Tensor:
    """Your original structural encoder."""
    return _train_node2vec_like(
        data.edge_index, data.num_nodes,
        embedding_dim=int(params.get("struct_emb_dim", 128)),
        walk_length=int(params.get("walk_length", 20)),
        context_size=int(params.get("context_size", 10)),
        walks_per_node=int(params.get("walks_per_node", 10)),
        p=float(params.get("p", 1.0)),
        q=float(params.get("q", 1.0)),
        epochs=int(params.get("struct_epochs", params.get("epochs", 20))),
        batch_size=int(params.get("struct_batch_size", 128)),
        lr=float(params.get("struct_lr", 1e-3)),
        device="cpu",
        sparse=True,
    )


# ---------- DeepWalk (wrapper around Node2Vec with p=q=1.0) ----------
def train_deepwalk(data, params) -> torch.Tensor:
    """
    DeepWalk is the unbiased random-walk version of Node2Vec.
    """
    # ignore any p/q passed in and force p=q=1.0
    return _train_node2vec_like(
        data.edge_index, data.num_nodes,
        embedding_dim=int(params.get("struct_emb_dim", 128)),
        walk_length=int(params.get("walk_length", 40)),    # DW often uses longer walks
        context_size=int(params.get("context_size", 10)),
        walks_per_node=int(params.get("walks_per_node", 10)),
        p=1.0, q=1.0,
        epochs=int(params.get("struct_epochs", params.get("epochs", 20))),
        batch_size=int(params.get("struct_batch_size", 128)),
        lr=float(params.get("struct_lr", 1e-3)),
        device="cpu",
        sparse=True,
    )


# ---------- struc2vec (via karateclub) ----------
def train_struc2vec(data, params) -> torch.Tensor:
    """
    Role/structural-identity embeddings using karateclub's Struc2Vec.
    Requires: pip install karateclub==1.3.3
    """
    try:
        import networkx as nx
        from karateclub import Struc2Vec
    except Exception as e:
        raise ImportError(
            "struc2vec requires 'karateclub' (pip install karateclub) and networkx."
        ) from e

    # convert PyG graph to undirected NetworkX
    ei = data.edge_index.cpu()
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    # each column is an edge (u,v)
    g.add_edges_from(ei.t().tolist())

    # hyperparams (mirrored to your naming)
    dim = int(params.get("struct_emb_dim", 128))
    walk_len = int(params.get("walk_length", 10))      # struc2vec default is small; tune as needed
    num_walks = int(params.get("walks_per_node", 10))
    workers = int(params.get("struct_workers", 4))
    window_size = int(params.get("context_size", 10))

    model = Struc2Vec(
        dimensions=dim,
        walk_length=walk_len,
        num_walks=num_walks,
        workers=workers,
        window_size=window_size,
        # remaining params keep defaults (OPT1/OPT2 hierarchy, until-layer, etc.)
    )
    model.fit(g)
    X = model.get_embedding()  # numpy [N, dim]
    return torch.tensor(X, dtype=torch.float32)  # CPU


# ---------- one-line dispatcher you can call everywhere ----------
def train_structural_embeddings(data, params) -> torch.Tensor:
    """
    Select structural encoder by params['struct_method']:
      'node2vec' (default) | 'deepwalk' | 'struc2vec'
    """
    method = str(params.get("struct_method", "node2vec")).lower()
    if method == "node2vec":
        return train_node2vec(data, params)
    if method == "deepwalk":
        return train_deepwalk(data, params)
    if method in ("struc2vec", "struct2vec"):
        return train_struc2vec(data, params)
    raise ValueError(f"Unknown struct_method='{method}'")
