import torch
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec

from src.utils.utils import get_device


def train_node2vec(graph, params):
    device = get_device(params)
    print("Training Node2Vec for structural embeddings…")
    node2vec = Node2Vec(
        graph.edge_index,
        embedding_dim=params["struct_dim"],
        walk_length=params["struct_walk_length"],
        context_size=params["struct_context_size"],
        walks_per_node=params["struct_walks_per_node"],
        num_negative_samples=params["struct_neg_samples"],
        p=params.get("struct_p", 1.0),
        q=params.get("struct_q", 1.0),
        sparse=True,
    ).to(device)

    loader = node2vec.loader(
        batch_size=params.get("struct_batch_size", 128), shuffle=True
    )
    optim_n2v = torch.optim.SparseAdam(
        node2vec.parameters(), lr=params.get("struct_lr", 1e-3)
    )

    for epoch in range(params.get("struct_epochs", 5)):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optim_n2v.zero_grad()
            loss_n2v = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss_n2v.backward()
            optim_n2v.step()
            total_loss += loss_n2v.item()
        print(
            f"  Node2Vec epoch {epoch+1}/{params['struct_epochs']}, loss={total_loss:.4f}"
        )
    print("Node2Vec training complete.")

    # detach and store on CPU
    struct_embs = F.normalize(
        node2vec.embedding.weight.detach().cpu(), p=2, dim=1
    )
    return struct_embs