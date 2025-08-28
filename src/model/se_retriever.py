import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.optim import AdamW

class SERetriever(nn.Module):
    """
    Structure-Enhanced Retriever:
    A lightweight GNN (GraphSAGE) that produces retrieval embeddings for each node.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        # first layer
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        # additional layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        # final projection to retrieval space
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        return self.project(h)


def train_retriever(
    retriever: SERetriever,
    data,
    split: dict,
    params: dict
) -> SERetriever:
    """
    Train the SERetriever with a contrastive objective (positive vs. negative pairs).

    Key fixes vs. previous version:
      * Keep index tensors on CPU when indexing CPU tensors (e.g., data.y).
      * Compute the full-graph embeddings once per step: GNN layers require
        x.shape[0] to match the node ids referenced by edge_index.
      * Slice the resulting embedding matrix with CPU/GPU indices as needed.
    """
    device = next(retriever.parameters()).device
    retriever.train()
    optimizer = AdamW(retriever.parameters(), lr=params.get("retriever_lr", 1e-3))

    # training node indices (CPU LongTensor)
    train_idx = split["train"].nonzero(as_tuple=False).view(-1)
    num_train = train_idx.size(0)

    x_all_cpu = data.x            # CPU tensor
    y_all_cpu = data.y.view(-1)   # CPU labels
    edge_index_dev = data.edge_index.to(device)

    batch_size = params.get("retriever_batch_size", 64)
    temp = params.get("retriever_temp", 0.1)

    for epoch in range(params.get("retriever_epochs", 5)):
        epoch_losses = []
        for _ in range(params.get("retriever_batches_per_epoch", 100)):
            # 1) Sample anchors on CPU (indices stay on CPU to index CPU tensors)
            idx = torch.randint(0, num_train, (batch_size,))
            anchors_cpu = train_idx[idx]
            labels_cpu = y_all_cpu[anchors_cpu]

            # 2) For each anchor, pick one positive & one negative (CPU indices)
            pos_list, neg_list = [], []
            for a_cpu, c_cpu in zip(anchors_cpu.tolist(), labels_cpu.tolist()):
                same = train_idx[(y_all_cpu[train_idx] == c_cpu)]
                diff = train_idx[(y_all_cpu[train_idx] != c_cpu)]

                # Guard against degenerate cases
                if same.numel() == 0:
                    same = train_idx
                if diff.numel() == 0:
                    diff = train_idx

                pos = same[torch.randint(0, same.size(0), (1,))]
                neg = diff[torch.randint(0, diff.size(0), (1,))]
                pos_list.append(pos)
                neg_list.append(neg)

            pos_idx_cpu = torch.cat(pos_list).view(-1)
            neg_idx_cpu = torch.cat(neg_list).view(-1)

            # 3) Move *features* to device and compute full-graph embeddings once
            x_all_dev = x_all_cpu.to(device)
            H = retriever(x_all_dev, edge_index_dev)  # [N, D] on device

            # 4) Build device indices for slicing H
            anchors_dev = anchors_cpu.to(device)
            pos_idx_dev = pos_idx_cpu.to(device)
            neg_idx_dev = neg_idx_cpu.to(device)

            emb_a = H[anchors_dev]
            emb_p = H[pos_idx_dev]
            emb_n = H[neg_idx_dev]

            # 5) Contrastive loss (BCE on logits)
            sim_pos = F.cosine_similarity(emb_a, emb_p, dim=1) / temp
            sim_neg = F.cosine_similarity(emb_a, emb_n, dim=1) / temp
            loss_pos = F.binary_cross_entropy_with_logits(sim_pos, torch.ones_like(sim_pos))
            loss_neg = F.binary_cross_entropy_with_logits(sim_neg, torch.zeros_like(sim_neg))
            loss = 0.5 * (loss_pos + loss_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        print(f"[SERetriever] Epoch {epoch+1}/{params.get('retriever_epochs',5)} - loss: {avg_loss:.4f}")

    return retriever
