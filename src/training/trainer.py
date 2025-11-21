import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from src.utils.utils import get_device_from_model


def train_gnn(model, data, split, optimizer, params):
    model.train()
    device = get_device_from_model(model)

    train_mask = split["train"]
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    drop_p = float(params.get("dropedge_p", 0.0))
    if drop_p > 0:
        edge_index, _ = dropout_edge(edge_index, p=drop_p, training=True)
    y = data.y[train_mask].squeeze().to(device)

    z = model.encode(x, edge_index)[train_mask]
    y_pred = model.predict(z)

    loss = F.cross_entropy(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_da(model, data, split, optimizer, params):
    model.train()
    device = get_device_from_model(model)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    loss = F.cross_entropy(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()