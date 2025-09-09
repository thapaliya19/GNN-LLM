import torch.nn.functional as F

from src.utils.utils import get_device_from_model


def train_gnn(model, data, split, optimizer, params):
    model.train()
    device = get_device_from_model(model)

    train_mask = split["train"]
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y[train_mask].squeeze().to(device)

    z = model.encode(x, edge_index)[train_mask]
    y_pred = model.predict(z)

    loss = F.cross_entropy(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()