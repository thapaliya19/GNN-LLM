import torch
import torch.nn.functional as F
from src.utils.eval import dataset2metric, evaluate


from src.utils.utils import get_device_from_model


def eval_gnn(model, data, split, params):
    """ Evaluate the GNN model, returns accuracy on train/val/test splits
    """
    model.eval()
    device = get_device_from_model(model)

    train_mask, val_mask, test_mask = split["train"], split["val"], split["test"]

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    # forward
    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    # get per-class probabilities and confidences
    probs = F.softmax(y_pred, dim=-1)
    confs, preds = torch.max(probs, dim=-1)  # preds is the argmax class
    confs = confs.cpu().tolist()
    preds = preds.cpu().tolist()
    y_true = y.cpu().tolist()

    # get all train indices
    train_idx = train_mask.nonzero(as_tuple=False).squeeze().tolist()
    if isinstance(train_idx, int):
        train_idx = [train_idx]

    # select only the *correct* ones
    true_train_idx = [i for i in train_idx if preds[i] == y_true[i]]

    # sort those by confidence (descending)
    true_train_idx_sorted = sorted(true_train_idx, key=lambda i: confs[i], reverse=True)

    # now pull out the sorted confidences & descriptions
    true_train_confs = [confs[i] for i in true_train_idx_sorted]
    true_train_descs = [data.text_descriptions[i] for i in true_train_idx_sorted]

    # compute overall metrics
    train_value = evaluate(y_pred, y, train_mask, params)
    val_value = evaluate(y_pred, y, val_mask, params)
    test_value = evaluate(y_pred, y, test_mask, params)

    return {
        "train": train_value,
        "val": val_value,
        "test": test_value,
        "metric": dataset2metric[params["data"]],
        # NEW: correct (true) train‐predictions, sorted by confidence
        "true_train_indices": true_train_idx_sorted,
        "true_train_confs": true_train_confs,
        "true_train_descs": true_train_descs,
    }
