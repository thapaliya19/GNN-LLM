import torch
import torch.nn.functional as F

from src.utils.eval import entropy
from src.utils.utils import get_device_from_model

def per_node_predictions(model, data):
    """Return preds, probs, entropies for ALL nodes (CPU tensors/lists)."""
    device = get_device_from_model(model)
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.edge_index.to(device))
        logits = model.predict(z)
        probs = F.softmax(logits, dim=-1)
    probs_cpu = probs.detach().cpu()
    ent = entropy(probs).detach().cpu()
    confs, preds = torch.max(probs_cpu, dim=-1)
    return preds.tolist(), probs_cpu.tolist(), ent.tolist()


def eval_snapshot(model, data, split, dataset):
    """
    For each node, compute and return:
    - node_idx
    - true_label
    - pred_label
    - entropy (uncertainity) of prediction
    - top_prob: top predicted class probability
    - probs: list of per-class probabilities (probability distribution over predicted classes)
    """
    preds, probs_list, ents = per_node_predictions(model, data)
    y = data.y.view(-1).tolist()
    out = []
    for i, (p, pl, e) in enumerate(zip(preds, probs_list, ents)):
        out.append(
            {
                "node_idx": i,
                "true_label": int(y[i]),
                "pred_label": int(p),
                "entropy": float(e),
                "top_prob": float(max(pl)),
                "probs": [float(x) for x in pl],
            }
        )
    return out