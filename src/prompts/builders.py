import json
import torch
import os.path as osp
import torch.nn.functional as F
from typing import List, Tuple

from src.model.prompt_buffer import PromptBuffer
from src.data.labels import label_name
from src.utils.consts import TTA_NEW_PROMPT_TEMPLATE, TTA_PROMPT_TEMPLATE
from src.utils.utils import get_device_from_model
from src.utils.eval import entropy


def build_tta_prompt(old_desc: str, demos: List[Tuple[str, str]]) -> str:
    """Construct the exact prompt text for the LLM (or for logging when using cache)."""
    demos_block = "\n".join(
        [
            f"- Topological information: {d[1]}\n\tLLM generated description: {d[0]}\n"
            for d in demos
        ]
    )
    prompt = TTA_PROMPT_TEMPLATE.format(
        current_node_description=old_desc, demos=demos_block
    )
    return prompt


def format_exemplar_history(nid, hist_dict, dataset, max_steps=3):
    if nid not in hist_dict or len(hist_dict[nid]) == 0:
        return "  History: (no prior iterations logged)\n"
    H = hist_dict[nid][-max_steps:]  # most recent few
    lines = ["  History of predictions for this node in different iterations (most recent first):"]
    for h in reversed(H):
        lines.append(
            f"   - In iter {h['iter']}: \nThe text description used for this node: {h['text_description']} "
            f"Predicted label ={label_name(dataset, h['pred_label'])}, "
            f"which {'is correct' if h['pred_label']==h['true_label'] else 'wrong'} "
            f"top_prob={h['top_prob']:.2f}, entropy={h['entropy']:.3f}, "
        )
    return "\n".join(lines) + "\n"


def format_example_block(
    demo_nodes, meta_by_node, dataset, hist=None, hist_max_steps=3
):
    """
    Build the exemplars string for the prompt. Includes desc, true label, pred label, and per-class probs.
    """
    lines = []
    for j, nid in enumerate(demo_nodes):
        m = meta_by_node[nid]
        # keep probs readable but complete
        probs_str = ", ".join(
            [f"{cls}: {prob:.3f}" for cls, prob in m["probs_by_class"].items()]
        )
        lines.append(
            f"- Example {j+1} | node_idx={nid}\n"
            f"  Description: {m['original_text']}\n"
            f"  TrueLabel: {label_name(dataset, m['true_label'])} "
            f"| GNNPred: {label_name(dataset, m['pred_label'])}\n"
            f"  ClassProbs: {probs_str}"
        )
        if hist is not None:
            lines.append(
                format_exemplar_history(nid, hist, dataset, max_steps=hist_max_steps)
            )

    return "\n".join(lines)

def build_full_prompt_for_node(node_idx, data, examples_block, dataset):
    """
    Prompt includes: original text, topo info for TARGET, and exemplar blocks (with preds/probs).
    """
    topo = (
        data.topo_descriptions[node_idx]
        if hasattr(data, "topo_descriptions")
        else "N/A"
    )
    graph_type = (
        "citation network" if dataset in ["cora", "pubmed"] else "airport network"
    )

    return TTA_NEW_PROMPT_TEMPLATE.format(
        original_description=data.text_descriptions[node_idx],
        topo=topo,
        examples_block=examples_block,
        graph_type=graph_type,
    ).strip()


def append_history_jsonl(path, iteration, snapshot, train_idx_set):
    """Only persist TRAIN nodes for in-context learning history"""
    recs = []
    for rec in snapshot:
        if rec["node_idx"] in train_idx_set:
            recs.append({"iter": iteration, **rec})
    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"[HIST] Appended {len(recs)} training records to {path}")

def load_history(path):
    """Load the history of predictions from JSONL file -- this history is used in prompts for ICL."""
    if not osp.exists(path):
        return {}
    hist = {}
    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)
            nid = int(r["node_idx"])
            hist.setdefault(nid, []).append(r)
    return hist


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
                "text_description": data.text_descriptions[i],
            }
        )
    return out


def dump_train_pred_json(path, model, data, split, dataset):
    """Write train-node predictions & metadata to JSON."""
    preds, probs, ents = per_node_predictions(model, data)
    train_idx = split["train"].nonzero(as_tuple=False).view(-1).tolist()
    items = []
    for i in train_idx:
        # map prob vector -> {class_name: prob}
        prob_map = {label_name(dataset, j): float(p) for j, p in enumerate(probs[i])}
        items.append(
            {
                "node_idx": int(i),
                "original_text": data.text_descriptions[i],
                "true_label": int(data.y[i]),
                "pred_label": int(preds[i]),
                "probs": [float(p) for p in probs[i]],
                "probs_by_class": prob_map,
                "entropy": float(ents[i]),
            }
        )
    blob = {
        "dataset": dataset,
        "num_classes": int(data.y.max().item() + 1),
        "train_items": items,
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[BUFFER] Wrote training predictions to {path} (n={len(items)})")
    return blob  # also return it for immediate use


def load_buffer_from_json(buffer_json, model_sbert, device, max_size=None):
    """
    Fill PromptBuffer with ALL training items from the JSON (text + y) using SBERT text embeddings.
    Returns (PromptBuffer, meta_by_node) where meta_by_node[node_idx] = dict from JSON.
    """
    train_items = buffer_json["train_items"]
    if max_size is None:
        max_size = len(train_items)
    pb = PromptBuffer(max_size=max_size)
    meta_by_node = {}
    for it in train_items:
        idx = it["node_idx"]
        txt = it["original_text"]
        y = it["true_label"]
        emb = torch.tensor(model_sbert.encode(txt), dtype=torch.float32, device=device)
        pb.add(node_idx=idx, label=y, text=txt, text_emb=emb)
        meta_by_node[idx] = it
    print(f"[BUFFER] Loaded {len(train_items)} training items into PromptBuffer.")
    return pb, meta_by_node
