import json
import math
import os
import yaml
import torch
import os.path as osp
from pathlib import Path

import torch.nn.functional as F
from typing import List, Sequence, Tuple, Optional, Dict, Any, Union
from sentence_transformers import SentenceTransformer

from src.model.prompt_buffer import PromptBuffer
from src.data.labels import label_name
from src.utils.consts import TTA_NEW_PROMPT_TEMPLATE, TTA_PROMPT_TEMPLATE
from src.utils.utils import get_device_from_model
from src.utils.eval import entropy
from src.utils.consts import CORA_ID2LABEL, PUBMED_ID2LABEL, AIRPORT_ID2LABEL


_TEMPLATES = None
_EX_BLOCK_CFG: Optional[Dict[str, Any]] = None

JSONLike = Union[str, os.PathLike, Path, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


def _load_templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        path = osp.join(osp.dirname(__file__), "templates.yaml")
        with open(path, "r") as f:
            _TEMPLATES = yaml.safe_load(f)["templates"]
    return _TEMPLATES


def _load_ex_block_cfg(yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load + normalize example-block config YAML once.
    """
    global _EX_BLOCK_CFG
    if _EX_BLOCK_CFG is not None and yaml_path is None:
        return _EX_BLOCK_CFG

    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "example_blocks.yaml")
    with open(yaml_path, "r") as f:
        cfg_raw = yaml.safe_load(f)

    defaults = cfg_raw.get("defaults", {}) or {}
    templates_raw = cfg_raw.get("example_block_templates", {}) or {}

    # Normalize each template into a dict with keys: body, include_history, history_attach, history (optional)
    templates: Dict[str, Dict[str, Any]] = {}
    for name, val in templates_raw.items():
        if isinstance(val, str):
            t = {"body": val}
        else:
            t = dict(val)
        t.setdefault("include_history", defaults.get("include_history", False))
        t.setdefault("history_attach", defaults.get("history_attach", "per_example"))
        # merge history defaults (max_steps, header, item)
        hist_default = defaults.get("history", {}) or {}
        hist_local = dict(hist_default)
        hist_local.update(t.get("history", {}) or {})
        t["history"] = hist_local
        templates[name] = t

    _EX_BLOCK_CFG = {"templates": templates, "defaults": defaults}
    return _EX_BLOCK_CFG


# ---------- HISTORY RENDERING (YAML-driven) ----------
def _format_exemplar_history_yaml(
    nid: int,
    hist_dict: Dict[int, List[Dict[str, Any]]],
    dataset: str,
    hist_cfg: Dict[str, Any],
) -> str:
    """
    Render history for a single node using YAML config.
    hist_cfg expects keys: max_steps, header, item
    hist_dict[nid] = list of dicts with keys like:
      'iter', 'text_description', 'pred_label', 'true_label', 'top_prob', 'entropy'
    """
    if nid not in hist_dict or len(hist_dict[nid]) == 0:
        return "  History: (no prior iterations logged)\n"

    max_steps = int(hist_cfg.get("max_steps", 3))
    header = hist_cfg.get("header", "History:")
    item_tpl = hist_cfg.get("item", " - Iter {iter}: pred={pred_label_name} ({correctness}), top_prob={top_prob:.2f}, entropy={entropy:.3f}")

    H = hist_dict[nid][-max_steps:]  # most recent few
    lines = [f"  {header}"]
    for h in reversed(H):
        true_name = label_name(dataset, h["true_label"])
        pred_name = label_name(dataset, h["pred_label"])
        correctness = "correct" if h["pred_label"] == h["true_label"] else "wrong"
        # Safe defaults
        top_prob = float(h.get("top_prob", float("nan")))
        if math.isnan(top_prob):
            top_prob = 0.0
        entropy = float(h.get("entropy", float("nan")))
        if math.isnan(entropy):
            entropy = 0.0
        text_desc = h.get("text_description", "(no text)")

        line = item_tpl.format(
            iter=h.get("iter", "?"),
            text_description=text_desc,
            pred_label_name=pred_name,
            true_label_name=true_name,
            correctness=correctness,
            top_prob=top_prob,
            entropy=entropy,
        )
        lines.append(line)
    # Ensure trailing newline
    s = "\n".join(lines)
    if not s.endswith("\n"):
        s += "\n"
    return s

def format_target_history_block(
    nid: int,
    hist_dict: Dict[int, List[Dict[str, Any]]],
    dataset: str,
    max_steps: Optional[int] = None,
) -> str:
    """
    Build a short history block for the *target* node.

    Summary focuses on:
      - GNN prediction and correctness,
      - top predicted probability,
      - predictive entropy,
      - and the description text used in each past iteration.
    """
    if hist_dict is None:
        return ""
    if nid not in hist_dict or len(hist_dict[nid]) == 0:
        return ""

    records = hist_dict[nid]
    if max_steps is not None:
        records = records[-max_steps:]  # keep most recent few

    lines: List[str] = []
    lines.append("### Prediction history for this target node (most recent first)")

    for h in reversed(records):  # newest first
        true_name = label_name(dataset, h["true_label"])
        pred_name = label_name(dataset, h["pred_label"])
        correctness = "correct" if h["pred_label"] == h["true_label"] else "wrong"
        top_prob = float(h.get("top_prob", 0.0))
        entropy_val = float(h.get("entropy", 0.0))
        text_desc = h.get("text_description", "")
        iter_id = h.get("iter", "?")

        lines.append(
            f"- Iteration {iter_id}: GNN prediction = {pred_name} ({correctness}), "
            f"top probability = {top_prob:.3f}, predictive entropy = {entropy_val:.3f}."
        )
        if text_desc:
            lines.append(f"  Description used in this iteration: {text_desc}")

    return "\n".join(lines) + "\n"


# ---------- EXAMPLE BLOCK RENDERING ----------
def format_example_block(
    demo_nodes: Sequence[int],
    meta_by_node: Dict[int, Dict[str, Any]],
    dataset: str,
    hist: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    hist_max_steps: Optional[int] = None,  # kept for backward-compat; overrides YAML if given
    *,
    template_name: str = "default",
    yaml_path: Optional[str] = None,
    keyphrases_by_node: Optional[Dict[int, list]] = None,
) -> str:
    """
    Build the exemplars string for the prompt from YAML template + settings.

    Placeholders supported in template body:
      {item_number}, {node_idx}, {original_description}, {topological_description},
      {gt_label}, {gnn_pred}, {class_probs}

    History is controlled by YAML:
      include_history: bool
      history_attach: per_example | after_block | none
      history: {max_steps, header, item}
    """
    cfg = _load_ex_block_cfg(yaml_path)
    templates = cfg["templates"]
    if template_name not in templates:
        raise ValueError(f"Unknown example block template '{template_name}'. Available: {list(templates)}")
    tcfg = templates[template_name]
    body_tpl: str = tcfg["body"]
    include_history: bool = bool(tcfg.get("include_history", False))
    attach_mode: str = tcfg.get("history_attach", "per_example")
    hist_cfg: Dict[str, Any] = dict(tcfg.get("history", {}))

    # CLI override for max steps if provided
    if hist_max_steps is not None:
        hist_cfg["max_steps"] = hist_max_steps

    # Build examples
    lines: List[str] = []
    per_example_hist: List[str] = []

    for j, nid in enumerate(demo_nodes, start=1):
        m = meta_by_node[nid]

        # class probability string
        probs_by_class = m.get("probs_by_class", {})
        class_probs = ", ".join([f"{cls}: {prob:.3f}" for cls, prob in probs_by_class.items()])

        # topo summary (try common keys; fall back gracefully)
        topo = (
            m.get("topological_description")
            or m.get("topo")
            or m.get("topo_features")
            or "N/A"
        )

        # Use keyphrases instead of raw text when available
        if keyphrases_by_node and nid in keyphrases_by_node:
            from src.neighbors.keyphrases import format_keyphrases
            orig_desc = format_keyphrases(keyphrases_by_node[nid])
        else:
            orig_desc = m.get("original_text", "")

        fields = {
            "item_number": j,
            "node_idx": nid,
            "original_description": orig_desc,
            "topological_description": topo,
            "gt_label": label_name(dataset, m["true_label"]),
            "gnn_pred": label_name(dataset, m["pred_label"]),
            "class_probs": class_probs,
        }
        # Render example body
        ex_str = body_tpl.format(**fields)
        lines.append(ex_str)

        # Optional per-example history
        if include_history and attach_mode == "per_example" and hist is not None:
            lines.append(_format_exemplar_history_yaml(nid, hist, dataset, hist_cfg))

        # Collect for after-block (if needed)
        if include_history and attach_mode == "after_block" and hist is not None:
            per_example_hist.append(_format_exemplar_history_yaml(nid, hist, dataset, hist_cfg))

    # After block history
    if include_history and attach_mode == "after_block" and per_example_hist:
        lines.append("\n".join(per_example_hist))

    return "\n".join(lines)


def _allowed_classes_str(dataset: str) -> str:
    if dataset == "cora":
        names = [CORA_ID2LABEL[i] for i in sorted(CORA_ID2LABEL)]
    elif dataset == "pubmed":
        names = [PUBMED_ID2LABEL[i] for i in sorted(PUBMED_ID2LABEL)]
    else:
        names = [AIRPORT_ID2LABEL[i] for i in sorted(AIRPORT_ID2LABEL)]
    # bullet or comma list both work; bullets are clearer to the model
    return "\n".join(f"- {n}" for n in names)

def _default_topk_for_dataset(dataset: str) -> int:
    # Mirror TANS: cora -> 3; pubmed -> 1 (you can override via params later)
    return 3 


def render_prompt(
    template_name: str,
    graph_type: str,
    current: str,
    demos: str,
    topological_description: str,
    target_history_block: str = "",
    class_hint_prefix: str = "",
    class_hint_suffix: str = "",
    neighbor_context: Optional[str] = None,
    classes: str = "",
    top_k_classes: int = 3,
) -> str:
    """Render a named template with the given inputs."""
    templates = _load_templates()
    # Auto-select neighbor template when neighbor_context is provided
    if neighbor_context and template_name == "default":
        template_name = "default_with_neighbors"
    if template_name not in templates:
        raise ValueError(
            f"xTemplate '{template_name}' not found in templates. Available: {list(templates.keys())}"
        )
    template = templates[template_name]
    fmt_kwargs = dict(
        graph_type=graph_type,
        original_description=current,
        examples_block=demos,
        topological_description=topological_description,
        target_history_block=target_history_block,
        class_hint_prefix=class_hint_prefix,
        class_hint_suffix=class_hint_suffix,
        classes=classes,
        top_k_classes=top_k_classes,
    )
    if neighbor_context is not None:
        fmt_kwargs["neighbor_context"] = neighbor_context
    return template.format(**fmt_kwargs)


def build_full_prompt_for_node_with_template(
    node_idx,
    data,
    examples_block,
    dataset,
    template_name: str,
    target_history_block: str = "",
    neighbor_context: Optional[str] = None,
    top_k_classes: int = 3,
):
    graph_type = (
        "citation network" if dataset in ["cora", "pubmed"] else "airport network"
    )
    topological_description = data.topo_descriptions[node_idx]
    current = data.text_descriptions[node_idx]
    demos_text = examples_block

    # Class list for classification instruction
    classes = _allowed_classes_str(dataset)

    # Airport-specific guidance about class semantics
    if dataset in ["usa", "europe", "brazil"]:
        class_hint_prefix = (
            "Classes represent different levels of airport activity "
            "(e.g., very low, low, medium, very high), based on how many "
            "connections the airport has and its position in the network."
        )
        class_hint_suffix = (
            "When rewriting the description, especially highlight why this node is more "
            "likely to belong to some classes or activity levels rather than others "
            "(e.g., global hub vs regional hub vs local airport)."
        )
    else:
        class_hint_prefix = ""
        class_hint_suffix = ""

    return render_prompt(
        template_name,
        graph_type,
        current,
        demos_text,
        topological_description,
        target_history_block=target_history_block,
        class_hint_prefix=class_hint_prefix,
        class_hint_suffix=class_hint_suffix,
        neighbor_context=neighbor_context,
        classes=classes,
        top_k_classes=top_k_classes,
    )


def append_history_jsonl(path, iteration, snapshot, keep_idx_set=None):
    """
    Append per-node prediction snapshot to JSONL.

    If keep_idx_set is provided, only nodes whose index is in the set are written.
    If keep_idx_set is None, *all* nodes from the snapshot are stored.

    This file is later used to provide prediction/entropy history for
    the *target* node across rewrite iterations.
    """
    recs = []
    for rec in snapshot:
        nid = rec["node_idx"]
        if keep_idx_set is not None and nid not in keep_idx_set:
            continue
        recs.append({"iter": iteration, **rec})

    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    print(f"[HIST] Appended {len(recs)} records to {path}")


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
                "topological_description": data.topo_descriptions[i],
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


def load_buffer_from_json(
    json_input: JSONLike,
    sbert: SentenceTransformer,
    *,
    device: torch.device = torch.device("cpu"),
    struct_embs: Optional[torch.Tensor] = None,   # [N, Ds] or None
    sim_method: str = "weighted_sum",
) -> Tuple[PromptBuffer, Dict[int, Dict[str, Any]]]:
    """
    Accepts:
      - path to .json or .jsonl
      - list[dict] (records)
      - dict with records under 'records'/'data'/'items'/'examples'/'train_items'/'rows'/'entries'/'samples'
      - dict-of-dicts (values are records)
      - list[str] where each str is a JSON/dict literal line
    Each record must have (with alias support):
      node_idx  (aliases: 'nid','node','index','id')
      true_label (aliases: 'gt_label','label','y_true','target')
      original_text (aliases: 'text','desc','description','node_text','node_description')
    Optional keys:
      pred_label (aliases: 'pred','prediction','y_pred')
      probs_by_class (aliases: 'class_probs','probs','probabilities')
      topological_description (aliases: 'topo','topo_features','topological_summary')
    """
    REQUIRED = {"node_idx", "true_label", "original_text"}

    KEY_ALIASES = {
        "node_idx": ["node_idx", "nid", "node", "index", "id"],
        "true_label": ["true_label", "gt_label", "label", "y_true", "target"],
        "original_text": ["original_text", "text", "desc", "description", "node_text", "node_description"],
        "pred_label": ["pred_label", "pred", "prediction", "y_pred"],
        "probs_by_class": ["probs_by_class", "class_probs", "probs", "probabilities"],
        "topological_description": ["topological_description", "topo", "topo_features", "topological_summary"],
    }

    PREFERRED_LIST_KEYS = ("records","data","items","examples","train_items","rows","entries","samples")

    def _coerce_records(obj: Any) -> List[Any]:
        # direct list
        if isinstance(obj, list):
            return obj
        # dict wrappers
        if isinstance(obj, dict):
            # 1) try well-known keys
            for k in PREFERRED_LIST_KEYS:
                v = obj.get(k, None)
                if isinstance(v, list):
                    return v
            # 2) generic: any value that is list[dict]-like
            for v in obj.values():
                if isinstance(v, list) and len(v) and all(isinstance(e, dict) for e in v):
                    return v
            # 3) dict-of-dicts
            if obj and all(isinstance(v, dict) for v in obj.values()):
                return list(obj.values())
            # 4) fall back to single “record” (may still be a wrapper; later we’ll error if fields missing)
            return [obj]
        # scalar → wrap
        return [obj]

    def _parse_json_or_literal(s: str) -> Any:
        # try JSON
        try:
            return json.loads(s)
        except Exception:
            pass
        # try Python literal (handles single quotes)
        try:
            import ast
            return ast.literal_eval(s)
        except Exception:
            return s  # leave as-is

    def _ensure_dict(r: Any) -> Dict[str, Any]:
        if isinstance(r, dict):
            return r
        if isinstance(r, str):
            parsed = _parse_json_or_literal(r)
            if isinstance(parsed, dict):
                return parsed
        raise TypeError(f"Record is not a dict or parseable string: {type(r)} — {str(r)[:120]}...")

    def _get_with_aliases(rec: Dict[str, Any], logical_key: str, *, required: bool, default: Any=None):
        for k in KEY_ALIASES[logical_key]:
            if k in rec:
                return rec[k]
        if required:
            raise KeyError(f"Missing required field '{logical_key}' (aliases: {KEY_ALIASES[logical_key]}) in record: {str(rec)[:200]}")
        return default

    # ---- Load input into a list of raw records (possibly strings) ----
    recs_raw: List[Any] = []
    if isinstance(json_input, (str, os.PathLike, Path)):
        path = str(json_input)
        # First try as a single JSON (wrapper or array). If that fails, treat as JSONL.
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
            recs_raw = _coerce_records(loaded)
        except Exception:
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # lines may be json objects or dict-literals; store raw and parse later
            recs_raw = lines
    elif isinstance(json_input, (list, dict)):
        recs_raw = _coerce_records(json_input)
    else:
        raise TypeError("json_input must be a path, list, or dict")

    # ---- Normalize to dicts and (if still wrapper) drill again ----
    recs_norm: List[Dict[str, Any]] = []
    for r in recs_raw:
        recs_norm.append(_ensure_dict(r))

    # If the first normalized element still looks like a wrapper (missing required fields),
    # attempt one more drilling pass into any list-of-dicts it contains.
    def _looks_like_record(d: Dict[str, Any]) -> bool:
        return any(alias in d for alias in KEY_ALIASES["node_idx"]) and \
               any(alias in d for alias in KEY_ALIASES["true_label"]) and \
               any(alias in d for alias in KEY_ALIASES["original_text"])

    if recs_norm and not _looks_like_record(recs_norm[0]):
        drilled = None
        # 1) prefer known keys if present
        for k in PREFERRED_LIST_KEYS:
            if isinstance(recs_norm[0].get(k), list) and recs_norm[0][k]:
                cand = recs_norm[0][k]
                if all(isinstance(x, dict) for x in cand):
                    drilled = cand
                    break
        # 2) otherwise scan all values for list[dict]
        if drilled is None:
            for v in recs_norm[0].values():
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    drilled = v
                    break
        if drilled is not None:
            recs_norm = drilled

    if not recs_norm:
        raise ValueError("No records found in buffer JSON input.")

    # ---- Build arrays + meta ----
    node_idx: List[int] = []
    labels:   List[int] = []
    texts:    List[str] = []
    meta_by_node: Dict[int, Dict[str, Any]] = {}

    for i, r in enumerate(recs_norm):
        try:
            nid = int(_get_with_aliases(r, "node_idx", required=True))
            tl  = _get_with_aliases(r, "true_label", required=True)
            # allow strings like "3" or class names mapping elsewhere; here we coerce numeric if possible
            try:
                tl = int(tl)
            except Exception:
                # if it’s a string label, keep as-is; downstream label_name can handle mapping if needed
                pass
            txt = str(_get_with_aliases(r, "original_text", required=True))
        except KeyError as e:
            # show available keys and a hint if we hit a wrapper again
            raise KeyError(f"Record #{i} missing fields: {e}. Available keys: {list(r.keys())}") from None

        node_idx.append(nid)
        labels.append(tl if isinstance(tl, int) else int(r.get("true_label_numeric", 0)))
        texts.append(txt)

        meta_by_node[nid] = {
            "original_text": txt,
            "true_label": tl if isinstance(tl, int) else r.get("true_label_numeric", tl),
            "pred_label": _get_with_aliases(r, "pred_label", required=False, default=tl),
            "probs_by_class": _get_with_aliases(r, "probs_by_class", required=False, default={}),
            "topological_description": _get_with_aliases(r, "topological_description", required=False, default=None),
        }

    # ---- Encode texts ----
    text_vecs = [sbert.encode(t) for t in texts]
    T = torch.tensor(text_vecs, dtype=torch.float32)

    # ---- struct rows for the selected node_idx ----
    if struct_embs is not None:
        idx_tensor = torch.tensor(node_idx, dtype=torch.long)
        S = struct_embs[idx_tensor]  # [M, Ds]
    else:
        S = None

    buf = PromptBuffer.from_tensors(
        node_idx=node_idx,
        labels=labels,
        text_descs=texts,
        text_embs=T,
        struct_embs=S,
        device=device,
        sim_method=sim_method,
    )
    return buf, meta_by_node