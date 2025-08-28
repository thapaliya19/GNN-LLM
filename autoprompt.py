import copy
import json
import math
import os
import os.path as osp
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import Node2Vec

import wandb
from src.data.data import get_data
from src.data.split import get_split
from src.model.llm_client_ollama import LLMClientOllama
from src.model.model import GNN
from src.model.prompt_buffer import PromptBuffer
from src.utils.args import get_args
from src.utils.consts import SBERT_MODELS, TTA_PROMPT_TEMPLATE, DESCRIPTION_TEMPLATE, NODE_TYPES, CORA_ID2LABEL
from src.utils.early_stop import EarlyStopping
from src.utils.eval import dataset2metric, evaluate
from src.utils.logger import Logger
from src.utils.utils import (
    flip_edges,
    get_device,
    get_device_from_model,
    seed_everything,
)
from src.model.se_retriever import SERetriever, train_retriever
from src.model.retrieval_helpers import get_k_retrieval_mmr


def label_name(dataset: str, y_id: int) -> str:
    if dataset == "cora":
        return CORA_ID2LABEL.get(int(y_id), f"unknown({y_id})")
    return str(int(y_id))


def build_tta_prompt(old_desc: str, demos: List[Tuple[str,str]]) -> str:
    """Construct the exact prompt text for the LLM (or for logging when using cache)."""
    demos_block = "\n".join([f"- Topological information: {d[1]}\n\tLLM generated description: {d[0]}\n" for d in demos])
    prompt = TTA_PROMPT_TEMPLATE.format(
        current_node_description=old_desc, demos=demos_block
    )
    return prompt


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


def eval_gnn(model, data, split, params):
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

    # (you can still compute the overall metrics as before)
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


def pred_gnn(
    model: torch.nn.Module,
    data: Union[Data, HeteroData],
    split: Dict[str, torch.Tensor],
    params: Dict[str, Any],
) -> List[Tuple[int, str, str, int, int, float]]:
    """
    Return up to top-K highest-entropy test nodes (after applying entropy threshold).
    Selection is performed on CPU to avoid device mismatch.
    If params['tta_topk'] <= 0, no cap is applied (uses all above threshold, sorted by entropy).
    """
    model.eval()
    device = get_device_from_model(model)

    # Forward on the model's device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    with torch.no_grad():
        z = model.encode(x, edge_index)
        y_pred = model.predict(z)
        probs = F.softmax(y_pred, dim=-1)

    # Move everything needed for selection to CPU (prevents device mismatch)
    entropies_cpu = entropy(probs).detach().cpu()
    confs_cpu, preds_cpu = torch.max(probs.detach().cpu(), dim=-1)
    y_cpu = y.detach().cpu()

    test_mask = split["test"]
    test_mask_cpu = test_mask.detach().cpu() if hasattr(test_mask, "device") else test_mask

    # Indices of test nodes (CPU)
    test_indices = test_mask_cpu.nonzero(as_tuple=False).view(-1)
    if test_indices.numel() == 0:
        return []

    # Entropies of test nodes (CPU)
    test_entropies = entropies_cpu[test_indices]

    # Filter by threshold, still CPU
    thr = float(params["entropy_thr"])
    cand_mask = test_entropies > thr
    if cand_mask.sum().item() == 0:
        return []

    cand_indices = test_indices[cand_mask]
    cand_entropies = test_entropies[cand_mask]

    # Apply top-k cap (if provided)
    k = int(params.get("tta_topk", 0) or 0)
    if k > 0 and cand_entropies.numel() > k:
        _, topk_pos = torch.topk(cand_entropies, k, largest=True, sorted=True)
        selected_indices = cand_indices[topk_pos]
    else:
        order = torch.argsort(cand_entropies, descending=True)
        selected_indices = cand_indices[order]

    # Build records
    records: List[Tuple[int, str, str, int, int, float]] = []
    for idx in selected_indices.tolist():
        records.append(
            (
                idx,
                data.text_descriptions[idx],
                int(y_cpu[idx].item()),
                int(preds_cpu[idx].item()),
                float(confs_cpu[idx].item()),
                float(entropies_cpu[idx].item()),  # <-- include entropy
            )
        )
    return records




def embed_text(text: str, model: nn.Module) -> torch.Tensor:
    """Encode text using sentence-bert model"""
    return model.encode(text)


def main():
    params = get_args()
    sweep_mode = params["sweep_mode"]
    params["data_path"] = osp.join(os.path.dirname(__file__), "datasets")

    dataset = params["data"]
    emb_method = params["emb_method"]
    node_text = "_".join(params["node_text"])

    has_node_text = dataset in ["cora", "pubmed"]

    if params["use_params"]:
        with open(f"src/config/finetune.yaml", "r") as f:
            default_params = yaml.safe_load(f)
            if has_node_text:
                params.update(default_params[emb_method][node_text][dataset])
            else:
                params.update(default_params[emb_method][dataset])

    print(params["seed"])

    if has_node_text:
        params.update({"node_text": node_text})
    else:
        params.update({"node_text": "none"})

    if sweep_mode:
        metrics = run(params)
        # simply print or return the key metrics
        print(f"SWEEP METRICS → best_val={metrics['best_val']:.4f}")
        return
    wandb.init(
        project="TTA",
        name="Data:{} | Backbone:{} | LLM:{} | EMB_METHOD: {}".format(
            params["data"], params["backbone"], params["llm"], emb_method
        ),
        config=params,
        mode=(
            "disabled" if params["debug"] else "online"
        ),  # sweep only works in online mode
        group=params["group"],
    )
    params = dict(wandb.config)

    print(params)

    run(params)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)


def run(
    params: Namespace,
):
    seed = params["seed"]
    device = get_device(params)

    graph = get_data(params, params["data"])
    split = get_split(graph.y, params)  # returns a dict, not a list

    if params['sweep_mode']:
        split['test'] = split['val']

    seed_everything(seed)
    if params.get("demos_method") == "retrieval":
        print("Using Structure-Enhanced Retriever for retrieval...")
        # --- Train & build the Structure‐Enhanced Retriever ---
        retriever = SERetriever(
            in_dim=graph.num_dim,
            hidden_dim=params["retriever_hidden_dim"],
            num_layers=params["retriever_num_layers"],
        ).to(device)
        retriever = train_retriever(retriever, graph, split, params)

        # --- 2) Precompute retrieval embeddings on CPU ---
        with torch.no_grad():
            R = retriever(graph.x.to(device), graph.edge_index.to(device))
        graph.retrieval_embs = R.cpu()
    else:
        # --- Node2Vec training ---
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
        graph.struct_embs = F.normalize(node2vec.embedding.weight.detach().cpu(), p=2, dim=1)

    model = GNN(
        input_dim=graph.num_dim,
        hidden_dim=params["hidden_dim"],
        output_dim=graph.num_classes,
        activation=nn.PReLU,
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    ).to(device)
    logger = Logger()

    llm_client = LLMClientOllama(
        model_name=params["llm"],
    )
    updated_records: List[Tuple[int, str]] = []

    model_sbert = SentenceTransformer(SBERT_MODELS[params["emb_model"]])

    out_dir = osp.join(params["output_dir"], params["data"], params["llm"])
    os.makedirs(out_dir, exist_ok=True)

    buffer = PromptBuffer(max_size=params["buffer_size"])

    data = copy.deepcopy(graph)

    if params["feat_noise"] != 0:
        data.x = (1 - params["feat_noise"]) * data.x + params[
            "feat_noise"
        ] * torch.randn_like(data.x)
        print(
            "Add Gaussian noise on nodes with level {} on target!".format(
                params["feat_noise"]
            )
        )

    if params["edge_noise"] != 0:
        data = flip_edges(data, p=params["edge_noise"])
        print("Randomly flip {} edges on target!".format(params["edge_noise"]))

    task_model = copy.deepcopy(model).to(device)

    optimizer = AdamW(
        task_model.parameters(), lr=params["lr"], weight_decay=params["decay"]
    )
    stopper = EarlyStopping(patience=params["early_stop"])

    for epoch in range(1, params["epochs"] + 1):
        loss = train_gnn(task_model, data, split, optimizer, params)
        result = eval_gnn(task_model, data, split, params)

        is_stop = stopper(result)
        logger.log(seed, epoch, loss, result)
        if not params["sweep_mode"]:
            wandb.log(
                {
                    "train/loss_train": loss,
                    "train/train": result["train"],
                    "train/val": result["val"],
                    "train/test": result["test"],
                    "train/metric": result["metric"],
                }
            )

        # if last epoch, add best confs to the buffer
        if epoch == params["epochs"] or is_stop:
            # get the buffer size best confs and descriptions
            print("Adding best confs to the buffer...")

            num_classes = data.y.max().item() + 1
            quota = max(1, params["buffer_size"] // int(num_classes))
            selected_per_class = {c: [] for c in range(int(num_classes))}

            # true_train_indices/confidences came from eval_gnn()
            cand = result["true_train_indices"]  # indices in full graph
            confs = result["true_train_confs"]
            paired = sorted([(i, confs[idx]) for idx, i in enumerate(cand)], key=lambda t: t[1], reverse=True)
            
            for node_idx, _ in paired:
                c = int(graph.y[node_idx])
                if len(selected_per_class[c]) < quota:
                    selected_per_class[c].append(node_idx)
                # stop early if we already have buffer_size
                if sum(len(v) for v in selected_per_class.values()) >= params["buffer_size"]:
                    break
            
            # fallback to fill remaining slots (if some classes lacked confident samples)
            if sum(len(v) for v in selected_per_class.values()) < params["buffer_size"]:
                for node_idx, _ in paired:
                    if all(node_idx not in v for v in selected_per_class.values()):
                        # add to the class with the smallest count
                        cmin = min(selected_per_class.keys(), key=lambda c: len(selected_per_class[c]))
                        selected_per_class[cmin].append(node_idx)
                    if sum(len(v) for v in selected_per_class.values()) >= params["buffer_size"]:
                        break

            # Add entries
            added = 0
            for c, nodes in selected_per_class.items():
                for node_idx in nodes:
                    desc = result["true_train_descs"][result["true_train_indices"].index(node_idx)]
                    emb = torch.tensor(embed_text(desc, model=model_sbert), dtype=torch.float32).to(device)
                    buffer.add(node_idx=node_idx, label=int(data.y[node_idx]), text=desc, text_emb=emb)
                    added += 1
                    print(f"Adding node {node_idx} with label y:{data.y[node_idx]}({label_name(params['data'], data.y[node_idx])}) to the buffer.")
            print(f"Added {added} nodes to the buffer.")
        if is_stop:
            print("Early Stopping at Epoch:", epoch)
            break
    # run eval after training
    after_train_result = eval_gnn(task_model, data, split, params)
    print(
        "After Train Result/Before TTA - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(
            after_train_result["train"],
            after_train_result["val"],
            after_train_result["test"],
        )
    )

    # save model
    if params["save_model"]:
        model_save_path = os.path.join(out_dir, f"model_seed_{seed}.pth")
        torch.save(
            {
                "model_state_dict": task_model.state_dict(),
                "params": params,
            },
            model_save_path,
        )
        print(f"Saved trained GNN to {model_save_path}")
    if params['do_tta']:
        # ---- Test-Time Adaptation (TTA) ----
        print("Starting Test-Time Adaptation (TTA)…")
        buffer_alpha = params.get("buffer_alpha", 0.5)

        # to track high entropy nodes and their updated descriptions
        history_records = []
        prev_high_set = None
        for iteration in range(params["tta_max_iters"]):
            out_dir = osp.join(params["output_dir"], params["data"], params["llm"])
            os.makedirs(out_dir, exist_ok=True)

            # Paths for this iteration
            desc_csv = osp.join(
                out_dir,
                f"updated_descriptions_{params['data']}_{params['llm']}_tta_iter_{iteration}_seed_{seed}.csv",
            )
            iter_json = osp.join(out_dir, f"tta_results_seed_{seed}_iter_{iteration}.json")

            # --- CACHE-MISS: original TTA generation & save ---
            print(
                f"Test-time adaptation iteration {iteration + 1}/{params['tta_max_iters']}"
            )
            high_entropy = pred_gnn(task_model, data, split, params)
            if not high_entropy:
                print(
                    f"No high-entropy nodes found at iteration {iteration + 1}; stopping TTA."
                )
                break
            (
                node_idxs,
                old_texts,
                true_labels,
                pred_labels,
                confidences,
                entropies
            ) = zip(*high_entropy)
            curr_set = set(node_idxs)
            print(f"{len(curr_set)} high-entropy nodes: {sorted(curr_set)}")

            print(f"Found {len(high_entropy)} high-entropy nodes.")

            # 2 compare to last iteration
            if prev_high_set is not None:
                stayed = prev_high_set & curr_set
                dropped = prev_high_set - curr_set
                new = curr_set - prev_high_set
                print(f"    stayed: {sorted(stayed)}")
                print(f"    dropped: {sorted(dropped)}")
                print(f"    new:     {sorted(new)}")
            prev_high_set = curr_set

            # Load cached descriptions or generate via LLM
            if osp.exists(desc_csv) and not params["sweep_mode"]:
                print(
                    f"[seed={seed} iter={iteration}] loading cached descriptions from {desc_csv}"
                )
                df = pd.read_csv(desc_csv, sep="\t")

                # Build a mapping: node_idx → answer
                df.set_index("node_idx", inplace=True)
                answer_map = df["answer"].to_dict()

                # Only update those high-entropy nodes; also collect demos and prompts (no LLM call)
                updated = 0
                refined_texts = []
                demos_list = []  # store demo texts per node (for prompt construction)
                prompts_list = []  # store constructed prompts for trace
                for idx, old_desc in zip(
                    node_idxs, old_texts
                ):
                    new = answer_map.get(
                        idx, old_desc
                    )  # cached new description or fallback to old

                    # choose demos as per method, and record only the texts
                    if params.get("demos_method", "struct") == "struct":
                        demos = buffer.get_k_struct(
                            query_idx=idx,
                            struct_embs=data.struct_embs,
                            k=params["buffer_k"],
                        )
                    elif params.get("demos_method") == "text":
                        emb = torch.tensor(
                            embed_text(old_desc, model=model_sbert), dtype=torch.float32
                        ).to(device)
                        demos = buffer.get_k_text(emb, params["buffer_k"])
                    elif params.get("demos_method") == "combined":
                        emb_text = torch.tensor(
                            embed_text(old_desc, model=model_sbert), dtype=torch.float32
                        ).to(device)
                        demos = buffer.get_k_combined(
                            query_idx=idx,
                            text_emb=emb_text,
                            struct_embs=data.struct_embs,
                            k=params["buffer_k"],
                            buffer_alpha=buffer_alpha,
                        )
                    elif params.get("demos_method") == "retrieval":
                        demo_ids = get_k_retrieval_mmr(
                            query_idx=idx,
                            R=data.retrieval_embs,
                            y=data.y.cpu(),
                            k=params["buffer_k"],
                            pos_ratio=params.get("demo_pos_ratio", 0.5),
                            lambda_mmr=params.get("mmr_lambda", 0.7),
                        )
                        demos = [
                            (
                                nid,
                                F.cosine_similarity(
                                    data.retrieval_embs[idx],
                                    data.retrieval_embs[nid],
                                    dim=0,
                                ).item(),
                                data.text_descriptions[nid],
                            )
                            for nid in demo_ids
                        ]
                    else:
                        demos = []

                    demo_texts = [(demo[-1], demo[-2]) for demo in demos]
                    demos_list.append(demo_texts)

                    # Construct prompt (do NOT query LLM here)
                    prompt_txt = build_tta_prompt(old_desc, demo_texts)
                    prompts_list.append(prompt_txt)

                    # Update in-memory descriptions and embeddings using the cached "new" text
                    data.text_descriptions[idx] = new
                    
                    default_text = data.get("default_text", None)
                    t = default_text[idx] if default_text is not None else "na"
                    new_text = DESCRIPTION_TEMPLATE.format(NODE_TYPES[params['data']], t, new)

                    emb = torch.tensor(
                        embed_text(new_text, model=model_sbert), dtype=torch.float32
                    ).to(device)
                    data.x[idx] = emb
                    updated += 1
                    refined_texts.append(new)

                print(f"Updated {updated}/{len(node_idxs)} high-entropy nodes from cache")

            else:
                # For each high entropy text, get the structurally similar node's descriptions
                demos_list = []
                for idx, old_desc in zip(node_idxs, old_texts):
                    if params.get("demos_method", "struct") == "struct":
                        demos = buffer.get_k_struct(
                            query_idx=idx,
                            struct_embs=data.struct_embs,
                            k=params["buffer_k"],
                        )
                        print(
                            f"[STRUCT] Node {idx}: Top-{params['buffer_k']} similar nodes (with scores):"
                        )
                        for nid, sim, txt in demos:
                            print(f"  → Node {nid}, sim = {sim:.4f}")
                    elif params.get("demos_method") == "text":
                        emb = torch.tensor(
                            embed_text(old_desc, model=model_sbert), dtype=torch.float32
                        ).to(device)
                        demos = buffer.get_k_text(emb, params["buffer_k"])
                        print(
                            f"[TEXT] Node {idx}: Top-{params['buffer_k']} similar nodes (with scores):"
                        )
                        for nid, sim, txt in demos:
                            print(f"  → Node {nid}, sim = {sim:.4f}")
                    elif params.get("demos_method") == "combined":
                        emb_text = torch.tensor(
                            embed_text(old_desc, model=model_sbert), dtype=torch.float32
                        ).to(device)
                        # Retrieve top-k combined similar nodes
                        demos = buffer.get_k_combined(
                            query_idx=idx,
                            text_emb=emb_text,
                            struct_embs=data.struct_embs,
                            k=params["buffer_k"],
                            buffer_alpha=buffer_alpha,
                        )
                        print(
                            f"[COMBINED] Node {idx}: Top-{params['buffer_k']} combined similar nodes:"
                        )
                        for nid, score, cos_t, cos_s, label, txt in demos:
                            print(
                                f"  → Node {nid}: combined={score:.4f} (text={cos_t:.4f}, struct={cos_s:.4f}), label={label}"
                            )
                    elif params.get("demos_method") == "retrieval":
                        # inside TTA loop, for each high-entropy node idx:
                        demo_ids = get_k_retrieval_mmr(
                            query_idx=idx,
                            R=data.retrieval_embs,
                            y=data.y.cpu(),
                            k=params["buffer_k"],
                            pos_ratio=params.get("demo_pos_ratio", 0.5),
                            lambda_mmr=params.get("mmr_lambda", 0.7),
                        )
                        demos = [
                            (
                                nid,
                                F.cosine_similarity(
                                    data.retrieval_embs[idx],
                                    data.retrieval_embs[nid],
                                    dim=0,
                                ).item(),
                                data.text_descriptions[nid],
                            )
                            for nid in demo_ids
                        ]
                        print(
                            f"[RETRIEVAL] Node {idx}: Top-{params['buffer_k']} retriever-similar nodes:"
                        )
                        for nid, sim, txt in demos:
                            print(f"  → Node {nid}, sim = {sim:.4f}")

                    demos_list.append([demo for demo in demos])

                # Rewrite all in batch with individual demos
                refined_texts = llm_client.batch_rewrite(
                    old_texts, demos_list, dataset=params['data']
                )
                print(f"Refined {len(refined_texts)} high-entropy nodes.")

                # 3) Update embeddings/descriptions in-place (fuse with fixed alpha)
                for (
                    node_idx,
                    old_text,
                    true_lbl,
                    pred_lbl,
                    conf,
                    ent
                ), new_desc in zip(high_entropy, refined_texts):
                    embed_alpha = params.get("embed_alpha", 0.5)
                    
                    default_text = data.get("default_text", None)
                    t = default_text[node_idx] if default_text is not None else "na"
                    new_text = new_desc # new description obtained from the LLM
                    
                    llm_emb = embed_text(new_text, model=model_sbert)
                    llm_emb = torch.tensor(llm_emb, dtype=torch.float32).to(device)
                    orig_emb = torch.tensor(embed_text(data.text_descriptions[node_idx], model=model_sbert), dtype=torch.float32).to(device)
                    fused = embed_alpha * llm_emb + (1.0 - embed_alpha) * orig_emb
                    data.x[node_idx] = fused.detach().cpu()
                    data.text_descriptions[node_idx] = new_desc
                    updated_records.append((node_idx, new_desc))
                if not params["sweep_mode"]:
                    # Save updated descriptions in a CSV file
                    df = pd.DataFrame(
                        {
                            "node_idx": range(len(data.text_descriptions)),
                            "answer": data.text_descriptions,
                        }
                    )
                    df.to_csv(desc_csv, index=False, sep="\t")
                    print(f"Saved descriptions to {desc_csv}")

            # record history
            for idx, old_desc, new, true_lbl, pred_lbl, conf, ent in zip(
                node_idxs,
                old_texts,
                refined_texts,
                true_labels,
                pred_labels,
                confidences,
                entropies
            ):
                history_records.append(
                    {
                        "iteration": iteration,
                        "node_idx": idx,
                        "true_label": true_lbl,
                        "pred_label": pred_lbl,
                        "confidence": conf,
                        "entropy": ent,    
                        "old_desc": old_desc,
                        "new_desc": new,
                    }
                )

            tta_result = eval_gnn(task_model, data, split, params)
            print(
                "TTA Result - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(
                    tta_result["train"], tta_result["val"], tta_result["test"]
                )
            )
            if not params["sweep_mode"]:
                wandb.log(
                    {
                        "tta/train": tta_result["train"],
                        "tta/val": tta_result["val"],
                        "tta/test": tta_result["test"],
                        "tta/metric": tta_result["metric"],
                    }
                )
                # write per‐iter JSON
                with open(iter_json, "w") as jf:
                    json.dump(
                        {
                            "seed": seed,
                            "iteration": iteration,
                            "train": tta_result["train"],
                            "val": tta_result["val"],
                            "test": tta_result["test"],
                        },
                        jf,
                        indent=4,
                    )


        if not params["sweep_mode"]:
            # save the per-seed history out to disk
            print(f"About to save {len(history_records)} history rows")
            df_hist = pd.DataFrame(history_records)
            df_hist.to_csv(
                osp.join(out_dir, f"tta_history_seed_{seed}.csv"), index=False, sep="\t"
            )
    # After adaptation, run final evaluation
    final_result = eval_gnn(task_model, data, split, params)
    if not params["sweep_mode"]:

        wandb.log(
            {
                "best/test": final_result["test"],
                "best/val": final_result["val"],
                "best/train": final_result["train"],
            }
        )

        wandb.log(
            {"meta/run": logger.get_run_raw(), "meta/best": logger.get_best_raw()}
        )

        wandb.finish()

    print(
        "Final Result - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(
            final_result["train"], final_result["val"], final_result["test"]
        )
    )

    return {
        "best_val": final_result["val"],
        "best_test": final_result["test"],
    }


if __name__ == "__main__":
    main()
