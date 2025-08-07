import copy
import json
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
from src.utils.consts import SBERT_MODELS
from src.utils.early_stop import EarlyStopping
from src.utils.eval import dataset2metric, evaluate
from src.utils.logger import Logger
from src.utils.utils import (
    flip_edges,
    get_device,
    get_device_from_model,
    seed_everything,
)


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
    true_train_descs = [data.topo_descriptions[i] for i in true_train_idx_sorted]

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
    model.eval()
    device = get_device_from_model(model)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    # forward
    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    probs = F.softmax(y_pred, dim=-1)
    # per-node confidence and predicted class
    confs, preds = torch.max(probs, dim=-1)
    entropies = entropy(probs=probs)

    test_mask = split["test"]
    # indices within the test split whose entropy is high
    high_entropy_idxs = torch.nonzero(
        entropies[test_mask] > params["entropy_thr"], as_tuple=False
    )
    if high_entropy_idxs.numel() == 0:
        return []

    records = []
    # gather global indices of high-entropy nodes
    test_indices = test_mask.nonzero(as_tuple=False).squeeze()
    for rel_idx in high_entropy_idxs.squeeze().tolist():
        idx = test_indices[rel_idx].item()
        records.append(
            (
                idx,
                data.text_descriptions[idx],
                data.topo_descriptions[idx],
                y[idx].item(),  # true label
                preds[idx].item(),  # predicted label
                confs[idx].item(),  # confidence
            )
        )
    return records


def embed_text(text: str, model: nn.Module) -> torch.Tensor:
    """Encode text using sentence-bert model"""
    return model.encode(text)


def main():
    params = get_args()
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
    # Ensure seed is a list even if wandb.config returned an int
    if isinstance(params.get("seed"), int):
        params["seed"] = [params["seed"]]

    print(params)

    run(params)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)


def run(
    params: Namespace,
):
    seed_everything(1)  # Set seed for reproducibility
    device = get_device(params)

    graph = get_data(params, params["data"])
    splits = get_split(graph.y, params)

    # # --- Node2Vec training ---
    # print("Training Node2Vec for structural embeddings…")
    # node2vec = Node2Vec(
    #     graph.edge_index,
    #     embedding_dim=params["struct_dim"],
    #     walk_length=params["struct_walk_length"],
    #     context_size=params["struct_context_size"],
    #     walks_per_node=params["struct_walks_per_node"],
    #     num_negative_samples=params["struct_neg_samples"],
    #     p=params.get("struct_p", 1.0),
    #     q=params.get("struct_q", 1.0),
    #     sparse=True,
    # ).to(device)

    # loader = node2vec.loader(
    #     batch_size=params.get("struct_batch_size", 128), shuffle=True
    # )
    # optim_n2v = torch.optim.SparseAdam(
    #     node2vec.parameters(), lr=params.get("struct_lr", 1e-3)
    # )

    # for epoch in range(params.get("struct_epochs", 5)):
    #     total_loss = 0
    #     for pos_rw, neg_rw in loader:
    #         optim_n2v.zero_grad()
    #         loss_n2v = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
    #         loss_n2v.backward()
    #         optim_n2v.step()
    #         total_loss += loss_n2v.item()
    #     print(
    #         f"  Node2Vec epoch {epoch+1}/{params['struct_epochs']}, loss={total_loss:.4f}"
    #     )
    # print("Node2Vec training complete.")

    # # detach and store on CPU
    # graph.struct_embs = node2vec.embedding.weight.detach().cpu()


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

    for seed_idx, (seed, split) in enumerate(zip(params["seed"], splits)):
        seed_everything(seed)
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
            logger.log(seed_idx, epoch, loss, result)

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
                # Sort by confidence and take the top k
                sorted_indices = sorted(
                    range(len(result["true_train_confs"])),
                    key=lambda i: result["true_train_confs"][i],
                    reverse=True,
                )[: params["buffer_size"]]
                # Add to buffer
                print(f"Adding {len(sorted_indices)} best confs to the buffer.")
                for buff_idx in sorted_indices:
                    node_idx = result["true_train_indices"][buff_idx]
                    desc = result["true_train_descs"][buff_idx]
                    emb = torch.tensor(
                        embed_text(desc, model=model_sbert), dtype=torch.float32
                    ).to(device)
                    buffer.add(node_idx, desc, emb)
                    print(f"Adding node {node_idx} to the buffer.")
                print(
                    f"Buffer size after adding: {len(buffer.buffer)} / {params['buffer_size']}"
                )
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
        
        # --- Use GNN embeddings for “structural” retrieval ---
        # After training each seed’s GNN (`task_model`), compute its hidden representations:
        with torch.no_grad():
            # encode() returns the final hidden layer before classification head
            hidden = task_model.encode(
                data.x.to(device),
                data.edge_index.to(device)
            )
        # detach and store on CPU for PromptBuffer
        data.struct_embs = hidden.cpu()
        # ---- Test-Time Adaptation (TTA) ----
        print("Starting Test-Time Adaptation (TTA)…")
        alpha = params.get("alpha", 0.5)

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
            iter_json = osp.join(
                out_dir, f"tta_results_seed_{seed}_iter_{iteration}.json"
            )

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
                topo_descriptions,
                true_labels,
                pred_labels,
                confidences,
            ) = zip(*high_entropy)
            curr_set = set(node_idxs)
            print(f"{len(curr_set)} high‑entropy nodes: {sorted(curr_set)}")

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
            if osp.exists(desc_csv):
                print(
                    f"[seed={seed} iter={iteration}] loading cached descriptions from {desc_csv}"
                )
                df = pd.read_csv(desc_csv, sep="\t")

                # Build a mapping: node_idx → answer
                df.set_index("node_idx", inplace=True)
                answer_map = df["answer"].to_dict()

                # Only update those high‑entropy nodes
                updated = 0
                refined_texts = []
                for idx, old_desc in zip(node_idxs, old_texts):
                    new = answer_map.get(idx, old_desc)
                    data.text_descriptions[idx] = new

                    emb = embed_text(new, model=model_sbert)
                    data.x[idx] = torch.tensor(emb, dtype=torch.float32).to(device)
                    updated += 1
                    refined_texts.append(new)
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
                            alpha=alpha
                        )
                        print(f"[COMBINED] Node {idx}: Top-{params['buffer_k']} combined similar nodes:")
                        for nid, score, cos_t, cos_s, txt in demos:
                            print(
                                f"  → Node {nid}: combined={score:.4f} (text={cos_t:.4f}, struct={cos_s:.4f})"
                            )

                print(
                    f"Updated {updated}/{len(node_idxs)} high‑entropy nodes from cache"
                )
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
                            alpha=alpha
                        )
                        print(f"[COMBINED] Node {idx}: Top-{params['buffer_k']} combined similar nodes:")
                        for nid, score, cos_t, cos_s, txt in demos:
                            print(
                                f"  → Node {nid}: combined={score:.4f} (text={cos_t:.4f}, struct={cos_s:.4f})"
                            )
                    demos_list.append([demo[-1] for demo in demos])

                # Rewrite all in batch with individual demos
                refined_texts = llm_client.batch_rewrite(
                    old_texts, demos_list, topo_descriptions
                )
                print(f"Refined {len(refined_texts)} high-entropy nodes.")

                # 3) Update embeddings/descriptions in-place
                for (
                    node_idx,
                    old_text,
                    topo_desc,
                    true_lbl,
                    pred_lbl,
                    conf,
                ), new_desc in zip(high_entropy, refined_texts):
                    new_emb = embed_text(new_desc, model=model_sbert)
                    new_emb = torch.tensor(new_emb, dtype=torch.float32).to(device)
                    data.x[node_idx] = new_emb
                    data.text_descriptions[node_idx] = new_desc
                    updated_records.append((node_idx, new_desc))
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
            for idx, old_desc, new, true_lbl, pred_lbl, conf in zip(
                node_idxs,
                old_texts,
                refined_texts,
                true_labels,
                pred_labels,
                confidences,
            ):
                history_records.append(
                    {
                        "iteration": iteration,
                        "node_idx": idx,
                        "true_label": true_lbl,
                        "pred_label": pred_lbl,
                        "confidence": conf,
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

        # After adaptation, run final evaluation
        final_result = eval_gnn(task_model, data, split, params)

        # save the per-seed history out to disk
        print(f"About to save {len(history_records)} history rows")
        df_hist = pd.DataFrame(history_records)
        df_hist.to_csv(
            osp.join(out_dir, f"tta_history_seed_{seed}.csv"), index=False, sep="\t"
        )

        single_best = logger.get_single_best(seed_idx)
        wandb.log(
            {
                "best/train": single_best["train"],
                "best/val": single_best["val"],
                "best/test": final_result["test"],
            }
        )

    wandb.log({"meta/run": logger.get_run_raw(), "meta/best": logger.get_best_raw()})

    wandb.finish()


if __name__ == "__main__":
    main()
