import copy
import json
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from torch_geometric.data import Data, HeteroData

from src.model.llm_client_vllm import LLMClientVllm
from src.model.llm_client_ollama import LLMClientOllama
from src.model.prompt_buffer import PromptBuffer
from src.model.model import GNN

from argparse import Namespace

from src.utils.args import get_args

import wandb
import yaml

import os.path as osp
import os

from src.data.data import get_data
from src.data.split import get_split
from src.utils.early_stop import EarlyStopping
from src.utils.eval import evaluate, dataset2metric
from src.utils.logger import Logger
from src.utils.utils import (
    flip_edges,
    get_device,
    get_device_from_model,
    seed_everything,
)
import pandas as pd
from src.utils.consts import SBERT_MODELS

from sentence_transformers import SentenceTransformer

os.environ["VLLM_USE_V1"] = "1"


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
    confs, preds = torch.max(probs, dim=-1)                # preds is the argmax class
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
    val_value   = evaluate(y_pred, y, val_mask, params)
    test_value  = evaluate(y_pred, y, test_mask, params)

    return {
        "train": train_value,
        "val":   val_value,
        "test":  test_value,
        "metric": dataset2metric[params["data"]],
        # NEW: correct (true) train‐predictions, sorted by confidence
        "true_train_indices": true_train_idx_sorted,
        "true_train_confs":   true_train_confs,
        "true_train_descs":   true_train_descs,
    }


def pred_gnn(
    model: torch.nn.Module,
    data: Union[Data, HeteroData],  # PyG Data object
    split: Dict[str, torch.Tensor],  # {'train': mask, 'val': mask, 'test': mask}
    params: Dict[str, Any],
) -> Dict[str, Any]:
    model.eval()
    device = get_device_from_model(model)

    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    # Forward
    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    # Compute confidence scores
    probs = F.softmax(y_pred, dim=-1)

    entropies = entropy(probs=probs)

    # use entropy threshold only on test split nodes
    test_mask = split["test"]
    high_entropy_idxs = torch.nonzero(
        entropies[test_mask] > params["entropy_thr"], as_tuple=False
    )
    if high_entropy_idxs.numel() == 0:
        return []
    nodes = []
    for rel_idx in high_entropy_idxs.squeeze().tolist():
        idx = test_mask.nonzero()[rel_idx].item()
        nodes.append((idx, data.text_descriptions[idx], data.topo_descriptions[idx]))
    return nodes


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

    # llm_client = LLMClientVllm(
    #     model_name=params['llm'],
    #     max_output_tokens=params['max_output_tokens'],
    #     temperature=params['temperature'],
    # )
    llm_client = LLMClientOllama(
        model_name=params["llm"],
    )
    updated_records: List[Tuple[int, str]] = []

    model_sbert = SentenceTransformer(SBERT_MODELS[params["emb_model"]])

    # Prepare results file path
    out_dir = osp.join(params["output_dir"], params["data"], params["llm"])
    os.makedirs(out_dir, exist_ok=True)
    results_path = osp.join(out_dir, "results.json")

    # Load existing results if available
    if osp.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}


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
                    desc = result["true_train_descs"][buff_idx]
                    emb = embed_text(desc, model=model_sbert)
                    emb = torch.tensor(emb, dtype=torch.float32).to(device)
                    buffer.add(desc, emb)
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
                after_train_result["train"], after_train_result["val"], after_train_result["test"]
            )
        )

        # After training, run test-time-adaptation
        seed_entry: Dict[str, Any] = {"tta": {}}

        # to track high entropy nodes and their updated descriptions
        history_records = []
        prev_high_set = None
        for iteration in range(params["tta_max_iters"]):
            out_dir = osp.join(params["output_dir"], params["data"], params["llm"])
            os.makedirs(out_dir, exist_ok=True)

            # Paths for this iteration
            desc_csv = osp.join(
                out_dir,
                f"updated_descriptions_{params['data']}_{params['llm']}_tta_iter_{iteration}_seed_{seed}.csv"
            )
            iter_json = osp.join(
                out_dir,
                f"tta_results_seed_{seed}_iter_{iteration}.json"
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
            node_idxs, old_texts, topo_descriptions = zip(*high_entropy)
            curr_set = set(node_idxs)
            print(f"{len(curr_set)} high‑entropy nodes: {sorted(curr_set)}")

            print(f"Found {len(high_entropy)} high-entropy nodes.")
            
            # 2) compare to last iteration
            if prev_high_set is not None:
                stayed = prev_high_set & curr_set
                dropped = prev_high_set - curr_set
                new     = curr_set - prev_high_set
                print(f"    stayed: {sorted(stayed)}")
                print(f"    dropped: {sorted(dropped)}")
                print(f"    new:     {sorted(new)}")
            prev_high_set = curr_set

            # Load cached descriptions or generate via LLM
            if osp.exists(desc_csv):
                print(f"[seed={seed} iter={iteration}] loading cached descriptions from {desc_csv}")
                df = pd.read_csv(desc_csv)

                # Build a mapping: row_index → answer
                answer_map: Dict[int, str] = df['answer'].to_dict()

                # Only update those high‑entropy nodes
                updated = 0
                refined_texts = []
                for idx, old in zip(node_idxs, old_texts):
                    new = answer_map.get(idx, old)
                    data.text_descriptions[idx] = new

                    emb = embed_text(new, model=model_sbert)
                    data.x[idx] = torch.tensor(emb, dtype=torch.float32).to(device)
                    updated += 1
                    refined_texts.append(new)

                print(f"Updated {updated}/{len(node_idxs)} high‑entropy nodes from cache")
            else:
            # For each high entropy text, use the node_idxs
                demos_list = []
                for desc in old_texts:
                    emb = embed_text(desc, model=model_sbert)
                    emb = torch.tensor(emb, dtype=torch.float32).to(device)
                    demos = buffer.get_k(desc, emb, params["buffer_k"])
                    if not demos:
                        print(
                            "No demos found in buffer for text: {}".format(desc)
                        )
                        demos = [""]
                    demos_list.append(demos)

                # Rewrite all in batch with individual demos
                refined_texts = llm_client.batch_rewrite(old_texts, demos_list, topo_descriptions)
                print(f"Refined {len(refined_texts)} high-entropy nodes.")

                # 3) Update embeddings/descriptions in-place
                for (node_idx, _, _), new_desc in zip(high_entropy, refined_texts):
                    new_emb = embed_text(new_desc, model=model_sbert)
                    new_emb = torch.tensor(new_emb, dtype=torch.float32).to(device)
                    data.x[node_idx] = new_emb
                    data.text_descriptions[node_idx] = new_desc
                    updated_records.append((node_idx, new_desc))
                # Save updated descriptions in a CSV file
                df = pd.DataFrame(data.text_descriptions, columns=["answer"])
                df.to_csv(desc_csv, index=False)
                print(f"Saved descriptions to {desc_csv}")

            # record history
            for idx, old, new in zip(node_idxs, old_texts, refined_texts):
                history_records.append({
                    "iteration": iteration,
                    "node_idx":  idx,
                    "old_desc":  old,
                    "new_desc":  new,
                })

            tta_result = eval_gnn(task_model, data, split, params)
            print(
                "TTA Result - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(
                    tta_result["train"], tta_result["val"], tta_result["test"]
                )
            )
            wandb.log(
                {
                    "tta/train": tta_result["train"],
                    "tta/val":   tta_result["val"],
                    "tta/test":  tta_result["test"],
                    "tta/metric":tta_result["metric"],
                }
            )
            # write per‐iter JSON
            with open(iter_json, "w") as jf:
                json.dump({
                    "seed":      seed,
                    "iteration": iteration,
                    "train":     tta_result["train"],
                    "val":       tta_result["val"],
                    "test":      tta_result["test"],
                }, jf, indent=4)


        # After adaptation, run final evaluation
        final_result = eval_gnn(task_model, data, split, params)

        # save the per-seed history out to disk
        print(f"About to save {len(history_records)} history rows")
        df_hist = pd.DataFrame(history_records)
        df_hist.to_csv(osp.join(out_dir, f"tta_history_seed_{seed}.csv"), index=False)

        single_best = logger.get_single_best(seed_idx)
        wandb.log(
            {
                "best/train": single_best["train"],
                "best/val": single_best["val"],
                "best/test": final_result["test"],
            }
        )
        seed_entry["final"] = {"train": single_best["train"], "val": single_best["val"], "test": final_result["test"]}

        # Update the JSON file without overwriting existing entries
        all_results[str(seed)] = seed_entry
        with open(results_path, 'w') as jf:
            json.dump(all_results, jf, indent=4)
        print(f"Updated JSON results at {results_path} for seed {seed}")

    best = logger.get_best()
    wandb.log(
        {
            "final/train": "{:.2f} ± {:.2f}".format(
                best["train"]["mean"], best["train"]["std"]
            ),
            "final/val": "{:.2f} ± {:.2f}".format(
                best["val"]["mean"], best["val"]["std"]
            ),
            "final/test": "{:.2f} ± {:.2f}".format(
                best["test"]["mean"], best["test"]["std"]
            ),
            "final/train_mean": best["train"]["mean"],
            "final/val_mean": best["val"]["mean"],
            "final/test_mean": best["test"]["mean"],
            "final/train_std": best["train"]["std"],
            "final/val_std": best["val"]["std"],
            "final/test_std": best["test"]["std"],
        }
    )
    wandb.log({"meta/run": logger.get_run_raw(), "meta/best": logger.get_best_raw()})

    wandb.finish()


if __name__ == "__main__":
    main()
