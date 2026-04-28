import copy
import csv
import os
import os.path as osp
import sys
from argparse import Namespace
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW

import wandb
from src.data.data import get_data
from src.data.split import get_split
from src.embeddings.structural import train_structural_embeddings
from src.embeddings.text import embed_text
from src.model.llm_client_ollama import LLMClientOllama
from src.model.llm_client_openai import LLMClientOpenAI
from src.model.model import GNN
from src.prompts.builders import (
    append_history_jsonl,
    build_full_prompt_for_node_with_template,
    dump_train_pred_json,
    eval_snapshot,
    format_example_block,
    load_buffer_from_json,
    load_history,
    format_target_history_block, 
)

from src.utils.compute_profile import ComputeProfile

from src.training.evaluator import eval_gnn
from src.training.trainer import train_gnn
from src.utils.args import get_args
from src.utils.consts import SBERT_MODELS
from src.utils.early_stop import EarlyStopping
from src.utils.logger import Logger
from src.utils.utils import get_device, seed_everything
from src.utils.exp_naming import build_run_slug, ensure_run_dir, dump_config
from src.neighbors.selection import select_neighbors, build_neighbor_context_text, build_one_hop_from_edge_index
from src.neighbors.keyphrases import extract_keyphrases_for_corpus


def load_done_rows_tsv(path: str) -> dict[int, str]:
    """
    Read existing rows (node_idx, answer) from a TSV file, returning {node_idx: answer}.
    Skips empty/invalid rows. Safe to call if file doesn't exist.
    """
    done: dict[int, str] = {}
    if not osp.exists(path):
        return done
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    nid = int(row.get("node_idx", "").strip())
                    ans = (row.get("answer") or "").strip()
                    if ans:
                        done[nid] = ans
                except Exception:
                    continue
    except Exception:
        # fall back silently; you can log if you want
        pass
    return done

def append_row_tsv(path: str, nid: int, answer: str) -> None:
    """
    Append a single row (node_idx, answer) to TSV immediately and fsync for durability.
    Creates header if the file didn't exist.
    """
    file_exists = osp.exists(path)
    # ensure directory exists
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["node_idx", "answer"])
        writer.writerow([nid, answer])
        f.flush()
        os.fsync(f.fileno())

def main():
    params = get_args()
    sweep_mode = params["sweep_mode"]
    params["data_path"] = osp.join(str(REPO_ROOT), "datasets")

    dataset = params["data"]
    emb_method = params["emb_method"]
    node_text = "_".join(params["node_text"])

    has_node_text = dataset in ["cora", "pubmed"]

    if params["use_params"]:
        with open(REPO_ROOT / "src" / "config" / "finetune.yaml", "r") as f:
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
    
    import subprocess
    def _git_rev():
        try:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        except Exception:
            return "unknown"
        
    template_name = params.get("prompt_template", "default")
        
    wandb.init(
        project="TTA",
        name=f"Data:{params['data']}|Backbone:{params['backbone']}|LLM:{params['llm']}|EMB:{emb_method}|PT:{template_name}",
        config={**params, "git_rev": _git_rev(), "prompt_template": template_name},
        mode=("disabled" if params["debug"] else "online"),
        group=params.get("group"),
    )
    params = dict(wandb.config)

    print(params)

    run(params)


def run(
    params: Namespace,
):
    seed = params["seed"]
    device = get_device(params)

    profile_on = bool(params.get("profile_compute", False))
    profile_out = params.get("profile_out") or ""
    if profile_on and not profile_out:
        # default: put in out_dir once we know it; temporary placeholder now
        profile_out = ""

    prof = ComputeProfile(
        enabled=profile_on,
        out_path=None,  # set later after out_dir exists
        sample_gpu=True,
        params=dict(params),
    )
    prof.start_gpu_sampling()


    with prof.phase("load_data"):
        graph = get_data(params, params["data"], params['desc_csv'], params['topo_csv'])
    split = get_split(graph.y, params)  # returns a dict, not a list

    seed_everything(seed)

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

    # initialize LLM client (only needed when rewriting)
    llm_client = None
    model_sbert = None
    if params.get("rewrite_all", False):
        if params['llm'].strip().lower().startswith("gpt-4o"):
            llm_client = LLMClientOpenAI(
                model_name=params['llm'],
                temperature=float(params.get("temp", 0.0)),
                max_output_tokens=int(params.get("max_output_tokens", 512)),
                parallel_workers=int(params.get("parallel_workers", 4)),
                system_prompt=params.get("system_prompt"),  # optional
            )
        else:
            llm_client = LLMClientOllama(model_name=params['llm'])
        print(f"Using LLM: {params['llm']}")

        #  initialize transformer model for text embeddings
        model_sbert = SentenceTransformer(SBERT_MODELS[params["emb_model"]])
        print(f"Using encoding model: {params['emb_model']}")

    run_dir = params.get("run_dir", None)
    if run_dir and str(run_dir).strip():
        out_dir = str(run_dir)
    else:
        slug = params.get("run_slug", None) or build_run_slug(params)
        out_dir = osp.join(params["output_dir"], slug)

    ensure_run_dir(out_dir)
    dump_config(out_dir, params)
    # Organize subfolders
    desc_dir = osp.join(out_dir, "descriptions")
    res_dir  = osp.join(out_dir, "results")
    ckpt_dir = osp.join(out_dir, "checkpoints")
    os.makedirs(desc_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    Path(out_dir, "paths.json").write_text(
        json.dumps(
            {"out_dir": out_dir, "desc_dir": desc_dir, "res_dir": res_dir, "ckpt_dir": ckpt_dir},
            indent=2
        ),
        encoding="utf-8"
    )

    if profile_on:
        prof.out_path = prof.out_path or osp.join(out_dir, f"compute_profile_seed_{seed}.json")


    print(f"[RUN] out_dir = {out_dir}")

    data = copy.deepcopy(graph)

    task_model = copy.deepcopy(model).to(device)

    optimizer = AdamW(
        task_model.parameters(), lr=params["lr"], weight_decay=params["decay"]
    )
    stopper = EarlyStopping(patience=params["early_stop"])

    with prof.phase("gnn_train_initial"):
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
            if is_stop:
                print("Early Stopping at Epoch:", epoch)
                break

    # run eval after training for logging (for the last epoch -- though this is reduntant, makes logging easier to report final score)
    after_train_result = eval_gnn(task_model, data, split, params)
    print(
        "After Train Result/Before TTA - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(
            after_train_result["train"],
            after_train_result["val"],
            after_train_result["test"],
        )
    )

    # === rewrite the entire dataset using the JSON buffer ===
    # === rewrite the entire dataset using the JSON buffer ===
    if params.get("rewrite_all", False):
        # === dump training predictions JSON  ===
        train_json_path = osp.join(desc_dir, f"train_buffer_seed={seed}.json")
        with prof.phase("dump_train_pred_json"):
            buffer_json = dump_train_pred_json(
                train_json_path, task_model, data, split, params["data"]
            )
        print("[GLOBAL-REWRITE] Building PromptBuffer from JSON...")

        # Ensure we have structural embeddings available for retrieval
        with prof.phase("train_structural_embeddings"):
            if not hasattr(data, "struct_embs"):
                struct_embs = train_structural_embeddings(data, params)
                data.struct_embs = struct_embs

        # Build prompt buffer from training JSON
        with prof.phase("build_prompt_buffer"):

            prompt_buffer, meta_by_node = load_buffer_from_json(
                train_json_path,
                model_sbert,
                device=torch.device("cpu"),
                struct_embs=data.struct_embs,
                sim_method=params.get("sim_method", "weighted_sum"),
            )

        prompt_buffer.set_method(params.get("sim_method", "weighted_sum"))

        # === Load 1-hop neighbors and precompute embeddings for neighbor selection ===
        neighbor_strategy = params.get("neighbor_strategy", "none")
        one_hop = {}
        neighbor_embeddings = None
        if neighbor_strategy != "none":
            dataset = params["data"]
            if dataset in ["cora", "pubmed"]:
                one_hop_path = osp.join(str(REPO_ROOT), "datasets", "property", f"{dataset}_one_hop.npy")
                one_hop = np.load(one_hop_path, allow_pickle=True).item()
                print(f"[NEIGHBOR] Loaded 1-hop neighbors from {one_hop_path} ({len(one_hop)} nodes)")
            else:
                # Airport datasets: compute from edge_index
                one_hop = build_one_hop_from_edge_index(data.edge_index, data.num_nodes)
                print(f"[NEIGHBOR] Built 1-hop neighbors from edge_index ({len(one_hop)} nodes)")

            # Precompute embeddings for neighbor selection
            is_text_rich = dataset in ["cora", "pubmed"]
            if is_text_rich:
                print("[NEIGHBOR] Encoding node texts with SBERT for neighbor selection...")
                neighbor_embeddings = np.array(
                    [model_sbert.encode(txt) for txt in data.text_descriptions]
                )
            else:
                # Text-free: use topological features as embeddings
                topo_path = osp.join(str(REPO_ROOT), "datasets", "property", f"{dataset}_topo.npy")
                topo_feats = np.load(topo_path, allow_pickle=True).item()
                # Build feature matrix: [num_nodes, 5] from topo feature dict
                feat_keys = ["degree", "betweenness", "closeness", "clustering", "square_clustering"]
                feat_matrix = np.zeros((data.num_nodes, len(feat_keys)))
                for i, key in enumerate(feat_keys):
                    if key in topo_feats:
                        vals = topo_feats[key]
                        if isinstance(vals, dict):
                            for nid, v in vals.items():
                                feat_matrix[int(nid), i] = v
                        else:
                            feat_matrix[:, i] = np.array(vals[:data.num_nodes])
                # Normalize each feature to [0, 1]
                for i in range(feat_matrix.shape[1]):
                    col = feat_matrix[:, i]
                    col_min, col_max = col.min(), col.max()
                    if col_max > col_min:
                        feat_matrix[:, i] = (col - col_min) / (col_max - col_min)
                neighbor_embeddings = feat_matrix
                print(f"[NEIGHBOR] Using normalized topo features as embeddings: shape={feat_matrix.shape}")

        # === Keyphrase extraction (GraphiT-style) ===
        neighbor_format = params.get("neighbor_format", "raw")
        keyphrases_by_node = {}
        is_text_rich = params["data"] in ["cora", "pubmed"]
        if neighbor_format == "keyphrases" and is_text_rich:
            print("[KEYPHRASE] Extracting keyphrases for all nodes...")
            keyphrases_by_node = extract_keyphrases_for_corpus(
                data.text_descriptions,
                method=params.get("keyphrase_method", "keybert"),
                top_k=int(params.get("keyphrase_top_k", 5)),
                sbert_model=model_sbert,
            )
            print(f"[KEYPHRASE] Done — extracted keyphrases for {len(keyphrases_by_node)} nodes")

        # Output paths after each iteration of rewriting
        global_csv_tmpl = osp.join(desc_dir, f"global_rewrite_iter={{{{it}}}}.tsv")  # only depends on iter now
        hist_jsonl = osp.join(desc_dir, f"icl_history_seed={seed}.jsonl")

        # Iterate to rewrite the entire dataset
        I = int(params.get("rewrite_iters", 1))
        template_name = params.get("prompt_template", "default")
        # Auto-select keyphrase template when using keyphrases
        if neighbor_format == "keyphrases" and template_name == "default":
            template_name = "default_with_neighbors_kp"
            print(f"[KEYPHRASE] Auto-selected template: {template_name}")

        for it in range(I):
            global_csv = global_csv_tmpl.format(it=it)
            print(f"[GLOBAL-REWRITE] >>> Iteration {it+1}/{I} writing to {global_csv}")

            done = {}
            if osp.exists(global_csv):
                try:
                    df_exist = pd.read_csv(global_csv, sep="\t")
                    for _, row in df_exist.iterrows():
                        nid = int(row["node_idx"])
                        ans = (
                            str(row["answer"])
                            if "answer" in row and pd.notna(row["answer"])
                            else ""
                        )
                        if len(ans.strip()) > 0:
                            done[nid] = ans.strip()
                    print(
                        f"[GLOBAL-REWRITE] Resume detected: {len(done)} rows already completed."
                    )
                except Exception as e:
                    print(
                        f"[GLOBAL-REWRITE] Could not read existing file for resume ({e}); proceeding fresh."
                    )

            node_order = list(range(data.num_nodes))
            max_nodes = int(params.get("rewrite_max_nodes", 0) or 0)
            if max_nodes > 0:
                node_order = node_order[:max_nodes]
            remaining = [nid for nid in node_order if nid not in done]

            # Load history of node predictions so far (all nodes, all iters)
            hist_dict = load_history(hist_jsonl)

            if len(remaining) == 0:
                print(
                    "[GLOBAL-REWRITE] All nodes already rewritten. Skipping generation."
                )
            else:
                K = int(params.get("buffer_k", 5))
                buffer_alpha = params.get("buffer_alpha", 0.5)
                bs = int(params.get("rewrite_batch_size", 64))

                # how many candidates to pull from the buffer before we split into
                # positives + hard negatives
                pool_size_cfg = int(params.get("retrieval_pool_size", max(K * 4, K)))
                pool_size = max(K, pool_size_cfg)
                pool_size = min(pool_size, int(prompt_buffer.node_idx.numel()))

                print(
                    f"[GLOBAL-REWRITE] Generating {len(remaining)} missing rows in batches of {bs} "
                    f"(buffer_k={K}, pool_size={pool_size})..."
                )

                # Process in mini-batches, saving after each batch so we can resume safely
                for start in range(0, len(remaining), bs):
                    batch_nids = remaining[start : start + bs]

                    prompts = []
                    for nid in batch_nids:
                        # text embedding for query node
                        emb_text = torch.tensor(
                            embed_text(
                                data.text_descriptions[nid], model=model_sbert
                            ),
                            dtype=torch.float32,
                        ).to(device)

                        # retrieve a pool of similar training nodes
                        cand_items = prompt_buffer.get_k_combined(
                            query_idx=nid,
                            text_emb=emb_text,
                            struct_embs=data.struct_embs,
                            k=pool_size,
                            buffer_alpha=buffer_alpha,
                        )

                        # --- mix positives + hard negatives ---
                        # target label (we rely on ground-truth labels for training nodes;
                        # this is fine since this is an offline rewriting stage)
                        target_label = int(data.y[nid].item())

                        if pool_size <= K:
                            # Not enough room to do fancy splitting; just take top-K
                            demo_nodes = cand_items[:K]
                        else:
                            pos = [it for it in cand_items if it.label == target_label]
                            neg = [it for it in cand_items if it.label != target_label]

                            selected = []
                            # Aim for ~half positives, half hard negatives
                            desired_pos = max(1, K // 2)
                            # take as many positives as we can up to desired_pos
                            selected.extend(pos[:desired_pos])

                            remaining_slots = K - len(selected)
                            if remaining_slots > 0:
                                selected.extend(neg[:remaining_slots])

                            # if still short (e.g., not enough negs), top up from remaining pool
                            if len(selected) < K:
                                leftover = [
                                    it
                                    for it in cand_items
                                    if it not in selected
                                ]
                                selected.extend(leftover[: (K - len(selected))])

                            demo_nodes = selected

                        demo_nodes_ids = [e.node_idx for e in demo_nodes]

                        # exemplar block: NO history for exemplars now
                        ex_block_tpl = params.get("ex_block_template", "default")
                        if neighbor_format == "keyphrases" and ex_block_tpl == "default":
                            ex_block_tpl = "keyphrases"
                        examples_block = format_example_block(
                            demo_nodes_ids,
                            meta_by_node,
                            params["data"],
                            hist=None,  # history moved to TARGET node instead
                            hist_max_steps=None,
                            template_name=ex_block_tpl,
                            yaml_path=params.get("ex_block_yaml", None),
                            keyphrases_by_node=keyphrases_by_node if neighbor_format == "keyphrases" else None,
                        )

                        # history for the TARGET node (predictive entropy, previous descs)
                        target_history_block = format_target_history_block(
                            nid,
                            hist_dict,
                            dataset=params["data"],
                            max_steps=params.get("hist_max_steps", None),
                        )

                        # Build neighbor context if strategy is active
                        nc = None
                        if neighbor_strategy != "none":
                            neighbors = one_hop.get(nid, [])
                            if len(neighbors) > 0:
                                selected = select_neighbors(
                                    nid, neighbors, neighbor_embeddings,
                                    k=int(params.get("num_neighbors", 5)),
                                    strategy=neighbor_strategy,
                                    mmr_lambda=float(params.get("mmr_lambda", 0.7)),
                                )
                                nc = build_neighbor_context_text(
                                    nid, selected,
                                    data.text_descriptions, data.topo_descriptions,
                                    is_text_rich=(params["data"] in ["cora", "pubmed"]),
                                    keyphrases_by_node=keyphrases_by_node if neighbor_format == "keyphrases" else None,
                                    neighbor_format=neighbor_format,
                                )
                            else:
                                nc = "No connected neighbors found."

                        prompt_txt = build_full_prompt_for_node_with_template(
                            nid,
                            data,
                            examples_block,
                            params["data"],
                            template_name,
                            target_history_block=target_history_block,
                            neighbor_context=nc,
                        )
                        prompts.append(prompt_txt)

                    print("[GLOBAL-REWRITE] Example prompt:\n" + prompts[0] + "\n---")
                    with prof.phase("llm_generate"):
                        gens = llm_client.batch_generate(prompts)
                    
                    prof.add("llm_output_chars", sum(len(g) for g in gens), mode="sum")


                    for nid, gen in zip(batch_nids, gens):
                        done[nid] = gen.strip()

                    df_out = pd.DataFrame(
                        {
                            "node_idx": sorted(done.keys()),
                            "answer": [done[k] for k in sorted(done.keys())],
                        }
                    )
                    df_out.to_csv(global_csv, index=False, sep="\t")
                    print(
                        f"[GLOBAL-REWRITE] Saved progress: {len(done)}/{data.num_nodes} rows → {global_csv}"
                    )

            # === After finishing (or resuming) this iteration: re-embed and evaluate ===
            # Load the just-written CSV as new descriptions
            df_new = pd.read_csv(global_csv, sep="\t")
            new_texts = [""] * data.num_nodes
            for _, row in df_new.iterrows():
                new_texts[int(row["node_idx"])] = str(row["answer"])
            data.text_descriptions = new_texts

            # Re-embed with SBERT
            with prof.phase("sbert_reembed_all"):
                emb_list = [model_sbert.encode(txt) for txt in data.text_descriptions]
                data.x = torch.tensor(emb_list, dtype=torch.float32)

            # Train on new descriptions, Evaluate snapshot and append prediction history
            with prof.phase("gnn_train_after_rewrite"):

                print("[GLOBAL-REWRITE] Training on rewritten descriptions...")
                optimizer = AdamW(
                    task_model.parameters(), lr=params["lr"], weight_decay=params["decay"]
                )
                stopper = EarlyStopping(patience=params["early_stop"])
                for epoch in range(1, params["epochs"] + 1):
                    loss = train_gnn(task_model, data, split, optimizer, params)
                    result = eval_gnn(task_model, data, split, params)

                    is_stop = stopper(result)

                    if is_stop:
                        print("Early Stopping at Epoch:", epoch)
                        break

                snap = eval_snapshot(task_model, data, split, params["data"])
                # IMPORTANT: log ALL nodes now, so target nodes (train/val/test)
                # have entropy + description history.
                append_history_jsonl(
                    hist_jsonl, iteration=it, snapshot=snap, keep_idx_set=None
                )

        # Print API cost summary if using OpenAI client
        if hasattr(llm_client, "get_cost_summary"):
            print(llm_client.get_cost_summary())
        print("[GLOBAL-REWRITE] Done.")


    # save model
    if params["save_model"]:
        model_save_path = os.path.join(ckpt_dir, f"model_seed={seed}.pth")
        torch.save(
            {
                "model_state_dict": task_model.state_dict(),
                "params": params,
            },
            model_save_path,
        )
        print(f"Saved trained GNN to {model_save_path}")

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

    prof.stop_gpu_sampling()
    if profile_on and prof.out_path:
        prof.dump(prof.out_path)
        print(f"[compute] wrote profile → {prof.out_path}")

    return {
        "best_val": final_result["val"],
        "best_test": final_result["test"],
    }


if __name__ == "__main__":
    main()
