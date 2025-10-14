import copy
import csv
import os
import os.path as osp
from argparse import Namespace

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
)
from src.training.evaluator import eval_gnn
from src.training.trainer import train_gnn
from src.utils.args import get_args
from src.utils.consts import SBERT_MODELS
from src.utils.early_stop import EarlyStopping
from src.utils.logger import Logger
from src.utils.utils import get_device, seed_everything


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

    # initialize LLM client
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

    out_dir = osp.join(params["output_dir"], params["data"], params["llm"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"All outputs will be saved to {out_dir}")

    data = copy.deepcopy(graph)

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
    if params.get("rewrite_all", False):
        # === dump training predictions JSON  ===
        train_json_path = osp.join(out_dir, f"train_buffer_seed_{seed}.json")
        buffer_json = dump_train_pred_json(
            train_json_path, task_model, data, split, params["data"]
        )
        print("[GLOBAL-REWRITE] Building PromptBuffer from JSON...")

        # Ensure we have structural embeddings available for retrieval
        if not hasattr(data, "struct_embs"):
            struct_embs = train_structural_embeddings(data, params)
            data.struct_embs = struct_embs

        # Build prompt buffer from training JSON
        prompt_buffer, meta_by_node = load_buffer_from_json(
            train_json_path,
            model_sbert,
            device=torch.device("cpu"),
            struct_embs=data.struct_embs,  
            sim_method=params.get("sim_method", "weighted_sum"),  
        )

        prompt_buffer.set_method(params.get("sim_method", "weighted_sum"))

        # Output paths after each iteration of rewriting
        global_csv_tmpl = osp.join(
            out_dir,
            f"global_rewrite_iter{{it}}_{params['data']}_{params['llm']}_seed_{seed}.csv",
        )  # new description CSV after each iteration
        hist_jsonl = osp.join(
            out_dir, f"icl_history_seed_{seed}.jsonl"
        )  # history of training nodes predictions only, for ICL

        # For history filtering of training nodes only
        train_idx = split["train"].nonzero(as_tuple=False).view(-1).tolist()
        train_idx_set = set(train_idx)

        # Iterate to rewrite the entire dataset
        I = int(params.get("rewrite_iters", 1))

        template_name = params.get("prompt_template", "default")
        train_idx = split["train"].nonzero(as_tuple=False).view(-1)
        val_idx   = split["val"].nonzero(as_tuple=False).view(-1)

        cache = {}  # create once outside the rewrite loop and reuse
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
            remaining = [nid for nid in node_order if nid not in done]

            if len(remaining) == 0:
                print(
                    "[GLOBAL-REWRITE] All nodes already rewritten. Skipping generation."
                )
            else:
                K = int(params.get("buffer_k", 5))
                buffer_alpha = params.get("buffer_alpha", 0.5)
                bs = int(params.get("rewrite_batch_size", 64))
                # Load history so far
                hist_dict = load_history(hist_jsonl)

                print(
                    f"[GLOBAL-REWRITE] Generating {len(remaining)} missing rows in batches of {bs}..."
                )

                # Process in mini-batches, saving after each batch so we can resume safely
                for start in range(0, len(remaining), bs):
                    batch_nids = remaining[start : start + bs]

                    # Build prompts for the batch
                    prompts = []
                    for nid in batch_nids:
                        emb_text = torch.tensor(
                            embed_text(data.text_descriptions[nid], model=model_sbert),
                            dtype=torch.float32,
                        ).to(device)
                        demo_nodes = prompt_buffer.get_k_combined(
                            query_idx=nid,
                            text_emb=emb_text,
                            struct_embs=data.struct_embs,
                            k=K,
                            buffer_alpha=buffer_alpha,
                        )
                        demo_nodes_ids = [e.node_idx for e in demo_nodes]
                        examples_block = format_example_block(
                            demo_nodes_ids,
                            meta_by_node,
                            params["data"],
                            hist=hist_dict,  # history will be included/placed per YAML unless you override max steps below
                            hist_max_steps=params.get("hist_max_steps", None),  # leave None to use YAML; set an int to override
                            template_name=params.get("ex_block_template", "default"),
                            yaml_path=params.get("ex_block_yaml", None),
                        )
                        prompt_txt = build_full_prompt_for_node_with_template(
                            nid, data, examples_block, params["data"], template_name
                        )
                        prompts.append(prompt_txt)

                    print("[GLOBAL-REWRITE] Example prompt:\n" + prompts[0] + "\n---")
                    # Generate with the LLM
                    gens = llm_client.batch_generate(prompts)

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
            emb_list = [model_sbert.encode(txt) for txt in data.text_descriptions]
            data.x = torch.tensor(emb_list, dtype=torch.float32)

            # Train on new descriptions, Evaluate snapshot and append train history
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
            append_history_jsonl(
                hist_jsonl, iteration=it, snapshot=snap, train_idx_set=train_idx_set
            )

        print("[GLOBAL-REWRITE] Done.")

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
