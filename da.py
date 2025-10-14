import os
import os.path as osp
import copy
import yaml
import random

import wandb

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from src.model.model import GNN
from src.data.data import get_data
from src.data.split import get_split, ratio_split
from src.utils.eval import evaluate, dataset2metric
from src.utils.args import get_args
from src.utils.logger import Logger
from src.utils.early_stop import EarlyStopping
from src.utils.utils import (
    seed_everything,
    get_device,
    get_device_from_model,
)


def train(model, data, split, optimizer, params):
    model.train()
    device = get_device_from_model(model)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    loss = F.cross_entropy(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval(model, data, split, params):
    model.eval()
    device = get_device_from_model(model)

    val_mask, test_mask = split["val"], split["test"]

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    val_value = evaluate(y_pred, y, val_mask, params)
    test_value = evaluate(y_pred, y, test_mask, params)

    return {
        "train": 0,
        "val": val_value,
        "test": test_value,
        "metric": dataset2metric[params["data"]],
    }


# --------------
# Main run logic
# --------------
def run(params):
    seed = params["seed"]
    device = get_device(params)

    # Source (pretrain) and Target (evaluate)
    src_name = params["pt_data"]
    tgt_name = params["data"]

    source = get_data(params, src_name, params['desc_csv'], params['topo_csv'])
    target = get_data(params, tgt_name, params['tgt_desc_csv'], params['tgt_topo_csv'])

    # Splits come from target labels (as in your code)
    split = get_split(target.y, params)

    seed_everything(seed)
    # Model template (copied per seed)
    model = GNN(
        input_dim=source.num_dim,
        hidden_dim=params["hidden_dim"],
        output_dim=source.num_classes,
        activation=nn.PReLU,
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    ).to(device)

    logger = Logger()

    # Per-seed training
    task_model = copy.deepcopy(model).to(device)

    optimizer = AdamW(
        task_model.parameters(), lr=params["lr"], weight_decay=params["decay"]
    )
    stopper = EarlyStopping(patience=params["early_stop"])

    for epoch in range(1, params["epochs"] + 1):
        loss = train(task_model, source, split, optimizer, params)
        result = eval(task_model, target, split, params)

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

    final_result = eval(task_model, target, split, params)
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
        "After Train Result/Before TTA - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(
            final_result["train"],
            final_result["val"],
            final_result["test"],
        )
    )

    return {
        "best_val": final_result["val"],
        "best_test": final_result["test"],
    }


# -----------
# Entry point
# -----------
def main():
    params = get_args()

    # Use the same datasets root style as autoprompt.py
    params["data_path"] = osp.join(os.path.dirname(__file__), "datasets")

    pt_dataset = params["pt_data"]
    dataset = params["data"]
    emb_method = params["emb_method"]
    node_text = "_".join(params["node_text"])

    # Prompt template compatibility with autoprompt naming
    template_name = params.get("prompt_template", "default")

    # Optional defaults file for DA (if you keep one)
    if params.get("use_params", False):
        cfg_path = "src/config/da.yaml"
        if osp.exists(cfg_path):
            with open(cfg_path, "r") as f:
                default_params = yaml.safe_load(f)
                params.update(default_params[emb_method][pt_dataset][dataset])

    import subprocess

    def _git_rev():
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    # W&B init (match autoprompt.py’s style)
    run_name = f"DA:{pt_dataset}→{dataset}|Backbone:{params['backbone']}|LLM:{params['llm']}|EMB:{emb_method}|PT:{template_name}"
    wandb.init(
        project="TTA-DA",
        name=run_name,
        config={**params, "git_rev": _git_rev(), "prompt_template": template_name},
        mode=("disabled" if params.get("debug") else "online"),
        group=params.get("group"),
    )
    params = dict(wandb.config)
    print(params)

    run(params)


if __name__ == "__main__":
    main()
