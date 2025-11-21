import os
import os.path as osp
import copy
import yaml
import subprocess

import wandb

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from src.model.model import GNN
from src.data.data import get_data
from src.data.split import get_split
from src.utils.eval import evaluate, dataset2metric
from src.utils.args import get_args
from src.utils.logger import Logger
from src.utils.early_stop import EarlyStopping
from src.utils.utils import (
    seed_everything,
    get_device,
    get_device_from_model,
)


# -----------------
# Helper train ops
# -----------------
def _pretrain_step(model, data):
    """Pretrain on SOURCE (no mask) exactly once per batch/epoch."""
    model.train()
    device = get_device_from_model(model)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    loss = F.cross_entropy(y_pred, y)

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    loss.backward()
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            raise ValueError("NaN in gradients during pretrain.")
    return loss


def _finetune_step(model, data, train_mask):
    """Finetune on TARGET using TRAIN mask."""
    model.train()
    device = get_device_from_model(model)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    loss = F.cross_entropy(y_pred[train_mask], y[train_mask])

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    loss.backward()
    return loss


def _eval(model, data, split, params):
    """Eval on TARGET; report train/val/test metrics for consistency."""
    model.eval()
    device = get_device_from_model(model)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    train_mask, val_mask, test_mask = split["train"], split["val"], split["test"]
    train_value = evaluate(y_pred, y, train_mask, params)
    val_value = evaluate(y_pred, y, val_mask, params)
    test_value = evaluate(y_pred, y, test_mask, params)

    return {
        "train": train_value,
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

    # Datasets
    src_name = params["pt_data"]
    tgt_name = params["data"]
    
    seed_everything(seed)

    # Use DA-style get_data signatures (with desc/topo CSVs)
    source = get_data(params, src_name, params.get("desc_csv"), params.get("topo_csv"))
    target = get_data(
        params, tgt_name, params.get("tgt_desc_csv"), params.get("tgt_topo_csv")
    )

    split = get_split(target.y, params)  # masks from TARGET

    
    # Template model (copied for task stage)
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

    # ------------------
    # Stage 1: Pretrain
    # ------------------
    pt_optimizer = AdamW(model.parameters(), lr=params["pt_lr"], weight_decay=params["pt_decay"])
    for epoch in range(1, params["pt_epochs"] + 1):
        loss = _pretrain_step(model, source)
        pt_optimizer.step()

        if not params.get("sweep_mode"):
            wandb.log({"pretrain/loss": float(loss)})

    # -------------------
    # Stage 2: Finetune
    # -------------------
    task_model = copy.deepcopy(model).to(device)
    # Adapt classifier head to TARGET label space
    task_model.reset_lin(target.num_classes)

    optimizer = AdamW(task_model.parameters(), lr=params["lr"], weight_decay=params["decay"])
    stopper = EarlyStopping(patience=params["early_stop"])

    for epoch in range(1, params["epochs"] + 1):
        loss = _finetune_step(task_model, target, split["train"])
        optimizer.step()

        result = _eval(task_model, target, split, params)
        logger.log(seed, epoch, float(loss), result)

        if not params.get("sweep_mode"):
            wandb.log(
                {
                    "train/loss_train": float(loss),
                    "train/train": result["train"],
                    "train/val": result["val"],
                    "train/test": result["test"],
                    "train/metric": result["metric"],
                }
            )

        if stopper(result):
            print("Early Stopping at Epoch:", epoch)
            break

    # Final eval snapshot (post-stop)
    final_result = _eval(task_model, target, split, params)

    if not params.get("sweep_mode"):
        wandb.log(
            {
                "best/train": final_result["train"],
                "best/val": final_result["val"],
                "best/test": final_result["test"],
            }
        )
        wandb.log({"meta/run": logger.get_run_raw(), "meta/best": logger.get_best_raw()})
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

    # DA-style datasets root
    params["data_path"] = osp.join(os.path.dirname(__file__), "datasets")

    pt_dataset = params["pt_data"]
    dataset = params["data"]
    emb_method = params["emb_method"]
    node_text = "_".join(params["node_text"])
    template_name = params.get("prompt_template", "default")

    # Optional defaults file mirroring DA style
    if params.get("use_params", False):
        cfg_path = "src/config/transfer.yaml"
        if osp.exists(cfg_path):
            with open(cfg_path, "r") as f:
                default_params = yaml.safe_load(f)
                # Expect layout: default_params[emb_method][node_text][pt_dataset][dataset]
                if (
                    emb_method in default_params
                    and node_text in default_params[emb_method]
                    and pt_dataset in default_params[emb_method][node_text]
                    and dataset in default_params[emb_method][node_text][pt_dataset]
                ):
                    params.update(default_params[emb_method][node_text][pt_dataset][dataset])

    def _git_rev():
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    run_name = (
        f"PT→FT:{pt_dataset}→{dataset}"
        f"|Backbone:{params['backbone']}"
        f"|LLM:{params.get('llm','n/a')}"
        f"|EMB:{emb_method}"
        f"|PTmpl:{template_name}"
    )
    wandb.init(
        project="PT-FT",
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
