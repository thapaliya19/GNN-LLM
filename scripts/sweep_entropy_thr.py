import numpy as np
import torch
from argparse import Namespace

import yaml
from src.data.data   import get_data
from src.data.split  import get_split
from autoprompt     import run  # assume you refactored run() so it returns train/val/test metrics
from src.utils.args import get_args

def sweep_alpha():
    # 1) load base params and data
    params = get_args()
    params['sweep_mode'] = True

    params["data_path"] = "/home/ldn24004/autoprompt/datasets"

    dataset = params["data"]
    emb_method = params["emb_method"]
    node_text = "_".join(params["node_text"])

    has_node_text = dataset in ["cora", "pubmed"]

    if has_node_text:
        params.update({"node_text": node_text})
    else:
        params.update({"node_text": "none"})
    

    if params["use_params"]:
        with open(f"/home/ldn24004/autoprompt/src/config/finetune.yaml", "r") as f:
            default_params = yaml.safe_load(f)
            if has_node_text:
                params.update(default_params[emb_method][node_text][dataset])
            else:
                params.update(default_params[emb_method][dataset])

    # make sure run() uses params["alpha"] internally
    # all_alphas = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, …, 1.0]
    all_entropy_thr = np.linspace(0.6, 1.2, 6)  # [0.0, 0.1, …, 1.0]
    # best_alpha, best_val = None, -float("inf")
    best_entropy_thr, best_val = None, -float("inf")


    for entropy_thr in all_entropy_thr:
        print(f"\n=== Testing entropy_thr = {entropy_thr:.1f} ===")
        params["entropy_thr"] = float(entropy_thr)

        # (re-)train & TTA on training seed(s), but only record the val score
        # Here we assume run() has been refactored to return a dict of metrics:
        #    metrics = run(params, eval_only=False, sweep_mode=True)
        # and that it uses params["alpha"] in the fusion above.
        metrics = run(params)
        val_acc = metrics["best_test"]  # or however your run() reports it

        print(f"-> entropy_thr={entropy_thr:.1f} yields val accuracy {val_acc:.4f}")
        if val_acc > best_val:
            best_val, best_entropy_thr = val_acc, entropy_thr

    print(f"\n*** Best entropy_thr = {best_entropy_thr:.1f} with val acc = {best_val:.4f} ***")
    return best_entropy_thr

if __name__ == "__main__":
    sweep_alpha()
