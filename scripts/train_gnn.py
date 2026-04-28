"""
Standalone GNN training script matching TANS's evaluation protocol.

Decoupled from the LLM description-generation pipeline. Loads pre-computed
embeddings and runs multi-seed GNN training with mean +/- std aggregation.

Usage:
    # TANS baseline (high-label)
    python train_gnn.py --data cora --use_params --debug

    # TANS baseline (low-label)
    python train_gnn.py --data usa --use_params --label_setting number --debug

    # Evaluate GES embeddings (low-label)
    python train_gnn.py --data usa --use_params --label_setting number \
        --emb_file embeddings/usa/GES_minilm.pt --debug
"""

import os
import os.path as osp
import sys
import copy
import argparse
import json
import random
import datetime
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.transforms import ToUndirected
from torch_geometric.datasets import Airports
from torch_geometric.utils import is_undirected, to_dense_adj, degree

# Verified-identical modules from src/
from src.model.model import GNN
from src.utils.eval import evaluate, dataset2metric
from src.utils.early_stop import EarlyStopping
from src.utils.logger import Logger
from src.utils.utils import seed_everything


# ---------------------------------------------------------------------------
# Data loading (TANS-style: pre-computed .pt embeddings)
# ---------------------------------------------------------------------------

CITATION_NETWORKS = ['cora', 'pubmed']
AIRPORT_NETWORKS = ['usa', 'europe', 'brazil']


def resolve_emb_path(params, dataset_name):
    """Return path to the embedding .pt file. --emb_file overrides auto path."""
    emb_file = params.get('emb_file', 'none')
    if emb_file not in ('none', 'None', None, ''):
        return emb_file

    data_dir = params['data_path']
    has_node_text = dataset_name in CITATION_NETWORKS

    if has_node_text:
        text_path = '_'.join(params['node_text'] + [params['emb_method'], params['emb_model']])
    else:
        text_path = '_'.join([params['emb_method'], params['emb_model']])

    if params.get('wo_neigh', False):
        text_path += '_wo_neigh'

    return osp.join(data_dir, 'text_emb', dataset_name, f'{text_path}.pt')


def get_data(params, dataset_name):
    if dataset_name in CITATION_NETWORKS:
        return citation_graph(params, dataset_name)
    elif dataset_name in AIRPORT_NETWORKS:
        return airport_graph(params, dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def citation_graph(params, dataset_name):
    data_dir = params['data_path']
    path = osp.join(data_dir, 'dataset', dataset_name, f'{dataset_name}.pt')
    data = torch.load(path, weights_only=False)
    data = ToUndirected()(data)
    data.name = dataset_name

    emb_path = resolve_emb_path(params, dataset_name)
    print(f"Loading graph from {path}")
    print(f"Loading embeddings from {emb_path}")
    text_emb = torch.load(emb_path, weights_only=True).to(torch.float32)
    data.x = text_emb

    data.num_nodes = data.x.shape[0]
    data.num_dim = data.x.shape[1]
    data.num_edges = data.num_edges
    data.num_classes = data.y.max().item() + 1
    print(f"Dataset: {dataset_name}, #Nodes: {data.num_nodes}, "
          f"#Edges: {data.num_edges}, #Classes: {data.num_classes}")
    return data


def airport_graph(params, dataset_name):
    data_dir = params['data_path']
    data = Airports(root=osp.join(data_dir, 'dataset', 'airports'), name=dataset_name)[0]

    emb_path = resolve_emb_path(params, dataset_name)
    print(f"Loading embeddings from {emb_path}")
    text_emb = torch.load(emb_path, weights_only=True).to(torch.float32)
    data.x = text_emb

    data = ToUndirected()(data)
    data.num_dim = data.x.shape[1]
    data.num_classes = data.y.max().item() + 1
    data.num_nodes = data.x.shape[0]
    data.num_edges = data.edge_index.shape[1]
    data.name = dataset_name
    print(f"Dataset: {dataset_name}, #Nodes: {data.num_nodes}, "
          f"#Edges: {data.num_edges}, #Classes: {data.num_classes}")
    return data


# ---------------------------------------------------------------------------
# Data splitting (TANS-style: list of splits, no per-split seeding)
# ---------------------------------------------------------------------------

def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def ratio_split(label, train_ratio=0.1, val_ratio=0.1):
    n = label.shape[0]
    train_num = int(n * train_ratio)
    val_num = int(n * val_ratio)

    perm = torch.as_tensor(np.random.permutation(n))
    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + val_num]
    test_indices = perm[train_num + val_num:]

    return {
        'train': idx2mask(train_indices, n),
        'val': idx2mask(val_indices, n),
        'test': idx2mask(test_indices, n),
    }


def num_split(label, train_num=20, val_num=50):
    train_indices, val_indices, test_indices = [], [], []
    num_classes = label.max() + 1

    for c in range(num_classes):
        idx = torch.where(label == c)[0]
        idx = idx[torch.randperm(idx.size(0))]
        train_indices.append(idx[:train_num])
        val_indices.append(idx[train_num:train_num + val_num])
        test_indices.append(idx[train_num + val_num:])

    train_indices = torch.cat(train_indices, dim=0)
    val_indices = torch.cat(val_indices, dim=0)
    test_indices = torch.cat(test_indices, dim=0)

    n = label.shape[0]
    return {
        'train': idx2mask(train_indices, n),
        'val': idx2mask(val_indices, n),
        'test': idx2mask(test_indices, n),
    }


def get_split(label, params):
    """Generate a list of splits (one per seed). No per-split seeding (matches TANS)."""
    splits = []
    for _ in params['seed']:
        if params['label_setting'] == 'ratio':
            split = ratio_split(label, params['train_ratio'], params['val_ratio'])
        elif params['label_setting'] == 'number':
            split = num_split(label, params['train_num'], params['val_num'])
        else:
            raise ValueError('Unknown label setting')
        splits.append(split)
    return splits


# ---------------------------------------------------------------------------
# Training and evaluation (TANS-style: no DropEdge, simple eval)
# ---------------------------------------------------------------------------

def flip_edges(data, p=0.2):
    """Randomly flip edges (TANS noise injection)."""
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]

    if is_undirected(data.edge_index):
        num_flip = int(num_edges * p / 2)
    else:
        num_flip = int(num_edges * p)

    adj = to_dense_adj(data.edge_index)[0]
    flipped = torch.randint(0, num_nodes, size=(num_flip, 2))
    for n1, n2 in flipped:
        adj[n1, n2] = 1 - adj[n1, n2]
        adj[n2, n1] = 1 - adj[n2, n1]

    data.edge_index = adj.to_sparse().coalesce().indices()
    data.edge_attr = None
    return data


def train(model, data, split, optimizer, params, soft_labels=None):
    model.train()
    device = next(model.parameters()).device

    train_mask = split["train"]
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y[train_mask].squeeze().to(device)

    distill_alpha = params.get('distill_alpha', 0.0)
    distill_temp = params.get('distill_temp', 1.0)
    use_distill = distill_alpha > 0 and soft_labels is not None

    if use_distill:
        # Encode ALL nodes for distillation on unlabeled nodes
        z_all = model.encode(x, edge_index)
        y_pred_all = model.predict(z_all)
        y_pred = y_pred_all[train_mask]
    else:
        z = model.encode(x, edge_index)[train_mask]
        y_pred = model.predict(z)

    label_smoothing = params.get('label_smoothing', 0.0)
    ce_loss = F.cross_entropy(y_pred, y, label_smoothing=label_smoothing)

    loss = ce_loss
    if use_distill:
        # KL divergence ONLY on unlabeled nodes where LLM is confident
        unlabeled = ~(split['train'] | split['val'])
        teacher_all = soft_labels.to(device)
        conf_threshold = params.get('distill_conf', 0.0)
        if conf_threshold > 0:
            # Only distill on nodes where LLM max prob > threshold
            max_conf = teacher_all.max(dim=-1).values
            confident = max_conf > conf_threshold
            distill_mask = unlabeled & confident.cpu()
        else:
            distill_mask = unlabeled
        if distill_mask.any():
            log_student = F.log_softmax(y_pred_all[distill_mask] / distill_temp, dim=-1)
            teacher = teacher_all[distill_mask]
            if distill_temp != 1.0:
                teacher = torch.pow(teacher + 1e-8, 1.0 / distill_temp)
                teacher = teacher / teacher.sum(dim=-1, keepdim=True)
            kl_loss = F.kl_div(log_student, teacher, reduction='batchmean')
            loss = ce_loss + distill_alpha * kl_loss

    optimizer.zero_grad()
    loss.backward()

    grad_clip = params.get('grad_clip', 0.0)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    optimizer.step()

    return loss.item()


def eval_model(model, data, split, params):
    device = next(model.parameters()).device

    train_mask, val_mask, test_mask = split["train"], split["val"], split["test"]

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    tta_runs = params.get('tta_eval', 0)
    if tta_runs > 0:
        # Test-time augmentation: average predictions over multiple dropout masks
        model.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(tta_runs):
                z = model.encode(x, edge_index)
                preds.append(model.predict(z))
        y_pred = torch.stack(preds).mean(dim=0)
    else:
        model.eval()
        with torch.no_grad():
            z = model.encode(x, edge_index)
            y_pred = model.predict(z)

    # Label propagation post-processing
    lp_alpha = params.get('lp_alpha', 0.0)
    lp_iters = params.get('lp_iters', 0)
    if lp_alpha > 0 and lp_iters > 0:
        probs = F.softmax(y_pred, dim=-1)
        row, col = edge_index
        deg = degree(col, num_nodes=probs.size(0), dtype=probs.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        for _ in range(lp_iters):
            # Sparse message passing: D^{-1/2} A D^{-1/2} @ probs
            agg = torch.zeros_like(probs)
            agg.index_add_(0, col, probs[row] * norm.unsqueeze(-1))
            probs = (1 - lp_alpha) * F.softmax(y_pred, dim=-1) + lp_alpha * agg
        y_pred = probs

    train_value = evaluate(y_pred, y, train_mask, params)
    val_value = evaluate(y_pred, y, val_mask, params)
    test_value = evaluate(y_pred, y, test_mask, params)

    return {
        'train': train_value,
        'val': val_value,
        'test': test_value,
        'metric': dataset2metric[params['data']],
    }


# ---------------------------------------------------------------------------
# Main training loop (TANS-style multi-seed)
# ---------------------------------------------------------------------------

def run(params):
    device = torch.device(f"cuda:{params['device']}" if torch.cuda.is_available() else "cpu")

    graph = get_data(params, params['data'])
    splits = get_split(graph.y, params)

    # Load LLM soft labels for distillation / pseudo-labeling
    soft_labels = None
    distill_file = params.get('distill_file', 'none')
    if distill_file not in ('none', 'None', None, ''):
        soft_labels = torch.load(distill_file, weights_only=True).to(torch.float32)
        print(f"Loaded soft labels from {distill_file}, shape={soft_labels.shape}")
        assert soft_labels.shape[0] == graph.num_nodes, \
            f"Soft labels ({soft_labels.shape[0]}) != num_nodes ({graph.num_nodes})"

    model = GNN(
        input_dim=graph.num_dim,
        hidden_dim=params['hidden_dim'],
        output_dim=graph.num_classes,
        activation=nn.PReLU,
        num_layers=params['num_layers'],
        backbone=params['backbone'],
        normalize=params['normalize'],
        dropout=params['dropout'],
    ).to(device)

    logger = Logger()

    for idx, (seed, split) in enumerate(zip(params['seed'], splits)):
        seed_everything(seed)
        data = copy.deepcopy(graph)

        if params['feat_noise'] != 0:
            data.x = (1 - params['feat_noise']) * data.x + params['feat_noise'] * torch.randn_like(data.x)

        if params['edge_noise'] != 0:
            data = flip_edges(data, p=params['edge_noise'])

        # LLM pseudo-labeling: add LLM predictions as training data
        if params.get('llm_pseudo', False) and soft_labels is not None:
            llm_conf = params.get('llm_pseudo_conf', 0.7)
            max_probs, llm_preds = soft_labels.max(dim=-1)
            unlabeled = ~(split['train'] | split['val'])
            confident = max_probs > llm_conf
            pseudo_mask = unlabeled & confident
            n_pseudo = pseudo_mask.sum().item()
            if n_pseudo > 0:
                data.y[pseudo_mask] = llm_preds[pseudo_mask]
                split = copy.deepcopy(split)
                split['train'] = split['train'] | pseudo_mask
                if idx == 0:
                    print(f"  LLM pseudo-labels: added {n_pseudo} nodes (conf>{llm_conf})")

        st_rounds = params.get('self_train_rounds', 0)
        st_threshold = params.get('self_train_threshold', 0.9)
        current_split = copy.deepcopy(split)

        for st_round in range(max(1, st_rounds + 1)):
            task_model = copy.deepcopy(model).to(device)
            optimizer = AdamW(task_model.parameters(), lr=params['lr'], weight_decay=params['decay'])
            stopper = EarlyStopping(patience=params['early_stop'])

            scheduler = None
            if params.get('lr_schedule', 'none') == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=params['epochs'], eta_min=params['lr'] * 0.01)

            for epoch in range(1, params['epochs'] + 1):
                loss = train(task_model, data, current_split, optimizer, params, soft_labels=soft_labels)
                if scheduler is not None:
                    scheduler.step()
                result = eval_model(task_model, data, current_split, params)

                is_stop = stopper(result)
                logger.log(idx, epoch, loss, result)
                if is_stop:
                    break

            # Self-training: add pseudo-labels from high-confidence predictions
            if st_round < st_rounds:
                task_model.eval()
                with torch.no_grad():
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    z = task_model.encode(x, edge_index)
                    logits = task_model.predict(z)
                    probs = F.softmax(logits, dim=-1)
                    confs, preds = probs.max(dim=-1)

                # Select unlabeled nodes with high confidence
                unlabeled = ~(current_split['train'] | current_split['val'])
                high_conf = confs.cpu() > st_threshold
                pseudo_mask = unlabeled & high_conf
                n_pseudo = pseudo_mask.sum().item()

                if n_pseudo == 0:
                    break

                # Update labels and training mask
                data.y[pseudo_mask] = preds[pseudo_mask].cpu()
                current_split = copy.deepcopy(current_split)
                current_split['train'] = current_split['train'] | pseudo_mask
                print(f"    Self-train round {st_round+1}: added {n_pseudo} pseudo-labels")

            if not params.get('no_wandb', True):
                import wandb
                wandb.log({
                    "train/loss_train": loss,
                    "train/train": result['train'],
                    "train/val": result['val'],
                    "train/test": result['test'],
                    "train/metric": result['metric'],
                })

        single_best = logger.get_single_best(idx)
        print(f"  Seed {seed}: train={single_best['train']:.2f}  "
              f"val={single_best['val']:.2f}  test={single_best['test']:.2f}  "
              f"(epoch {single_best.get('epoch', '?')})")

        if not params.get('no_wandb', True):
            import wandb
            wandb.log({
                "best/train": single_best["train"],
                "best/val": single_best["val"],
                "best/test": single_best["test"],
            })

    best = logger.get_best()

    print(f"\n{'='*50}")
    print(f"Results ({len(params['seed'])} seeds):")
    for s in ('train', 'val', 'test'):
        print(f"  {s}: {best[s]['mean']:.2f} +/- {best[s]['std']:.2f}")
    print(f"{'='*50}")

    if not params.get('no_wandb', True):
        import wandb
        wandb.log({
            "final/train": f"{best['train']['mean']:.2f} +/- {best['train']['std']:.2f}",
            "final/val": f"{best['val']['mean']:.2f} +/- {best['val']['std']:.2f}",
            "final/test": f"{best['test']['mean']:.2f} +/- {best['test']['std']:.2f}",
            "final/train_mean": best['train']['mean'],
            "final/val_mean": best['val']['mean'],
            "final/test_mean": best['test']['mean'],
            "final/train_std": best['train']['std'],
            "final/val_std": best['val']['std'],
            "final/test_std": best['test']['std'],
        })
        wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})
        wandb.finish()

    save_results(params, best, logger)
    return best


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(params, best, logger):
    output_dir = params.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    emb_file = params.get('emb_file', 'none')
    if emb_file in ('none', 'None', None, ''):
        emb_tag = f"{params['emb_method']}_{params['emb_model']}"
    else:
        emb_tag = osp.basename(emb_file).replace('.pt', '')

    filename = f"{params['data']}_{params['backbone']}_{emb_tag}_{timestamp}.json"
    filepath = osp.join(output_dir, filename)

    result_dict = {
        'dataset': params['data'],
        'backbone': params['backbone'],
        'emb_method': params['emb_method'],
        'emb_model': params['emb_model'],
        'emb_file': emb_file if emb_file not in ('none', 'None', None, '') else 'auto',
        'seeds': params['seed'],
        'label_setting': params['label_setting'],
        'config': {
            'hidden_dim': params['hidden_dim'],
            'num_layers': params['num_layers'],
            'dropout': params['dropout'],
            'lr': params['lr'],
            'decay': params['decay'],
            'normalize': params['normalize'],
            'epochs': params['epochs'],
            'early_stop': params['early_stop'],
        },
        'train': {'mean': round(best['train']['mean'], 2),
                  'std': round(best['train']['std'], 2)},
        'val': {'mean': round(best['val']['mean'], 2),
                'std': round(best['val']['std'], 2)},
        'test': {'mean': round(best['test']['mean'], 2),
                 'std': round(best['test']['std'], 2)},
        'per_seed': {},
    }

    for idx, seed in enumerate(params['seed']):
        sb = logger.get_single_best(idx)
        result_dict['per_seed'][str(seed)] = {
            'train': sb['train'],
            'val': sb['val'],
            'test': sb['test'],
            'epoch': sb.get('epoch', -1),
        }

    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to {filepath}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser('GNN Training (TANS-compatible)')

    # General
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--use_params', action='store_true', help='Load tuned hyperparams from config')
    parser.add_argument('--config', type=str, default='src/config/finetune.yaml',
                        help='Path to hyperparameter config YAML (relative to repo root)')
    parser.add_argument('--group', type=str, default='base')
    parser.add_argument('--debug', action='store_true', help='Disable W&B, run locally')
    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--device', type=int, default=0)

    # Embedding
    parser.add_argument('--node_text', type=str, nargs='+', default=['title', 'abstract'])
    parser.add_argument('--emb_model', type=str, default='minilm')
    parser.add_argument('--emb_method', type=str, default='TANS')
    parser.add_argument('--emb_file', type=str, default='none',
                        help='Explicit path to embedding .pt file (overrides auto path)')
    parser.add_argument('--wo_neigh', action='store_true', help='Append _wo_neigh to auto path')

    # Model
    parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'mlp'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--normalize', type=str, default='none')

    # Training
    parser.add_argument('--label_setting', type=str, default='ratio', choices=['ratio', 'number'])
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--train_num', type=int, default=20)
    parser.add_argument('--val_num', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--decay', type=float, default=1e-5)

    # Training tricks
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['none', 'cosine'])
    parser.add_argument('--tta_eval', type=int, default=0, help='Number of TTA forward passes (0=disabled)')
    parser.add_argument('--self_train_rounds', type=int, default=0, help='Self-training rounds (0=disabled)')
    parser.add_argument('--self_train_threshold', type=float, default=0.9, help='Confidence threshold for pseudo-labels')
    parser.add_argument('--lp_alpha', type=float, default=0.0, help='Label propagation alpha (0=disabled)')
    parser.add_argument('--lp_iters', type=int, default=0, help='Label propagation iterations')

    # Distillation from LLM soft labels
    parser.add_argument('--distill_file', type=str, default='none',
                        help='Path to .pt file with LLM soft label probabilities (N x C)')
    parser.add_argument('--distill_alpha', type=float, default=0.0,
                        help='Weight for distillation KL loss (0=disabled)')
    parser.add_argument('--distill_temp', type=float, default=1.0,
                        help='Temperature for distillation softmax')
    parser.add_argument('--distill_conf', type=float, default=0.0,
                        help='Min LLM confidence to include in distillation (0=all)')
    parser.add_argument('--llm_pseudo', action='store_true',
                        help='Use LLM predictions as pseudo-labels for confident unlabeled nodes')
    parser.add_argument('--llm_pseudo_conf', type=float, default=0.7,
                        help='Min LLM confidence for pseudo-labeling')

    # Noise (TANS compat)
    parser.add_argument('--feat_noise', type=float, default=0.0)
    parser.add_argument('--edge_noise', type=float, default=0.0)

    # Output
    parser.add_argument('--output_dir', type=str, default='results')

    args = parser.parse_args()
    return vars(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    params = get_args()
    params['data_path'] = osp.join(str(REPO_ROOT), 'datasets')

    dataset = params['data']
    emb_method = params['emb_method']
    has_node_text = dataset in CITATION_NETWORKS

    if params['use_params']:
        config_path = params['config']
        if not osp.isabs(config_path):
            config_path = osp.join(str(REPO_ROOT), config_path)
        with open(config_path, 'r') as f:
            default_params = yaml.safe_load(f)

        node_text_key = '_'.join(params['node_text'])
        if has_node_text:
            params.update(default_params[emb_method][node_text_key][dataset])
        else:
            params.update(default_params[emb_method][dataset])

    # W&B
    no_wandb = params.get('debug', False)
    params['no_wandb'] = no_wandb

    if not no_wandb:
        import wandb
        wandb.init(
            project="TANS",
            name=f"Data:{dataset} | Backbone:{params['backbone']}",
            config=params,
            mode="online",
            group=params['group'],
        )
        params = dict(wandb.config)
        params['no_wandb'] = False

    print(params)
    run(params)


if __name__ == "__main__":
    main()
