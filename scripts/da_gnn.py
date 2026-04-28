"""
Standalone Domain Adaptation GNN script using pre-computed embeddings.
Trains on SOURCE dataset, evaluates on TARGET dataset.
Both datasets must have the same number of classes.

Usage:
    python da_gnn.py \
        --pt_data usa --data europe \
        --pt_emb_file embeddings/usa/GES_minilm.pt \
        --emb_file embeddings/europe/GES_minilm.pt \
        --backbone gcn --hidden_dim 128 --num_layers 2 --dropout 0.5 \
        --epochs 1000 --lr 0.005 --decay 0.0001 \
        --label_setting number --seed 1 2 3 4 5 6 7 8 9 10 --debug
"""

import os
import os.path as osp
import sys
import copy
import argparse
import json
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.transforms import ToUndirected
from torch_geometric.datasets import Airports

from src.model.model import GNN
from src.utils.eval import evaluate, dataset2metric
from src.utils.early_stop import EarlyStopping
from src.utils.logger import Logger
from src.utils.utils import seed_everything


CITATION_NETWORKS = ['cora', 'pubmed']
AIRPORT_NETWORKS = ['usa', 'europe', 'brazil']


def load_graph(dataset_name, emb_path, data_dir='datasets'):
    if dataset_name in CITATION_NETWORKS:
        path = osp.join(data_dir, 'dataset', dataset_name, f'{dataset_name}.pt')
        data = torch.load(path, weights_only=False)
        data = ToUndirected()(data)
    elif dataset_name in AIRPORT_NETWORKS:
        data = Airports(root=osp.join(data_dir, 'dataset', 'airports'), name=dataset_name)[0]
        data = ToUndirected()(data)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    text_emb = torch.load(emb_path, weights_only=True).to(torch.float32)
    data.x = text_emb
    data.name = dataset_name
    data.num_nodes = data.x.shape[0]
    data.num_dim = data.x.shape[1]
    data.num_edges = data.edge_index.shape[1]
    data.num_classes = data.y.max().item() + 1
    return data


def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def ratio_split(label, train_ratio=0.6, val_ratio=0.2):
    n = label.shape[0]
    train_num = int(n * train_ratio)
    val_num = int(n * val_ratio)
    perm = torch.as_tensor(np.random.permutation(n))
    return {
        'train': idx2mask(perm[:train_num], n),
        'val': idx2mask(perm[train_num:train_num + val_num], n),
        'test': idx2mask(perm[train_num + val_num:], n),
    }


def num_split(label, train_num=20, val_num=30):
    train_indices, val_indices, test_indices = [], [], []
    num_classes = label.max() + 1
    for c in range(num_classes):
        idx = torch.where(label == c)[0]
        idx = idx[torch.randperm(idx.size(0))]
        train_indices.append(idx[:train_num])
        val_indices.append(idx[train_num:train_num + val_num])
        test_indices.append(idx[train_num + val_num:])
    train_indices = torch.cat(train_indices)
    val_indices = torch.cat(val_indices)
    test_indices = torch.cat(test_indices)
    n = label.shape[0]
    return {
        'train': idx2mask(train_indices, n),
        'val': idx2mask(val_indices, n),
        'test': idx2mask(test_indices, n),
    }


def train_step(model, source, optimizer, device):
    """Train on ALL source nodes."""
    model.train()
    x = source.x.to(device)
    edge_index = source.edge_index.to(device)
    y = source.y.squeeze().to(device)

    z = model.encode(x, edge_index)
    y_pred = model.predict(z)
    loss = F.cross_entropy(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_model(model, target, split, params, device):
    """Evaluate on TARGET using its val/test splits."""
    model.eval()
    x = target.x.to(device)
    edge_index = target.edge_index.to(device)
    y = target.y.squeeze().to(device)

    with torch.no_grad():
        z = model.encode(x, edge_index)
        y_pred = model.predict(z)

    val_value = evaluate(y_pred, y, split['val'], params)
    test_value = evaluate(y_pred, y, split['test'], params)

    return {'train': 0, 'val': val_value, 'test': test_value,
            'metric': dataset2metric[params['data']]}


def run(params):
    device = torch.device(f"cuda:{params['device']}" if torch.cuda.is_available() else "cpu")
    data_dir = params['data_path']

    source = load_graph(params['pt_data'], params['pt_emb_file'], data_dir)
    target = load_graph(params['data'], params['emb_file'], data_dir)

    print(f"Source: {source.name} ({source.num_nodes} nodes, {source.num_classes} classes, {source.num_dim}d)")
    print(f"Target: {target.name} ({target.num_nodes} nodes, {target.num_classes} classes, {target.num_dim}d)")

    assert source.num_dim == target.num_dim
    assert source.num_classes == target.num_classes, \
        f"DA requires same #classes: source={source.num_classes}, target={target.num_classes}"

    # Generate splits for target
    splits = []
    for _ in params['seed']:
        if params['label_setting'] == 'ratio':
            splits.append(ratio_split(target.y, params['train_ratio'], params['val_ratio']))
        else:
            splits.append(num_split(target.y, params['train_num'], params['val_num']))

    logger = Logger()

    for idx, (seed, split) in enumerate(zip(params['seed'], splits)):
        seed_everything(seed)

        model = GNN(
            input_dim=source.num_dim,
            hidden_dim=params['hidden_dim'],
            output_dim=source.num_classes,
            activation=nn.PReLU,
            num_layers=params['num_layers'],
            backbone=params['backbone'],
            normalize=params['normalize'],
            dropout=params['dropout'],
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=params['decay'])
        stopper = EarlyStopping(patience=params['early_stop'])

        for epoch in range(1, params['epochs'] + 1):
            loss = train_step(model, source, optimizer, device)
            result = eval_model(model, target, split, params, device)

            logger.log(idx, epoch, loss, result)
            if stopper(result):
                break

        single_best = logger.get_single_best(idx)
        print(f"  Seed {seed}: val={single_best['val']:.2f}  test={single_best['test']:.2f}  "
              f"(epoch {single_best.get('epoch', '?')})")

    best = logger.get_best()
    print(f"\n{'='*50}")
    print(f"DA {params['pt_data']}→{params['data']} Results ({len(params['seed'])} seeds):")
    for s in ('val', 'test'):
        print(f"  {s}: {best[s]['mean']:.2f} +/- {best[s]['std']:.2f}")
    print(f"{'='*50}")

    # Save results
    output_dir = params.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"da_{params['pt_data']}_{params['data']}_{params['backbone']}_{timestamp}.json"
    filepath = osp.join(output_dir, filename)

    result_dict = {
        'da': f"{params['pt_data']}→{params['data']}",
        'pt_data': params['pt_data'],
        'data': params['data'],
        'backbone': params['backbone'],
        'pt_emb_file': params['pt_emb_file'],
        'emb_file': params['emb_file'],
        'seeds': params['seed'],
        'label_setting': params['label_setting'],
        'config': {
            'hidden_dim': params['hidden_dim'],
            'num_layers': params['num_layers'],
            'dropout': params['dropout'],
            'lr': params['lr'],
            'decay': params['decay'],
            'epochs': params['epochs'],
            'early_stop': params['early_stop'],
        },
        'val': {'mean': round(best['val']['mean'], 2), 'std': round(best['val']['std'], 2)},
        'test': {'mean': round(best['test']['mean'], 2), 'std': round(best['test']['std'], 2)},
    }

    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to {filepath}")

    return best


def get_args():
    parser = argparse.ArgumentParser('GNN Domain Adaptation')

    # Source
    parser.add_argument('--pt_data', type=str, required=True, help='Source dataset')
    parser.add_argument('--pt_emb_file', type=str, required=True, help='Source embedding .pt file')

    # Target
    parser.add_argument('--data', type=str, required=True, help='Target dataset')
    parser.add_argument('--emb_file', type=str, required=True, help='Target embedding .pt file')

    # Model
    parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'mlp'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalize', type=str, default='none')

    # Training
    parser.add_argument('--label_setting', type=str, default='number', choices=['ratio', 'number'])
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--train_num', type=int, default=20)
    parser.add_argument('--val_num', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--decay', type=float, default=1e-4)

    # General
    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return vars(args)


def main():
    params = get_args()
    params['data_path'] = osp.join(str(REPO_ROOT), 'datasets')
    print(params)
    run(params)


if __name__ == "__main__":
    main()
