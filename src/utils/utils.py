import datetime
import numpy as np
import os
import os.path as osp
import random
from pathlib import Path
import json

import string

import torch

from torch_geometric.utils import (
    remove_self_loops,
    k_hop_subgraph,
    coalesce,
    to_edge_index,
    to_torch_coo_tensor,
    is_undirected,
    to_dense_adj,
)
from torch_geometric.data import Data

EPS = 1e-6


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )
    return post_fix


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_MB(byte):
    return byte / 1024.0 / 1024.0


def combine_dicts(dicts, decimals=2):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    final_result = {}
    for key, value in result.items():
        if isinstance(value[0], list):
            final_result[key + "_mean"] = np.round(np.mean(value, axis=0), decimals)
            final_result[key + "_std"] = np.round(np.std(value, axis=0), decimals)
        else:
            final_result[key + "_mean"] = np.round(np.mean(value), decimals)
            final_result[key + "_std"] = np.round(np.std(value), decimals)

    return final_result


def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_mask(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.1):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        "train": indices[:train_size],
        "valid": indices[train_size : test_size + train_size],
        "test": indices[test_size + train_size :],
    }


def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def extract_graphs(data, mask, k=2):
    graphs = list()
    for i in torch.where(mask == True)[0]:
        subgraph = k_hop_subgraph(
            i.item(), num_hops=k, edge_index=data.edge_index, relabel_nodes=True
        )
        graphs.append(Data(x=data.x[subgraph[0]], edge_index=subgraph[1], y=data.y[i]))
    return graphs


def get_k_hop_graphs(data, k, use_self_loop=True):
    edge_index, edge_attr = data.edge_index, data.edge_attr
    N = data.x.shape[0]

    if k == 0:
        # return adjacent matrix only with self-loop
        edge_index = [list(range(N)), list(range(N))]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # edge_attr = torch.ones(edge_index.shape[1], device=edge_index.device)
        return edge_index, None

    adj = to_torch_coo_tensor(edge_index, size=(N, N))
    adj_base = adj.clone()

    for _ in range(k - 1):
        new_edge_index, _ = to_edge_index(adj_base @ adj)
        new_edge_index, _ = remove_self_loops(new_edge_index)

        edge_index = torch.cat([edge_index, new_edge_index], dim=1)

    if edge_attr is None:
        # edge_attr = torch.ones(edge_index.shape[1], device=edge_index.device)
        pass

    if not use_self_loop:
        edge_index, _ = remove_self_loops(edge_index)

    return coalesce(edge_index, edge_attr, N)


def random_string(k=16):
    random_string = str.join(
        "", random.choices(string.ascii_letters + string.digits, k=k)
    )
    return random_string


def flip_edges(data, p=0.2):
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]

    if is_undirected(data.edge_index):
        num_flip_edges = int(num_edges * p / 2)
    else:
        num_flip_edges = int(num_edges * p)

    adj = to_dense_adj(data.edge_index)[0]

    flipped_edges = torch.randint(0, num_nodes, size=(num_flip_edges, 2))

    for n1, n2 in flipped_edges:
        adj[n1, n2] = 1 - adj[n1, n2]
        adj[n2, n1] = 1 - adj[n2, n1]

    edge_index = adj.to_sparse().coalesce().indices()
    data.edge_index = edge_index
    data.edge_attr = None
    return data


def get_device_from_model(model):
    return next(model.parameters()).device


def get_device(params, optimized_params=None):
    if optimized_params is None or len(optimized_params) == 0:
        device = torch.device(f"cuda:{params['device']}")
    else:
        device = torch.device(f"cuda")
    return device
