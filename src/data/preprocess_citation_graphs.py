#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import random
import numpy as np
import pandas as pd

import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Generate text data for graph nodes')
    parser.add_argument('--data_name', type=str, default='cora', choices=['cora', 'pubmed'],
                        help='dataset name')
    parser.add_argument('--setting', type=str, default='text_limit', choices=['text_limit', 'text_rich'],
                        help='text setting')
    parser.add_argument('--mode', type=str, default='ours', choices=['ours', 'tans'],
                        help='ours: topology-only summary for our TTA prompt; '
                             'tans: full TANS-style prompt with node text, neighbor text, and classification instruction')
    parser.add_argument('--num_neighbors', type=int, default=5,
                        help='number of random 1-hop neighbors to include (tans mode only)')

    return parser.parse_args()


def check_dir(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


# Constants and templates
graph_stats = {
    'cora': {'graph_type': 'citation network', 'node_type': 'paper', 'edge_type': 'citation between papers',
             'total_node': 2708, 'total_edge': 5429},
    'pubmed': {'graph_type': 'citation network', 'node_type': 'paper', 'edge_type': 'citation between papers',
               'total_node': 19717, 'total_edge': 44324},
}

# Class lists for classification instruction (tans mode)
class_lists = {
    'cora': "['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory']",
    'pubmed': "['Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2', 'Experimental Diabetes']",
}

template = {
    'base': "Given a node from a {} graph, where the node type is {} with {} nodes, and the edge type is {} with {} edges. ",
    'property': 'The value of property "{}" is {:.4f}, ranked at {} among {} nodes. ',
    'node_text': "The textual node name is {}. The node description is {}. ",
    'neighbor_intro': "The following are the connectivity information of the similar nodes: ",
    'neighbor': "One similar node has name {}, and the description is {}. ",
    'final': "Output the potential three classes of the node and provide reasons for your assessment. The classes are {}. Your answer should be less than 200 words. ",
}


def load_data(args):
    """Load and prepare node data"""
    names = np.load(osp.join(osp.dirname(__file__), f'../../datasets/dataset/{args.data_name}/title.npy'),
                    allow_pickle=True)
    descriptions = np.load(osp.join(osp.dirname(__file__), f'../../datasets/dataset/{args.data_name}/abstract.npy'),
                           allow_pickle=True) if args.setting == 'text_rich' else ['na'] * len(names)
    return names, descriptions


def load_one_hop(args):
    """Load 1-hop neighbor lists"""
    path = osp.join(osp.dirname(__file__), f'../../datasets/property/{args.data_name}_one_hop.npy')
    return np.load(path, allow_pickle=True).item()


def process_topological_features(args):
    """Process and rank topological features"""
    topo_features = np.load(osp.join(osp.dirname(__file__), f'../../datasets/property/{args.data_name}_topo.npy'),
                            allow_pickle=True).item()
    rename = {
        'clustering': 'Clustering Coefficient',
        'degree': 'Node Degree',
        'square_clustering': 'Square Clustering Coefficient',
        'closeness': 'Closeness Centrality',
        'betweenness': 'Betweenness Centrality',
    }
    topo_features = {rename[k]: v for k, v in topo_features.items()}

    # Calculate ranks
    topo_features_rank = {}
    for method, values in topo_features.items():
        sorted_values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))
        rank = 0
        pre_value = -1
        node2rank = {}
        for node, value in sorted_values.items():
            if value != pre_value:
                rank += 1
                pre_value = value
            node2rank[node] = rank
        topo_features_rank[method] = node2rank

    # Restructure features
    new_topo_features = {}
    for method, values in topo_features.items():
        for node, value in values.items():
            if node not in new_topo_features:
                new_topo_features[node] = {}
            new_topo_features[node][method] = (value, topo_features_rank[method][node])

    return new_topo_features


def generate_text(node, methods, names, descriptions, args, one_hop=None):
    """Generate text description for a single node.

    mode='ours': topology-only summary (base + property lines).
    mode='tans': full TANS prompt (base + node text + neighbor text + properties + classification instruction).
    """
    stats = graph_stats[args.data_name]

    text = template['base'].format(
        stats['graph_type'],
        stats['node_type'],
        stats['total_node'],
        stats['edge_type'],
        stats['total_edge']
    )

    if args.mode == 'tans':
        # Node title + abstract (TANS includes this inline)
        text += template['node_text'].format(names[node], descriptions[node])

        # 1-hop neighbor text
        if one_hop is not None and node in one_hop:
            neighbors = one_hop[node]
            if len(neighbors) > 0:
                text += template['neighbor_intro']
                sampled = random.sample(neighbors, min(args.num_neighbors, len(neighbors)))
                for nb in sampled:
                    text += template['neighbor'].format(names[nb], descriptions[nb])

    # Topological properties (included in both modes)
    for method, (value, rank) in methods.items():
        text += template['property'].format(method, value, rank, stats['total_node'])

    if args.mode == 'tans':
        # Classification instruction
        text += template['final'].format(class_lists[args.data_name])

    return text

def main():
    args = parse_args()
    names, descriptions = load_data(args)
    topo_features = process_topological_features(args)

    one_hop = load_one_hop(args) if args.mode == 'tans' else None

    # Generate texts for all nodes with progress bar
    texts = []


    for node, methods in tqdm(topo_features.items(), desc="Generating texts", total=len(topo_features)):
        texts.append(generate_text(node, methods, names, descriptions, args, one_hop=one_hop))

    # Create and save dataframe
    df = pd.DataFrame({
        'topo_features': texts,
        'node_idx': range(len(texts))
    })

    save_dir = osp.join(osp.dirname(__file__), '../../datasets/response', args.setting, args.data_name)
    check_dir(save_dir)

    topo_features_file =  'topo_features.csv'

    df[['node_idx', 'topo_features']].to_csv(osp.join(save_dir, topo_features_file), index=False)


if __name__ == "__main__":
    main()
