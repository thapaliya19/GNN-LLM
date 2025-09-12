# data.py
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.datasets import Airports

from sentence_transformers import SentenceTransformer
import tqdm

from src.utils.consts import SBERT_MODELS

citation_networks = ['cora', 'pubmed']
airport_networks = ['usa', 'europe', 'brazil']

def encode_text_sbert(text, model):
    """Encode text using sentence-bert model"""
    return model.encode(text)


def check_path(path):
    """Create directory if it doesn't exist"""
    if not osp.exists(path):
        os.makedirs(path)


def get_appended_text(data_name, data, params):
    """Get default text features for nodes and append them to the text descriptions"""
    
    node_types = {
        'cora': 'paper',
        'pubmed': 'paper',
        'usa': 'airport',
        'europe': 'airport',
        'brazil': 'airport'
    }

    print(params['node_text'])
    if data_name in ['usa', 'europe', 'brazil']:
        default_text = data.topo_descriptions
    else:
        # if params['node_text] is list use as is, if it is str, then split
        if isinstance(params['node_text'], str):
            params.update({'node_text': params['node_text'].split('_')})
        path = osp.join(params['data_path'], 'dataset', data_name)
        default_text = [np.load(osp.join(path, f'{t}.npy')) for t in params['node_text']]
        default_text = list(zip(*default_text))
        default_text = [' '.join(t) for t in default_text]

        # Combine texts
    texts = []
    template = 'The node type is {}. The node description is {}. The additional node description is {}.'
    if params['desc_mode'] == 'llm_only':
        texts = data.text_descriptions
    elif params['desc_mode'] == 'orig_only':
        texts = default_text
    else:
        for i in range(len(data.text_descriptions)):
            t = default_text[i] if data_name in ['cora', 'pubmed'] else 'na'
            add_t = data.text_descriptions[i]
            texts.append(template.format(node_types[data_name], t, add_t))

    print(f"Desc mode: {params['desc_mode']}")
    print("Example node text: \n", texts[0]) # print first text for debugging
        
    return texts, default_text
    



def get_text_embeddings(texts, model):
    """Get embeddings for a list of texts"""
    embeddings = []
    for text in tqdm.tqdm(texts):
        embeddings.append(encode_text_sbert(text, model))
    return torch.tensor(embeddings, dtype=torch.float64)


def get_data(params, dataset_name):
    if dataset_name in citation_networks:
        return citation_graph(params, dataset_name)
    elif dataset_name in airport_networks:
        return airport_graph(params, dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def citation_graph(params, dataset_name):
    data_dir = params['data_path']
    # load graph
    path = osp.join(data_dir, 'dataset', dataset_name, f'{dataset_name}.pt')
    data = torch.load(path, weights_only=False)
    data = ToUndirected()(data)
    data.name = dataset_name

    # load topological features
    topo_df = pd.read_csv(params['topo_csv'], sep=',')
    data.topo_descriptions = topo_df['topo_features'].tolist()

    # load text descriptions from CSV
    desc_csv = params['desc_csv']
    df = pd.read_csv(desc_csv, sep='\t')
    data.text_descriptions = df['answer'].tolist()

    texts, default_text = get_appended_text(dataset_name, data, params)
    data.text_descriptions = texts

    sbert = SentenceTransformer(SBERT_MODELS[params["emb_model"]])
    new_embs = [sbert.encode(txt) for txt in texts]
    data.x = torch.tensor(new_embs, dtype=torch.float32)

    # metadata
    data.num_nodes, data.num_dim = data.x.shape
    data.num_edges = data.num_edges
    data.num_classes = int(data.y.max().item() + 1)
    print(f"Loaded {dataset_name}: #nodes={data.num_nodes}, #edges={data.num_edges}, #classes={data.num_classes}")
    return data


def airport_graph(params, dataset_name):
    data_dir = params['data_path']
    data = Airports(root=osp.join(data_dir, 'dataset', 'airports'), name=dataset_name)[0]
    data = ToUndirected()(data)
    data.name = dataset_name

    # node_text = params['node_text']
    # if params['emb_file'] == 'none':
    #     emb_file = osp.join(data_dir, 'text_emb', dataset_name, f"{node_text}_{params['emb_method']}_{params['llm']}_{params['emb_model']}.pt")
    # else:
    #     emb_file = params['emb_file']
    # data.x = torch.load(emb_file).to(torch.float32)

    # load topological features
    topo_df = pd.read_csv(params['topo_csv'], sep=',')
    data.topo_descriptions = topo_df['topo_features'].tolist()

    desc_csv = params['desc_csv']
    df = pd.read_csv(desc_csv, sep='\t')
    data.text_descriptions = df['answer'].tolist()

    texts, default_text = get_appended_text(dataset_name, data, params)

    data.text_descriptions = texts

    sbert = SentenceTransformer(SBERT_MODELS[params["emb_model"]])
    new_embs = [sbert.encode(txt) for txt in texts]
    data.x = torch.tensor(new_embs, dtype=torch.float32)

    data.num_nodes = data.x.shape[0]
    data.num_edges = data.edge_index.shape[1]
    data.num_dim = data.x.shape[1]
    data.num_classes = int(data.y.max().item() + 1)
    return data


def degree_bucketing(num_nodes, in_degree, max_degree=32):
    features = torch.zeros([num_nodes, max_degree])
    for i in range(num_nodes):
        d = in_degree[i] if i < len(in_degree) else 0
        features[i][min(d, max_degree-1)] = 1
    return features
