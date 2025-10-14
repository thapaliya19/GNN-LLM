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


# --- robust TSV open that tolerates weird bytes in the path ---
import os, io, sys, re, subprocess, glob
from pathlib import Path
import pandas as pd

_CTRL = re.compile(r'[\x00-\x1F\x7F]')

def _clean(s: str) -> str:
    return _CTRL.sub('', s)

def _resolve_by_iterdir(p: Path) -> Path | None:
    """Walk each component; if missing, pick the entry whose name matches after cleaning (and with ? as wildcard)."""
    is_abs = p.is_absolute()
    parts = [part for part in str(p).split('/') if part not in ('', '.')]
    cur = Path('/') if is_abs else Path.cwd()
    for part in parts:
        cand = cur / part
        if cand.exists():
            cur = cand
            continue
        # try match by cleaned name
        try:
            entries = list(cur.iterdir())
        except Exception:
            return None
        cleaned_target = _clean(part)
        # exact cleaned match
        for e in entries:
            if _clean(e.name) == cleaned_target:
                cur = e
                break
        else:
            # treat ? as wildcard for a single char (including control chars)
            # build a regex from part where ? -> .
            rx = re.compile('^' + re.escape(part).replace(r'\?', '.') + r'$')
            pick = None
            for e in entries:
                if rx.match(e.name):
                    pick = e
                    break
            if pick is None:
                return None
            cur = pick
    return cur

def read_tsv_weird(path_str: str) -> pd.DataFrame:
    # 0) direct attempt
    p = Path(path_str)
    if p.exists():
        return pd.read_csv(str(p), sep='\t')

    parent, name = os.path.dirname(path_str), os.path.basename(path_str)

    # 1) glob: let '?' act as "one char" wildcard
    #    This frequently finds files that have an actual '\r' where your string has '?'
    hits = glob.glob(os.path.join(parent, name))
    if len(hits) == 1 and os.path.isfile(hits[0]):
        return pd.read_csv(hits[0], sep='\t')
    elif len(hits) > 1:
        # If multiple, prefer exact cleaned-name match
        cleaned = _clean(name)
        for h in hits:
            if _clean(os.path.basename(h)) == cleaned and os.path.isfile(h):
                return pd.read_csv(h, sep='\t')

    # 2) component-by-component resolution
    resolved = _resolve_by_iterdir(Path(path_str))
    if resolved and resolved.exists():
        return pd.read_csv(str(resolved), sep='\t')

    # 3) try a few simple variants for the last component
    variants = set()
    # replace '?' with '\r' (common case)
    variants.add(os.path.join(parent, name.replace('?', '\r')))
    # remove '?' entirely
    variants.add(os.path.join(parent, name.replace('?', '')))
    # also try cleaned parent
    parent_clean = _clean(parent)
    variants.add(os.path.join(parent_clean, name))
    variants.add(os.path.join(parent_clean, name.replace('?', '\r')))

    for v in list(variants):
        if os.path.isfile(v):
            return pd.read_csv(v, sep='\t')

    # 4) last resort: read bytes via subprocess and parse
    try:
        out = subprocess.check_output(['bash', '-lc', f'cat -- {repr(path_str)}'], stderr=subprocess.STDOUT)
        return pd.read_csv(io.BytesIO(out), sep=b'\t', engine='c')
    except Exception as e:
        sys.stderr.write(f"[weird-open] subprocess fallback failed: {e}\n")

    raise FileNotFoundError(f"Could not open (after robust resolution): {path_str!r}")


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
            # t = default_text[i]

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


def get_data(params, dataset_name, desc_csv=None, topo_csv=None):
    if dataset_name in citation_networks:
        return citation_graph(params, dataset_name, desc_csv, topo_csv)
    elif dataset_name in airport_networks:
        return airport_graph(params, dataset_name, desc_csv, topo_csv)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def citation_graph(params, dataset_name, desc_csv=None, topo_csv=None):
    data_dir = params['data_path']
    # load graph
    path = osp.join(data_dir, 'dataset', dataset_name, f'{dataset_name}.pt')
    data = torch.load(path, weights_only=False)
    data = ToUndirected()(data)
    data.name = dataset_name

    # load topological features
    topo_df = pd.read_csv(topo_csv, sep=',')
    data.topo_descriptions = topo_df['topo_features'].tolist()

    # load text descriptions from CSV
    df = read_tsv_weird(desc_csv)
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


def airport_graph(params, dataset_name, desc_csv=None, topo_csv=None):
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
    topo_df = pd.read_csv(topo_csv, sep=',')
    data.topo_descriptions = topo_df['topo_features'].tolist()

    df = read_tsv_weird(desc_csv)
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
