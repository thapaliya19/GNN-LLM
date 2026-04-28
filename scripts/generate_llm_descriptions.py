"""
Generate node descriptions for Cora and Pubmed using a local LLM (Qwen2.5-72B via vLLM).
Produces descriptions in 4 styles: TANS, GES, TAPE, KEA.
Encodes all with MiniLM for GNN experiments.

Usage:
    # Start vLLM server first, then run:
    python generate_llm_descriptions.py --api_base http://localhost:8000/v1
"""
import os, sys, csv, json, argparse, time
from pathlib import Path
import torch
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent

# Label maps
CORA_LABELS = ["Case Based", "Genetic Algorithms", "Neural Networks",
               "Probabilistic Methods", "Reinforcement Learning", "Rule Learning", "Theory"]
PUBMED_LABELS = ["Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2", "Experimental Diabetes"]


def load_dataset(name):
    """Load dataset with raw texts and topo features."""
    data = torch.load(f'datasets/dataset/{name}/{name}.pt', weights_only=False)
    texts = data.raw_texts

    # Load topo features if available
    topo_path = f'datasets/response/text_{"rich" if name in ["cora","pubmed"] else "free"}/{name}/topo_features.csv'
    topos = {}
    if os.path.exists(topo_path):
        with open(topo_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                topos[int(row['node_idx'])] = row['topo_features']

    return data, texts, topos


def generate_batch(client, prompts, model_name, max_tokens=300, temperature=0.0):
    """Generate completions for a batch of prompts."""
    results = []
    for prompt in tqdm(prompts, desc="Generating"):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error: {e}")
            results.append("")
    return results


def build_tans_prompts(dataset_name, texts, topos):
    """Build TANS-style prompts: topology + text -> description with class prediction."""
    labels = CORA_LABELS if dataset_name == 'cora' else PUBMED_LABELS
    graph_info = {
        'cora': "a citation network graph, where the node type is paper with 2708 nodes, and the edge type is citation between papers with 5429 edges",
        'pubmed': "a citation network graph, where the node type is paper with 19717 nodes, and the edge type is citation between papers with 44338 edges",
    }

    prompts = []
    for i in range(len(texts)):
        topo = topos.get(i, "No topological information available.")
        text = texts[i][:500]  # Truncate long texts

        prompt = f"""Given a node from {graph_info[dataset_name]}.

The node description is: {text}

{topo}

Based on the node's textual description and topological properties, classify this node into one of these categories: {', '.join(labels)}.

Provide:
1. Your top-3 most likely classes with confidence scores
2. A brief explanation (2-3 sentences) of why this node belongs to each predicted class

Format your response as a concise paragraph describing the node and its likely classification."""
        prompts.append(prompt)
    return prompts


def build_ges_prompts(dataset_name, texts, topos):
    """Build GES-style prompts: rewrite description for GNN classifier."""
    labels = CORA_LABELS if dataset_name == 'cora' else PUBMED_LABELS
    graph_type = "citation network of academic papers" if dataset_name in ['cora', 'pubmed'] else "airport network"

    prompts = []
    for i in range(len(texts)):
        topo = topos.get(i, "")
        text = texts[i][:500]

        prompt = f"""You are rewriting node descriptions to improve performance of a graph-based classifier.
Each node is from a {graph_type}.
The possible classes are: {', '.join(labels)}.

### Target Node
Original Description:
{text}

Topological Summary:
{topo}

### Instructions
- Rewrite the target node's description so it is clearer, more informative, and discriminative.
- Highlight key themes, methodologies, or topics that help distinguish this node from others.
- Use ONLY the provided inputs (do not invent information).
- Keep style natural; write in less than 200 words.

**Final Output (one paragraph, no bullet points, less than 200 words):**"""
        prompts.append(prompt)
    return prompts


def build_tape_prompts(dataset_name, texts, topos):
    """Build TAPE-style prompts: predict class with explanation."""
    labels = CORA_LABELS if dataset_name == 'cora' else PUBMED_LABELS

    prompts = []
    for i in range(len(texts)):
        text = texts[i][:500]

        prompt = f"""Given the following paper information:

{text}

Classify this paper into one of the following categories: {', '.join(labels)}.

First, provide your prediction. Then, explain your reasoning in 2-3 sentences, focusing on the key concepts, methods, or topics mentioned in the paper that led to your classification."""
        prompts.append(prompt)
    return prompts


def build_kea_prompts(dataset_name, texts, topos):
    """Build KEA-style prompts: knowledge-enhanced text augmentation."""
    labels = CORA_LABELS if dataset_name == 'cora' else PUBMED_LABELS

    prompts = []
    for i in range(len(texts)):
        text = texts[i][:500]

        prompt = f"""Given the following paper:

{text}

Enhance this description by:
1. Identifying the key technical concepts, methods, and algorithms mentioned
2. Explaining what research area or subfield this paper belongs to
3. Describing what makes this paper's approach distinctive

The possible research areas are: {', '.join(labels)}.

Write a concise, enhanced description (100-150 words) that captures the paper's core contribution and technical approach."""
        prompts.append(prompt)
    return prompts


def save_descriptions(descriptions, dataset_name, method_name, out_dir='embeddings'):
    """Save descriptions as TSV file."""
    out_root = f'node_descriptions/{dataset_name}/qwen2.5-72b'
    os.makedirs(out_root, exist_ok=True)
    path = f'{out_root}/{method_name}.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['node_idx', 'answer'])
        for i, desc in enumerate(descriptions):
            writer.writerow([i, desc])
    print(f"Saved {len(descriptions)} descriptions to {path}")
    return path


def encode_and_save(texts, out_path, model):
    """Encode with MiniLM and save."""
    print(f"Encoding {len(texts)} texts → {out_path}")
    emb = model.encode(texts, show_progress_bar=True, batch_size=128,
                       convert_to_tensor=True).to(torch.float32).cpu()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(emb, out_path)
    print(f"Shape: {emb.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_base', default='http://localhost:8000/v1')
    parser.add_argument('--model', default='Qwen/Qwen2.5-72B-Instruct-AWQ')
    parser.add_argument('--datasets', nargs='+', default=['cora', 'pubmed'])
    parser.add_argument('--methods', nargs='+', default=['tans', 'ges', 'tape', 'kea'])
    args = parser.parse_args()

    os.chdir(str(REPO_ROOT))

    # Connect to vLLM server
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)

    # Test connection
    print(f"Connecting to {args.api_base}...")
    models = client.models.list()
    model_name = models.data[0].id
    print(f"Using model: {model_name}")

    # Load encoder
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    prompt_builders = {
        'tans': build_tans_prompts,
        'ges': build_ges_prompts,
        'tape': build_tape_prompts,
        'kea': build_kea_prompts,
    }

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        data, texts, topos = load_dataset(dataset_name)

        for method in args.methods:
            print(f"\n--- {method.upper()} descriptions ---")

            # Build prompts
            prompts = prompt_builders[method](dataset_name, texts, topos)
            print(f"Built {len(prompts)} prompts")

            # Generate descriptions
            descriptions = generate_batch(client, prompts, model_name)

            # Save descriptions
            save_descriptions(descriptions, dataset_name, method)

            # Encode with MiniLM
            prefix = "title_abstract_" if dataset_name in ['cora', 'pubmed'] else ""
            emb_name = f"{prefix}{method.upper()}_qwen72b_minilm.pt"
            encode_and_save(descriptions, f'embeddings/{dataset_name}/{emb_name}', encoder)

    print("\n=== All done! ===")


if __name__ == '__main__':
    main()
