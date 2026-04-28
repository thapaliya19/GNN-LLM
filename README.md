# GES: GNN-guided Exemplar Selection for LLM-based Node Descriptions

Code for the COLM submission *Semantic Refinement with LLMs for Graph Representations*.

GES is a single-pass pipeline for node classification on text-attributed graphs (TAGs). Given an initial GNN trained on raw descriptions, GES builds a model-conditioned memory of training nodes, retrieves structurally and semantically similar exemplars for each target node, and asks an LLM to rewrite the target's description conditioned on those exemplars. The rewritten descriptions are encoded with a fixed text encoder (MiniLM-L12-v2, 384d) and fed back into the GNN.

## Repository layout

```
src/                 # package: data, model, prompts, training, retrieval, utils
scripts/             # entry points (pipeline, GNN eval, transfer, DA, generation)
embeddings/          # pre-computed .pt files used by the paper's main table
node_descriptions/   # generated GES descriptions (CSV), organized by dataset/model
```

## Installation

Python 3.10. Install dependencies into a fresh environment:

```bash
conda create -n ges python=3.10 -y
conda activate ges

# PyTorch + PyG (match your CUDA version; the paper used CUDA 11.8)
pip install torch==2.1.0
pip install torch_geometric torchmetrics

# Text encoders and utilities
pip install sentence-transformers pyyaml tqdm pandas numpy scikit-learn networkx

# LLM clients
pip install openai                       # for gpt-4o-mini (main-table results)
pip install vllm                         # optional: local Qwen2.5-72B inference
```

Set `OPENAI_API_KEY` in your shell if you plan to regenerate descriptions with gpt-4o-mini. For local inference, start a vLLM server with `scripts/start_vllm_server.sh` and point `--api_base` at it.

## Data

Raw graphs are **not** shipped. Download them from the TANS release and place them under `datasets/` in the repository root:

- TANS GitHub: <https://github.com/Zehong-Wang/TANS>
- TANS README links four Google Drive bundles to download: `dataset`, `property`, `response`, `text_emb`. Extract each into `datasets/` so the tree looks like:

```
datasets/
├── dataset/           # raw graphs and node texts (Cora, PubMed, Airports: USA/Europe/Brazil)
├── property/          # topological features (degree, betweenness, one-hop neighbors)
├── response/          # TANS-distributed questions and answers
└── text_emb/          # TANS pre-computed embeddings (used as the TANS baseline in the main table)
```

This release uses Cora, PubMed, USA-airport, Europe-airport, and Brazil-airport. TANS bundles additional TAGs (CiteSeer, WikiCS, etc.) that are also supported by the code but not reported in the paper.

The GES embeddings used by the paper's main table are already in `embeddings/`. The TANS baseline embeddings are **not** shipped with this repo — they come from the `text_emb` bundle in the TANS release. After downloading TANS, the relevant baseline files are:

- `datasets/text_emb/{cora,pubmed}/title_abstract_TANS_minilm.pt`
- `datasets/text_emb/{usa,europe,brazil}/none_TANS_gemma3-12b_minilm.pt` (or `TANS_minilm.pt` depending on bundle version)

## Reproducing results

All scripts run from the repository root. Python scripts prepend the repo root to `sys.path` via a shim, so imports resolve without `pip install -e .`.

### 1. Main table (Table 2 / 3): GNN evaluation on pre-computed embeddings

`scripts/train_gnn.py` loads a `.pt` embedding file and trains a GNN over multiple seeds, reporting test accuracy as mean ± std.

Cora GCN (ratio setting, 30 seeds), one run per description method:

```bash
# GES (ours) variants
for method in GES GES_gpt4omini; do
  python scripts/train_gnn.py --data cora --backbone gcn \
    --emb_file embeddings/cora/${method}_minilm.pt \
    --hidden_dim 128 --num_layers 2 --dropout 0.8 \
    --lr 0.001 --decay 0.0001 --label_setting ratio \
    --seed 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
           21 22 23 24 25 26 27 28 29 30 --debug
done

# Baselines (KEA, TAPE)
for method in KEA TAPE_E; do
  python scripts/train_gnn.py --data cora --backbone gcn \
    --emb_file embeddings/cora/${method}_minilm.pt \
    --hidden_dim 128 --num_layers 2 --dropout 0.8 \
    --lr 0.001 --decay 0.0001 --label_setting ratio \
    --seed 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
           21 22 23 24 25 26 27 28 29 30 --debug
done

# TANS baseline (after downloading the TANS text_emb bundle)
python scripts/train_gnn.py --data cora --backbone gcn \
  --emb_file datasets/text_emb/cora/title_abstract_TANS_minilm.pt \
  --hidden_dim 128 --num_layers 2 --dropout 0.8 \
  --lr 0.001 --decay 0.0001 --label_setting ratio \
  --seed 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
         21 22 23 24 25 26 27 28 29 30 --debug
```

PubMed GCN (use `embeddings/pubmed/{GES,KEA,TAPE_E}_minilm.pt`; PubMed has only one GES variant):

```bash
python scripts/train_gnn.py --data pubmed --backbone gcn \
  --emb_file embeddings/pubmed/GES_minilm.pt \
  --hidden_dim 256 --num_layers 2 --dropout 0.1 \
  --lr 0.005 --decay 0.0005 --label_setting ratio \
  --seed 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
         21 22 23 24 25 26 27 28 29 30 --debug
```

Airports (USA / Europe / Brazil), pick the GES variant from `embeddings/<ds>/{GES,GES_gpt4omini,GES_old}_minilm.pt`:

```bash
python scripts/train_gnn.py --data usa --backbone gcn \
  --emb_file embeddings/usa/GES_minilm.pt \
  --hidden_dim 128 --num_layers 3 --dropout 0.1 \
  --lr 0.05 --decay 0.0001 --label_setting ratio \
  --seed 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
         21 22 23 24 25 26 27 28 29 30 --debug
```

Swap `--label_setting ratio` → `--label_setting number --train_num 20 --val_num 30` (citation) or `--train_num 10 --val_num 20` (Brazil) for the low-label table. Swap `--backbone gcn` → `gat` or `mlp` for the other backbones.

### 2. Transfer (Table 4): pretrain → finetune across datasets

```bash
python scripts/transfer_gnn.py \
  --pt_data cora --data pubmed \
  --pt_emb_file embeddings/cora/GES_minilm.pt \
  --emb_file embeddings/pubmed/GES_minilm.pt \
  --backbone gcn --hidden_dim 128 --num_layers 2 --dropout 0.5 \
  --pt_epochs 500 --pt_lr 0.001 \
  --epochs 1000 --lr 0.001 --decay 0.0001 \
  --label_setting number --seed 1 2 3 4 5 6 7 8 9 10 --debug
```

### 3. Domain adaptation (Table 4): train source, evaluate target

```bash
python scripts/da_gnn.py \
  --pt_data usa --data brazil \
  --pt_emb_file embeddings/usa/GES_minilm.pt \
  --emb_file embeddings/brazil/GES_minilm.pt \
  --backbone gcn --hidden_dim 32 --num_layers 3 --dropout 0.5 \
  --epochs 1000 --lr 0.005 --decay 0.00005 \
  --label_setting number --seed 1 2 3 4 5 6 7 8 9 10 --debug
```

Per-direction hyperparameters are in `src/config/da.yaml` and `src/config/transfer.yaml`.

### 4. Regenerating node descriptions (optional)

GES descriptions used for the main table are already in `node_descriptions/<dataset>/gpt-4o-mini/` (see next section). To regenerate them with the full GES pipeline:

```bash
python scripts/ges_pipeline.py \
  --data cora --backbone gcn --llm gpt-4o-mini \
  --rewrite_all \
  --buffer_k 5 --buffer_alpha 0.5 --retrieval_tau 0.07 \
  --sim_method weighted_sum \
  --seed 1 --debug
```

For Qwen2.5-72B via a local vLLM server:

```bash
sbatch scripts/run_vllm_generation.sh
```

After regeneration, encode the new CSVs with MiniLM and point `train_gnn.py --emb_file` at the resulting `.pt` file.

## Generated node descriptions

`node_descriptions/` contains the GES descriptions generated by `scripts/ges_pipeline.py` with `gpt-4o-mini` that were used to produce the GES `.pt` files in `embeddings/`.

```
node_descriptions/
├── cora/gpt-4o-mini/
│   ├── GES.csv             # source of embeddings/cora/GES_minilm.pt          (seed=1)
│   └── GES_gpt4omini.csv   # source of embeddings/cora/GES_gpt4omini_minilm.pt (seed=1)
├── pubmed/gpt-4o-mini/
│   └── GES.csv             # source of embeddings/pubmed/GES_minilm.pt        (seed=2)
├── usa/gpt-4o-mini/
│   ├── GES.csv             # source of embeddings/usa/GES_minilm.pt           (seed=3)
│   └── GES_old.csv         # source of embeddings/usa/GES_old_minilm.pt       (seed=3)
├── europe/gpt-4o-mini/
│   ├── GES.csv             # seed=4
│   └── GES_old.csv         # seed=4
└── brazil/gpt-4o-mini/
    ├── GES.csv             # seed=5
    └── GES_old.csv         # seed=5
```

Each CSV is a tab-separated file with columns `node_idx\tanswer`. The `seed` in parentheses refers to the seed used during description *generation* (not the downstream 30-seed GNN evaluation). Rows are one per node, aligned with the node ordering in the corresponding TANS graph file.
