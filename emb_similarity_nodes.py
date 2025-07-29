"""
This script loads a CSV file containing history of node descriptions,
computes cosine similarity between old and new descriptions using a
sentence-transformers model, and prints similarity scores for a subset
of selected nodes. It also adds a correctness indicator for predictions.
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("/home/ldn24004/autoprompt/output/europe/gemma3-12b/tta_history_seed_3.csv", sep="\t")
dataset = "europe"

# Load sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute cosine similarity between old and new descriptions
def compute_similarity(row):
    emb_old = model.encode(row['old_desc'], convert_to_tensor=True)
    emb_new = model.encode(row['new_desc'], convert_to_tensor=True)
    return util.cos_sim(emb_old, emb_new).item()

df["similarity"] = df.apply(compute_similarity, axis=1)

# Add correctness indicator
df["correct"] = df["true_label"] == df["pred_label"]

# Define the subset of node indices
selected_nodes = [74, 126, 68, 121]  # change as needed

# Filter for selected nodes
df_selected = df[df["node_idx"].isin(selected_nodes)]

# Print similarity scores per node per iteration
for node in selected_nodes:
    print(f"\n🔹 Similarity scores between old and new description for node {node}:")
    node_df = df_selected[df_selected["node_idx"] == node][["iteration", "similarity"]].sort_values("iteration")
    print(node_df.to_string(index=False))
