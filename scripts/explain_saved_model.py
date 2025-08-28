import os
import os.path as osp
import random

import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph
import yaml

from src.data.data import get_data
from src.model.model import GNN
from src.utils.args import get_args
from src.utils.utils import get_device, seed_everything
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
seed_everything(1)
# 1) Re‐parse your arguments (so you have params['data'], etc.)
params = get_args()
params = get_args()
params["data_path"] = osp.join(os.path.dirname(__file__), "datasets")

dataset = params["data"]
emb_method = params["emb_method"]
node_text = "_".join(params["node_text"])

has_node_text = dataset in ["cora", "pubmed"]

if params["use_params"]:
    with open(f"src/config/finetune.yaml", "r") as f:
        default_params = yaml.safe_load(f)
        if has_node_text:
            params.update(default_params[emb_method][node_text][dataset])
        else:
            params.update(default_params[emb_method][dataset])

print(params["seed"])

if has_node_text:
    params.update({"node_text": node_text})
else:
    params.update({"node_text": "none"})



device = get_device(params)

# 2) Instantiate the model architecture exactly as before
graph = get_data(
    params, params["data"]
)  # loads x, edge_index, y, text_descriptions, ...
graph.x = graph.x.to(device)
graph.edge_index = graph.edge_index.to(device)
graph.y = graph.y.to(device)

model = GNN(
    input_dim=graph.num_dim,
    hidden_dim=params["hidden_dim"],
    output_dim=graph.num_classes,
    activation=torch.nn.PReLU,
    num_layers=params["num_layers"],
    backbone=params["backbone"],
    normalize=params["normalize"],
    dropout=params["dropout"],
)
model = model.to(device)

seed_everything(params["seed"][0])

# 3) Load the saved state dict
checkpoint = torch.load(
    f"output/{params['data']}/{params['llm']}/model_seed_{params['seed'][0]}.pth",
    map_location=device,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Loaded model weights.")

# 1) Run the model once to get its predicted labels
with torch.no_grad():
    out = model(graph.x, graph.edge_index)                    # shape [N, num_classes]
    pred_labels = out.argmax(dim=1).cpu()                     # tensor [N], each entry is the model’s predicted class



# -------------------------------------------------------------------
# 3. Set up GNNExplainer
# -------------------------------------------------------------------
explainer = Explainer(
    model               = model,
    algorithm           = GNNExplainer(epochs=200),   # train mask for 200 steps
    explanation_type    = 'model',                            # explain the model’s prediction
    node_mask_type      = 'attributes',                       # learn a mask over node features
    edge_mask_type      = 'object',                           # learn a mask over edges
    model_config        = dict(
        mode        = 'multiclass_classification',
        task_level  = 'node',
        return_type = 'log_probs',                            # match your model’s output
    ),
) 

def manual_visualize(explanation, node_idx, num_hops=2, top_k=10, save_dir=None):
    # 4.1) Extract the k‑hop subgraph around node_idx
    subset, edge_index_sub, mapping, hard_edge_mask = k_hop_subgraph(
        node_idx, num_hops, graph.edge_index, relabel_nodes=True
    )

    # 4.2) Edge importance for that subgraph
    edge_mask_sub = explanation.edge_mask[hard_edge_mask].cpu().numpy()

    # 4.3) Feature importances for just this node
    feat_mask_full = explanation.node_mask[node_idx].cpu().numpy()

    # Original node IDs in this subgraph
    orig_nodes = subset.tolist()

    # Build NetworkX graph using local indices
    G = nx.Graph()
    src, dst = edge_index_sub
    for i, (u_local, v_local) in enumerate(zip(src.tolist(), dst.tolist())):
        G.add_edge(u_local, v_local, weight=edge_mask_sub[i])

    # 4.4) Prepare node colors by true class label
    labels = [int(pred_labels[n].item()) for n in orig_nodes]
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap('tab10')
    class_to_color = {
        lab: cmap(i / max(len(unique_labels)-1, 1))
        for i, lab in enumerate(unique_labels)
    }
    node_colors = [ class_to_color[lab] for lab in labels ]

    # Local index of the target node
    central_local = mapping.item()
    # central_class = int(graph.y[node_idx].item())

    true_class = int(graph.y[node_idx].item())
    pred_class = int(pred_labels[node_idx].item())

    # 4.5) Plotting subgraph
    plt.figure(figsize=(10,10))
    k = 2.0 / np.sqrt(len(G.nodes()))
    pos = nx.spring_layout(
        G,
        k=k,                   # increases repulsion between all nodes
        iterations=100,        # more time for the layout to “settle”
        scale=2.0              # stretch the final coordinates
    )

    # Draw edges with importance weights
    weights = [d['weight'] for _,_,d in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos,
        width=[3*w for w in weights],
        edge_color=weights,
        edge_cmap=plt.cm.Blues
    )

    # Draw other nodes colored by label
    other_nodes = [n for n in G.nodes() if n != central_local]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=other_nodes,
        node_color=[node_colors[n] for n in other_nodes],
        edgecolors='black',
        node_size=300
    )

    # Draw central node highlighted and colored by its class
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[central_local],
        node_color=class_to_color[pred_class],
        edgecolors='red',
        linewidths=2,
        node_size=600
    )

    # Node labels: show real node_idx
    label_mapping = {local: orig for local, orig in enumerate(orig_nodes)}
    import textwrap
    wrapped_labels = {
        n: "\n".join(textwrap.wrap(str(label_mapping[n]), width=5))
        for n in G.nodes()
    }

    nx.draw_networkx_labels(
        G, pos,
        labels=wrapped_labels,     # or just label_mapping if short
        font_size=8,               # small enough to fit
        font_family='sans-serif',
        font_weight='normal',
        # bbox=dict(
        #     boxstyle='circle,pad=0.1',
        #     fc='white', ec='black', alpha=0.0
        # ),
        horizontalalignment='center',
        verticalalignment='center'
    )

    # Legend for class colors
    patches = [mpatches.Patch(color=col, label=f'Class {lab}')
               for lab, col in class_to_color.items()]
    plt.legend(handles=patches, title='Classes', loc='upper right')

    # Title with central node's class
    plt.title(f"Node {node_idx}: true: {true_class}, pred={pred_class}, subgraph (k={num_hops})")
    plt.axis('off')
    plt.show()
    
    if save_dir:
        save_path = osp.join(save_dir, f"explanation_node_{node_idx}.png")  
        plt.savefig(save_path, dpi=300)
        print(f"Saved explanation for node {node_idx} to {save_path}")

    # ——— PLOT TOP‑k FEATURE IMPORTANCES ———
    # Find the indices of the top‐k dimensions
    
    if save_dir:
        save_path = osp.join(save_dir, f"explanation_node_{node_idx}_topk.png")
        plt.savefig(save_path)
        explanation.visualize_feature_importance(save_path, top_k=top_k)
        print(f"Saved explanation for node {node_idx} to {save_path}")

# mis_idx = [28, 37, 64, 68, 107] # true label and pred label are different for these nodes
mis_idx = [74, 126, 253] # true label and pred label are same
# -------------------------------------------------------------------
# 4. Visualize explanations for a few misclassified nodes
# -------------------------------------------------------------------
# Create output directories for saving explanations
explanation_save_dir = f"output/{params['data']}/{params['llm']}/explanation"
os.makedirs(explanation_save_dir, exist_ok=True)

# 7) Run & visualize for first few misclassifications
for nid in mis_idx:
    print(f"\n➤ Explaining node {nid}")
    explanation = explainer(
        x          = graph.x,
        edge_index = graph.edge_index,
        index      = nid,
    )
    # save_path = osp.join(explanation_save_dir, f"explanation_node_{nid}_topk.png")
    # plt.savefig(save_path)
    # explanation.visualize_feature_importance(save_path, top_k=10)
    # print(f"Saved feature importance explanation for node {nid} to {save_path}")
    manual_visualize(explanation, nid, num_hops=1, save_dir=explanation_save_dir)

    # save_path = osp.join(explanation_save_dir, f"explanation_node_{nid}.pdf")  
    # plt.savefig(save_path, dpi=300)
    # explanation.visualize_graph(save_path)
    # print(f"Saved explanation subgraph for node {nid} to {save_path}")