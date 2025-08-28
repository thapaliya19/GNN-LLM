LLM_AVAILABLE_OLLAMA = {
    'gemma3-1b': 'gemma3:1b',
    'gemma3-12b': 'gemma3:12b',
}

LLM_AVAILABLE_VLLM = {
    'gemma3-1b': 'google/gemma-3-1b-it',
    'gemma3-12b': 'google/gemma-3-12b-it',
}

SBERT_MODELS = {
    'albert': 'paraphrase-albert-small-v2',
    'roberta': 'all-distilroberta-v1',
    'minilm': 'all-MiniLM-L12-v2',
    'mpnet': 'all-mpnet-base-v2'
}

DESCRIPTION_TEMPLATE = 'The node type is {}. The node description is {}. The additional node description is {}.'

NODE_TYPES = {
        'cora': 'paper',
        'pubmed': 'paper',
        'usa': 'airport',
        'europe': 'airport',
        'brazil': 'airport'
    }


# ---- Cora label mapping (id -> name) ----
CORA_ID2LABEL = {
    0: "Case Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory",
}


# TTA_PROMPT_TEMPLATE = """
# You are an expert in text-attributed graph networks and analyzing the structure of graphs.
# You will be given some examples of LLM generated text descriptions that performed well under GNN, and a similar text description that did not perform well.
# Your job is to analyze the current LLM generated node text descriptions, its topological information and the example LLM generated text descriptions, and provide a refined text description for the current node that is more suitable for GNN.
# Follow this EXACT OUTPUT FORMAT (no preamble, no extra lines):
# Here's a breakdown of potential classes for the node and reasoning:
# **1. <TopClassName>:** <one concise rationale grounded in the node's topology and terminology used in EXAMPLES; 1–2 sentences>.
# **2. <SecondClassName>:** <short rationale; 1 sentence>.
# **3. <ThirdClassName>:** <short rationale; 1 sentence>.
# The other classes are less directly relevant. While the concepts could be related, they lack the strong connections demonstrated by the three selected classes.

# Original node description:\n
# {current_node_description}
# \n\nTopological summary of the node (degree/centrality/neighborhood gist):\n
# {current_text}
# \n\nExemplar descriptions from structurally/textually similar nodes:\n
# {demos}
# \n\nTask:\n
# - Rewrite the node's description so it is clear, specific, and discriminative among the classes.\n
# - Keep the style similar to the exemplars (no JSON; plain text only).\n
# - (Optional) You may hint at the top 1–3 likely classes with a very brief rationale.
# """

TTA_PROMPT_TEMPLATE = """
You are rewriting node descriptions to improve performance of a graph-based classifier.  
Each node may belong to {graph_type}.
### Task
Rewrite the description of the target node so that:
- It becomes **clearer, more informative, and discriminative** compared to the original.
- Use the provided original description if available; otherwise, synthesize a meaningful description from the topology and neighbor summaries.
- Use exemplar nodes as **guidance** but do **not** assume or guess the label of the target node.
- Highlight core themes, roles, or relationships relevant to the node.
- Keep the rewritten description **factual** and based only on given inputs.
- Limit the final rewritten description to **≤200 words**.

---

### Target Node
**Original Description:**
{current_node_description}

---

### Retrieved Similar Nodes (for reference only):
{demos}

---

### Instructions
- Use similar exemplar styles to make the text natural and informative, but do not copy them verbatim.
- If original text exists, **refine and clarify it** using exemplar patterns.
- If no text exists, **generate a concise textual summary** using structural cues and neighbor patterns.
- Do **not** disclose or speculate on the node’s label.

**Final Output:**  
A single rewritten description for the target node in less than 200 words.

"""