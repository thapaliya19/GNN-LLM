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

TTA_PROMPT_TEMPLATE = """
You are an expert in text-attributed graph networks and analyzing the structure of graphs.
You will be given some examples of LLM generated text descriptions that performed well under GNN, and a similar text description that did not perform well.
Your job is to analyze the current LLM generated node text descriptions, its topological information and the example LLM generated text descriptions, and provide a refined text description for the current node that is more suitable for GNN.
Below is the original descriptions of the current node containing topological information:
{current_node_description}
Below is the current LLM generated text description that did not perform well:
{current_text}
Below are the examples of LLM generated text descriptions of other nodes that performed well under GNN:
{demos}
Please provide a refined text description for the current node that is more suitable for GNN.
Your refined text description should be concise, clear, and relevant to the node's attributes, and under 200 words.
"""