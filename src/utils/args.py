import argparse
from src.utils.consts import LLM_AVAILABLE_OLLAMA, LLM_AVAILABLE_OPENAI


def get_args():
    parser = argparse.ArgumentParser("Evaluation")

    # General Config
    parser.add_argument("--pt_data", type=str, default="na")
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument(
        "--use_params", action="store_true", help="Whether to use the params"
    )
    parser.add_argument("--group", type=str, default="base")
    parser.add_argument("--debug", action="store_true", help="Debug Mode")
    parser.add_argument("--desc_csv", type=str, default="none")
    parser.add_argument(
        "--topo_csv",
        type=str,
        default="none",
        help="Path to the CSV file containing topological features",
    )

    parser.add_argument("--tgt_desc_csv", type=str, default="none")
    parser.add_argument(
        "--tgt_topo_csv",
        type=str, default="none",
        help="Path to the CSV file containing topological features for the target dataset",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory for results"
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Save the model after training"
    )

    # Emb Config
    parser.add_argument("--node_text", type=str, nargs="+", default="none")
    parser.add_argument("--emb_model", type=str, default="minilm")
    parser.add_argument(
        "--emb_method",
        type=str,
        default="TTA",
        choices=["TANS", "TANS_NEW", "TANS_PLUS", "TTA"],
    )
    parser.add_argument("--emb_file", type=str, default="none")

    # Model Config
    parser.add_argument(
        "--backbone", type=str, default="gcn", choices=["gcn", "gat", "mlp"]
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalize", type=str, default="none")

    # LLM Config
    parser.add_argument(
        "--llm", type=str, default="gemma3-1b", choices=list(LLM_AVAILABLE_OLLAMA.keys()) + list(LLM_AVAILABLE_OPENAI.keys())
    )
    parser.add_argument(
        "--system_prompt", type=str, default="You are a helpful assistant in graph learning and graph theory.", help="Optional system prompt for LLM"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for LLM sampling"
    )
    parser.add_argument(
        "--max_output_tokens", type=int, default=512, help="Max output tokens for LLM"
    )
    parser.add_argument(
        "--ray_concurrency",
        type=int,
        default=1,
        help="Ray concurrency for batch inference",
    )
    parser.add_argument(
        "--ray_batch_size",
        type=int,
        default=64,
        help="Ray batch size for batch inference",
    )

    # Prompt Buffer Config
    parser.add_argument(
        "--buffer_size", type=int, default=10, help="Size of the prompt buffer"
    )
    parser.add_argument(
        "--buffer_k",
        type=int,
        default=5,
        help="Number of top-k prompts to retrieve from the buffer",
    )

    # Test time adaptation config
    parser.add_argument(
        "--tta_max_iters",
        type=int,
        default=10,
        help="Max iterations for test time adaptation",
    )
    parser.add_argument(
        "--entropy_thr", type=float, default=0.5, help="Entropy threshold for TTA"
    )
    # Training Config
    parser.add_argument(
        "--label_setting", type=str, default="ratio", choices=["ratio", "number"]
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Ratio of training nodes in target dataset",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of training nodes in target dataset",
    )
    parser.add_argument(
        "--train_num", type=int, default=20, help="Number of training nodes per class"
    )
    parser.add_argument(
        "--val_num", type=int, default=30, help="Number of validation nodes per class"
    )

    parser.add_argument(
        "--pt_epochs", type=int, default=500, help="Epochs for pretraining"
    )
    parser.add_argument(
        "--pt_lr", type=float, default=1e-3, help="Learning rate for pretraining"
    )
    parser.add_argument(
        "--pt_decay", type=float, default=0, help="Weight decay for pretraining"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Epochs for fine-tuning"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for fine-tuning"
    )
    parser.add_argument("--early_stop", type=int, default=200)
    parser.add_argument(
        "--decay", type=float, default=1e-5, help="Weight decay for fine-tuning"
    )

    # Others
    parser.add_argument("--feat_noise", type=float, default=0.0)
    parser.add_argument("--edge_noise", type=float, default=0.0)
    parser.add_argument(
        "--eval_disc",
        action="store_true",
        help="Whether to evaluate the discrepancy between source and target",
    )
    parser.add_argument(
        "--wo_neigh", action="store_true", help="Without using neighbors"
    )

    # Structural (Node2Vec) Embedding Config
    parser.add_argument(
        "--struct_method", type=str, default="node2vec", choices=["node2vec", "deepwalk", "struct2vec"]
    )
    parser.add_argument(
        "--struct_dim", type=int, default=128, help="Dimension of structural embedding"
    )
    parser.add_argument(
        "--struct_walk_length", type=int, default=80, help="Length of each random walk"
    )
    parser.add_argument(
        "--struct_context_size",
        type=int,
        default=10,
        help="Context (window) size for Node2Vec",
    )
    parser.add_argument(
        "--struct_walks_per_node", type=int, default=10, help="Number of walks per node"
    )
    parser.add_argument(
        "--struct_neg_samples",
        type=int,
        default=1,
        help="Negative samples per positive",
    )
    parser.add_argument(
        "--struct_p", type=float, default=1.0, help="Return parameter p"
    )
    parser.add_argument(
        "--struct_q", type=float, default=1.0, help="In­out parameter q"
    )
    parser.add_argument(
        "--struct_batch_size",
        type=int,
        default=128,
        help="Batch size for Node2Vec sampler",
    )
    parser.add_argument(
        "--struct_lr",
        type=float,
        default=0.01,
        help="Learning rate for structural embedding training",
    )
    parser.add_argument(
        "--struct_epochs",
        type=int,
        default=1,
        help="Number of epochs to train structural embeddings",
    )

    parser.add_argument("--demos_method", type=str, default=None)

    parser.add_argument(
        "--buffer_alpha",
        type=float,
        default=0.5,
        help="Weight for combining text and structural embeddings when retrieving from buffer",
    )

    parser.add_argument(
        "--embed_alpha",
        type=float,
        default=0.5,
        help="Weight for combining new and old text embeddings during TTA",
    )

    parser.add_argument(
        "--sweep_mode",
        action="store_true",
        help="run in sweep mode: skip heavy logging/saving and return metrics",
    )

    parser.add_argument(
        "--do_tta",
        action="store_true",
        help="Whether to perform Test Time Adaptation (TTA)",
    )

    parser.add_argument(
        "--rewrite_all",
        action="store_true",
        help="If set, rewrite descriptions for the entire dataset after training using the JSON buffer."
    )

    parser.add_argument(
        "--tta_topk",
        type=int,
        default=20,
        help="Max number of high-entropy test nodes to adapt per TTA iteration (0 = no cap).",
    )

    parser.add_argument(
        "--desc_mode",
        type=str,
        default="orig_only",
        choices=["both", "llm_only", "orig_only"],
        help="Mode for using node descriptions: 'both' for LLM and original, 'llm_only' for LLM only, 'orig_only' for original only",
    )

    parser.add_argument("--retriever_hidden_dim", type=int, default=128)
    parser.add_argument("--retriever_num_layers", type=int, default=2)
    parser.add_argument("--retriever_lr", type=float, default=1e-3)
    parser.add_argument("--retriever_epochs", type=int, default=50)
    parser.add_argument("--retriever_batch_size", type=int, default=64)
    parser.add_argument("--retriever_batches_per_epoch", type=int, default=100)
    parser.add_argument("--retriever_temp", type=float, default=0.1)

    # --- Probabilistic retriever ---
    parser.add_argument("--retrieval_sampler", type=str, default="softmax",
                        choices=["topk", "softmax", "nucleus", "stoch_mmr"],
                        help="Sampling rule for picking buffer demos.")
    parser.add_argument("--retrieval_tau", type=float, default=0.07,
                        help="Temperature for softmax/nucleus/stoch_mmr sampling.")
    parser.add_argument("--retrieval_top_p", type=float, default=0.9,
                        help="Top-p (nucleus) cumulative mass for 'nucleus' sampler.")
    parser.add_argument("--retrieval_pool_size", type=int, default=256,
                        help="How many high-scoring candidates to form the sampling pool from.")
    parser.add_argument("--retrieval_beta", type=float, default=0.5,
                        help="Reliability weight mixing coefficient for (score + beta*reliability).")

    parser.add_argument("--rewrite_iters", type=int, default=2,
                        help="Number of global rewrite iterations.")
    parser.add_argument("--hist_max_steps", type=int, default=3,
                        help="How many recent history steps per exemplar to include in prompts.")
    parser.add_argument(
        "--rewrite_batch_size",
        type=int,
        default=64,
        help="Batch size for the global rewrite stage; results are saved after every batch to support resume."
    )

    parser.add_argument("--sim_method", type=str, default="weighted_sum", choices=[
        "text_only", "struct_only", "weighted_sum", "max", "product", "concat_cosine"
    ], help="Similarity method for retrieving from the prompt buffer.")

    parser.add_argument("--ex_block_template", type=str, default="default",
                    help="Name of example-block template from example_blocks.yaml")
    parser.add_argument("--ex_block_yaml", type=str, default=None,
                    help="Path to YAML file with example block templates & history (defaults to src/prompts/example_blocks.yaml)")

    args = parser.parse_args()
    return vars(args)
