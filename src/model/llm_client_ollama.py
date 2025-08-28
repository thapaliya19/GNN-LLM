from typing import List

from ollama import generate as ollama_generate
from ollama._types import ResponseError

from src.utils.consts import LLM_AVAILABLE_OLLAMA as LLM_AVAILABLE, TTA_PROMPT_TEMPLATE, CORA_ID2LABEL
from tqdm.auto import tqdm

from src.model.prompt_buffer import BufferItemReturned


class LLMClientOllama:
    """
    Wraps the Ollama Python client for single and batch inference.
    """

    def __init__(
        self,
        model_name: str = "gemma3-1b",
    ):
        # Resolve and validate model name
        model_name = LLM_AVAILABLE.get(model_name, model_name)
        if model_name not in LLM_AVAILABLE.values():
            raise ValueError(
                f"Model {model_name} is not available. "
                f"Available models: {list(LLM_AVAILABLE.values())}"
            )
        self.model_name = model_name

    def single_generate(self, prompt: str, stream: bool = False) -> str:
        """
        Generate a single completion using the Ollama Python client.

        Args:
            prompt: The prompt text to generate from.
            stream: Whether to return a streaming generator.
        Returns:
            Generated text (full content if not streaming, concatenated otherwise).
        """
        response = ollama_generate(self.model_name, prompt, stream=False)
        return response["response"].strip().replace("\n", " ").replace("  ", " ")

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Perform batch inference by sequentially calling single_generate.
        """
        return [self.single_generate(prompt) for prompt in prompts]

    def batch_rewrite(
        self,
        texts: List[str],
        demos: BufferItemReturned,
        use_batch: bool = False,
        max_retries: int = 3,
        dataset="cora"
    ) -> List[str]:
        """
        Few-shot rewriting via Ollama:
         - If use_batch: uses batch_generate on constructed prompts.
         - Otherwise uses single_generate in a loop.
        """
        def label_name(dataset: str, y_id: int) -> str:
            if dataset == "cora":
                return CORA_ID2LABEL.get(int(y_id), f"unknown({y_id})")
            return str(int(y_id))

        prompts = []
        graph_type = "citation network" if dataset in ["cora", "pubmed"] else "airport network"
        for idx, text in enumerate(texts):
            # Handle per-instance or shared demos
            cur_demos = demos[idx] if isinstance(demos[0], list) else demos  # type: ignore

            similar_descriptions = "\n".join([f"- Description: {d.text_desc}\n\tLabel: {label_name(dataset, d.label)}" for d in cur_demos])

            prompt = TTA_PROMPT_TEMPLATE.format(
                current_node_description=text, demos=similar_descriptions, graph_type=graph_type
            )
            prompts.append(prompt)
        
        print("Example prompt:\n", prompts[0])

        if use_batch:
            return self.batch_generate(prompts)
        # Fallback to individual tldr calls with progress bar
        # fallback: loop with retries
        results: List[str] = []
        for idx, (prompt, original_desc) in enumerate(zip(prompts, texts)):
            for attempt in range(1, max_retries + 1):
                try:
                    gen = self.single_generate(prompt)
                    results.append(gen)
                    break
                except ResponseError as e:
                    if attempt < max_retries:
                        print(
                            f"[{idx}] Ollama error (attempt {attempt}/{max_retries}): {e}. retrying..."
                        )
                        continue
                    else:
                        print(
                            f"[{idx}] Ollama failed after {max_retries} attempts, "
                            "falling back to original description."
                        )
                        results.append(original_desc)
        return results
