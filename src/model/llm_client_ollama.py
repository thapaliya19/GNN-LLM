from typing import List

from ollama import generate as ollama_generate
from ollama._types import ResponseError

from src.utils.consts import LLM_AVAILABLE_OLLAMA as LLM_AVAILABLE, TTA_PROMPT_TEMPLATE
from tqdm.auto import tqdm


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
        demos: List[List[str]],
        topo_descriptions: List[str] = [],
        use_batch: bool = False,
        max_retries: int = 3,
    ) -> List[str]:
        """
        Few-shot rewriting via Ollama:
         - If use_batch: uses batch_generate on constructed prompts.
         - Otherwise uses single_generate in a loop.
        """

        prompts = []
        for idx, (text, topo_desc) in enumerate(zip(texts, topo_descriptions)):
            # Handle per-instance or shared demos
            cur_demos = demos[idx] if isinstance(demos[0], list) else demos  # type: ignore

            similar_descriptions = "\n\n".join(
                [f"Structurally similar node's properties: {demo}" for demo in cur_demos]
            )
            prompt = TTA_PROMPT_TEMPLATE.format(
                current_text=text, demos=similar_descriptions, current_node_description=topo_desc
            )
            prompts.append(prompt)
        
        print("Example prompt:\n", prompts[0])

        if use_batch:
            return self.batch_generate(prompts)
        # Fallback to individual tldr calls with progress bar
        # fallback: loop with retries
        results: List[str] = []
        for idx, (prompt, original_desc) in enumerate(zip(prompts, topo_descriptions)):
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
