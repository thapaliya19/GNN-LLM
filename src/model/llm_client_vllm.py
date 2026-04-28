"""
vLLM-based LLM client for in-process inference (no external server needed).
Loads the model directly on available GPUs using vLLM.
"""
from typing import List
from vllm import LLM, SamplingParams
from src.utils.consts import LLM_AVAILABLE_VLLM as LLM_AVAILABLE


class LLMClientVllm:
    def __init__(
        self,
        model_name: str = "qwen2.5-72b",
        max_output_tokens: int = 300,
        temperature: float = 0.0,
        tensor_parallel_size: int = 2,
    ):
        hf_name = LLM_AVAILABLE.get(model_name, model_name)
        print(f"[vLLM] Loading {hf_name} with tensor_parallel_size={tensor_parallel_size}...")
        self.llm = LLM(
            model=hf_name,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            quantization="awq" if "AWQ" in hf_name else None,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        print(f"[vLLM] Model loaded successfully")

    def single_generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def batch_generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
