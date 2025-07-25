# from typing import List, Tuple


# from vllm import LLM, SamplingParams
# from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
# import ray

# from src.utils.consts import LLM_AVAILABLE_VLLM as LLM_AVAILABLE, TTA_PROMPT_TEMPLATE


# class LLMClientVllm:
#     """
#     Wraps vLLM for single and Ray Data-based batch inference.
#     """
#     def __init__(
#         self,
#         model_name: str = "gemma3-1b",
#         max_output_tokens: int = 128,
#         temperature: float = 0.0,
#         ray_concurrency: int = 1,
#         ray_batch_size: int = 64,
#     ):
#         model_name = LLM_AVAILABLE.get(model_name, model_name)
#         if model_name not in LLM_AVAILABLE.values():
#             raise ValueError(f"Model {model_name} is not available. "
#                              f"Available models: {list(LLM_AVAILABLE.values())}")
#         # vLLM single-request client
#         self.llm = LLM(model=model_name, trust_remote_code=True)
#         self.sampling_params = SamplingParams(
#             temperature=temperature,
#             max_tokens=max_output_tokens,
#         )
#         # Ray Data config for batch
#         self.ray_config = vLLMEngineProcessorConfig(
#             model_source=model_name,
#             engine_kwargs={
#                 "enable_chunked_prefill": True,
#                 "max_num_batched_tokens": 4096,
#                 "max_model_len": 16384,
#             },
#             concurrency=ray_concurrency,
#             batch_size=ray_batch_size,
#         )

#     def single_generate(self, prompt: str) -> str:
#         """Generate a single completion with vLLM."""
#         outputs = self.llm.generate(prompt=prompt, sampling_params=self.sampling_params)
#         for result in self.llm.generate(outputs):
#             return result.outputs[0].text.strip()
#         return ""

#     def batch_generate_ray(self, prompts: List[str]) -> List[str]:
#         """
#         Use Ray Data + vLLM for data-parallel batch inference.
#         """
#         # Create a Ray Dataset of prompts
#         ds = ray.data.from_items([{"prompt": p} for p in prompts])

#         # Build the processor
#         processor = build_llm_processor(
#             self.ray_config,
#             preprocess=lambda row: dict(
#                 prompt=row["prompt"],
#                 sampling_params=dict(
#                     temperature=self.sampling_params.temperature,
#                     max_tokens=self.sampling_params.max_tokens,
#                 ),
#             ),
#             postprocess=lambda row: {"generated_text": row["generated_text"]},
#         )

#         # Run inference
#         result_ds = processor(ds)

#         # Collect results
#         # Assumes order is preserved
#         results = [row["generated_text"] for row in result_ds.take_all()]
#         return results

#     def batch_rewrite(
#         self,
#         texts: List[str],
#         demos: List[List[str]],
#         use_ray: bool = False,
#         batch_size: int = 8,
#     ) -> List[str]:
#         """
#         Few-shot rewriting:
#          - If use_ray: use batch_generate_ray on constructed prompts.
#          - Otherwise use single_generate in a loop with sampling_params.
#         """
#         # Construct few-shot prompts
#         prompts = []
#         rewrite_prompt = TTA_PROMPT_TEMPLATE
#         for idx, text in enumerate(texts):
#             # replace {current_text} with the actual text
#             prompt = rewrite_prompt.format(current_text=text)
#             cur_demos = demos[idx] if isinstance(demos[0], list) else demos

#             similar_descriptions = "\n\n".join(
#                 [f"Similar text description: {demo}" for demo in cur_demos]
#             )
#             prompt = prompt.format(demos, similar_descriptions)
#             print(prompt)
#             prompts.append(prompt)

#         if use_ray:
#             return self.batch_generate_ray(prompts)
#         else:
#             # non-ray batched via vLLM.generate()
#             results = [
#                 self.single_generate(p)
#                 for p in prompts
#             ]
#             return results