# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import time
from typing import List, Dict, Any, Iterable, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, BadRequestError, APIStatusError
from tqdm.auto import tqdm

# If you already maintain a map of friendly names -> real models, you can import it.
# Otherwise, we just resolve to gpt-4o-mini directly.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Optional: tiny fixed-size LRU-ish cache to avoid paying twice for identical prompts.
@dataclass
class _CacheEntry:
    text: str

class _PromptCache:
    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._data: Dict[str, _CacheEntry] = {}
        self._order: List[str] = []

    def get(self, key: str) -> Optional[str]:
        item = self._data.get(key)
        if item is None:
            return None
        # move to end (recently used)
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        return item.text

    def set(self, key: str, value: str) -> None:
        if key in self._data:
            # update + move to end
            self._data[key].text = value
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return
        # evict if needed
        if len(self._data) >= self.max_size and self._order:
            oldest = self._order.pop(0)
            self._data.pop(oldest, None)
        self._data[key] = _CacheEntry(value)
        self._order.append(key)


def _strip_spaces(s: str) -> str:
    # match your Ollama client's normalizing behavior
    return " ".join(s.strip().split())


def _retryable_call(fn, *, max_retries=6, base_delay=0.8, jitter=0.2):
    """
    Simple exponential backoff around OpenAI calls.
    Retries: rate limits, transient 5xx, and connection errors.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except (RateLimitError, APIConnectionError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_for = base_delay * (2 ** (attempt - 1))
            # jitter
            sleep_for += min(jitter, 0.001 + (jitter * 0.5))
            time.sleep(sleep_for)
        except APIStatusError as e:
            # Retry 5xx; surface 4xx
            if 500 <= e.status_code < 600:
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_for = base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_for)
            else:
                raise
        except BadRequestError:
            # e.g., content filters / token limits -> don't retry blindly
            raise


class LLMClientOpenAI:
    """
    Drop-in replacement for your Ollama client, using OpenAI's Chat Completions API
    with gpt-4o-mini. Designed to minimize cost:

      • uses gpt-4o-mini by default
      • lets you cap output via max_output_tokens
      • optional small prompt cache to avoid duplicate calls
      • supports parallel workers for faster batches without changing API

    Methods:
      - single_generate(prompt: str) -> str
      - batch_generate(prompts: List[str]) -> List[str]
      - batch_rewrite(texts, demos, ...) -> List[str]   (mirrors your Ollama client)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_MODEL,
        *,
        # Generation knobs (tune down to save tokens/cost)
        temperature: float = 1.0,
        max_output_tokens: int = 160,           # keep short; bump only if needed
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,

        # Throughput / $ control
        parallel_workers: int = 4,              # 4–8 is usually safe
        cache_size: int = 10_000,               # small in-memory cache
        system_prompt: Optional[str] = None,    # set once to avoid repeating
    ):
        # Respect existing env var
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")

        self.client = OpenAI()
        self.model_name = model_name or DEFAULT_OPENAI_MODEL

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        self.parallel_workers = max(1, int(parallel_workers))
        self.cache = _PromptCache(max_size=cache_size)
        self.system_prompt = system_prompt  # can be None

        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    # ---------- low-level call ----------
    def _chat(self, prompt: str) -> str:
        # cache hit?
        cached = self.cache.get(prompt)
        if cached is not None:
            return cached

        def _do_call():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=(
                    [{"role": "system", "content": self.system_prompt}]
                    if self.system_prompt else []
                ) + [
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_output_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )

        resp = _retryable_call(_do_call)
        # Track token usage
        if hasattr(resp, "usage") and resp.usage is not None:
            self._total_input_tokens += resp.usage.prompt_tokens or 0
            self._total_output_tokens += resp.usage.completion_tokens or 0
        out = resp.choices[0].message.content or ""
        out = _strip_spaces(out)
        self.cache.set(prompt, out)
        return out

    # ---------- public API (mirrors your Ollama client) ----------
    def single_generate(self, prompt: str, stream: bool = False) -> str:
        # stream is ignored here to keep same signature
        return self._chat(prompt)

    def _batched(
        self, items: List[str], fn, *, workers: int, desc: str
    ) -> List[str]:
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(fn, items[i]): i for i in range(len(items))
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    # Fail fast; you could also return "" here if you prefer
                    raise RuntimeError(f"OpenAI call failed at index {idx}: {e}") from e
        return results  # type: ignore

    def batch_generate(
        self,
        prompts: List[str],
        *,
        parallel_workers: Optional[int] = None,
        tqdm_desc: str = "OpenAI batch"
    ) -> List[str]:
        workers = parallel_workers or self.parallel_workers
        return self._batched(prompts, self._chat, workers=workers, desc=tqdm_desc)

    def get_cost_summary(self) -> str:
        """Return a summary of API usage and estimated cost."""
        # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
        input_cost = self._total_input_tokens * 0.15 / 1_000_000
        output_cost = self._total_output_tokens * 0.60 / 1_000_000
        total_cost = input_cost + output_cost
        return (
            f"API Cost Summary:\n"
            f"  Input tokens:  {self._total_input_tokens:,}\n"
            f"  Output tokens: {self._total_output_tokens:,}\n"
            f"  Estimated cost: ${total_cost:.4f} "
            f"(input: ${input_cost:.4f}, output: ${output_cost:.4f})"
        )

    # ---------- parity with your Ollama few-shot rewrite ----------
    # This mirrors your input signature so you don’t have to change callers.
    def batch_rewrite(
        self,
        texts: List[str],
        demos,  # src.model.prompt_buffer.BufferItemReturned
        use_batch: bool = False,  # kept for parity; we always batch here
        max_retries: int = 3,     # unused (we globally retry)
        dataset: str = "cora"
    ) -> List[str]:
        # late import to avoid circulars if any
        from src.utils.consts import (
            TTA_PROMPT_TEMPLATE,
            CORA_ID2LABEL, PUBMED_ID2LABEL, AIRPORT_ID2LABEL
        )

        def label_name(ds: str, y_id: int) -> str:
            if ds == "cora":
                return CORA_ID2LABEL.get(int(y_id), f"unknown({y_id})")
            elif ds == "pubmed":
                return PUBMED_ID2LABEL.get(int(y_id), f"unknown({y_id})")
            else:
                return AIRPORT_ID2LABEL.get(int(y_id), f"unknown({y_id})")

        prompts: List[str] = []
        graph_type = "citation network" if dataset in ["cora", "pubmed"] else "airport network"

        for idx, text in enumerate(texts):
            cur_demos = demos[idx] if isinstance(demos[0], list) else demos  # type: ignore
            demo_lines = []
            for d in cur_demos:
                demo_lines.append(
                    f"- Description: {d.text_desc}\n\tLabel: {label_name(dataset, d.label)}"
                )
            similar_descriptions = "\n".join(demo_lines)

            prompt = TTA_PROMPT_TEMPLATE.format(
                current_node_description=text,
                demos=similar_descriptions,
                graph_type=graph_type
            )
            prompts.append(prompt)

        # Helpful for debugging:
        if prompts:
            print("Example prompt:\n", prompts[0][:1000], "\n---")

        return self.batch_generate(prompts, tqdm_desc="OpenAI rewrite batch")
