from __future__ import annotations

import gc
from html.parser import HTMLParser
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from hybrid_agent import SYSTEM_PROMPT, ToolExecutor, build_tool_executor, extract_answer
from hybrid_agent.tools import PythonTool


class _DuckDuckGoResultsParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self.capture_item: dict[str, str] | None = None
        self.capture_key: str | None = None
        self.capture_tag: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            url = attributes.get("href") or ""
            parsed = urlparse(url)
            if parsed.path.startswith("/l/"):
                url = parse_qs(parsed.query).get("uddg", [url])[0]
            self.capture_item = {"title": "", "url": url, "snippet": ""}
            self.results.append(self.capture_item)
            self.capture_key = "title"
            self.capture_tag = tag
        elif "result__snippet" in classes and self.results:
            self.capture_item = self.results[-1]
            self.capture_key = "snippet"
            self.capture_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self.capture_tag:
            self.capture_item = None
            self.capture_key = None
            self.capture_tag = None

    def handle_data(self, data: str) -> None:
        if self.capture_item is not None and self.capture_key is not None:
            self.capture_item[self.capture_key] += data


class DuckDuckGoSearchTool:
    """No-key web search fallback; Tavily is preferred when configured."""

    name = "duckduckgo_search"
    trigger_tag = "search"

    def __init__(self, max_results: int = 5, result_length: int = 900, timeout: int = 15):
        self.max_results = max_results
        self.result_length = result_length
        self.timeout = timeout

    def execute(self, query: str, **kwargs) -> str:
        query = query.strip()
        if not query:
            return "Empty search query."

        url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                page = response.read().decode("utf-8", errors="replace")
        except Exception as error:
            return f"Search failed: {error}"

        parser = _DuckDuckGoResultsParser()
        parser.feed(page)
        formatted = []
        for index, result in enumerate(parser.results[: self.max_results], start=1):
            title = " ".join(result["title"].split())
            result_url = result["url"].strip()
            snippet = " ".join(result["snippet"].split())[: self.result_length]
            if not title and not snippet:
                continue
            lines = [f"Page {index}: {title or result_url}"]
            if result_url:
                lines.append(f"URL: {result_url}")
            if snippet:
                lines.append(snippet)
            formatted.append("\n".join(lines))
        return "\n\n".join(formatted) if formatted else "No search results found."


def _mlx_array(tensor: torch.Tensor) -> mx.array:
    return mx.array(tensor.detach().float().cpu().numpy()).astype(mx.bfloat16)


class MLXHybridLink:
    def __init__(self, adapter_path: Path, hidden_size: int):
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.pre_ln.weight, self.pre_ln.bias = _mlx_array(state["pre_ln.weight"]), _mlx_array(state["pre_ln.bias"])
        self.proj1.weight, self.proj1.bias = _mlx_array(state["proj1.weight"]), _mlx_array(state["proj1.bias"])
        self.proj2.weight, self.proj2.bias = _mlx_array(state["proj2.weight"]), _mlx_array(state["proj2.bias"])
        self.post_ln.weight, self.post_ln.bias = _mlx_array(state["post_ln.weight"]), _mlx_array(state["post_ln.bias"])
        del state

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = hidden.astype(mx.bfloat16)
        return self.post_ln(hidden + self.proj2(nn.gelu(self.proj1(self.pre_ln(hidden)))))


class MLXHybridAgent:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        adapter_path: str | Path | None = None,
        record_latent_readout: bool = True,
    ):
        self.model_dir = Path(model_dir)
        adapter_path = Path(adapter_path) if adapter_path else self.model_dir / "adapter.pt"
        if not adapter_path.is_file():
            raise FileNotFoundError(f"HybridLink adapter not found at {adapter_path}")
        self.model, self.tokenizer = load(str(self.model_dir))
        self.backbone = self.model.model
        hidden_size = int(self.backbone.args.hidden_size)
        self.hybridlink = MLXHybridLink(adapter_path, hidden_size)
        self.device = "mps"
        self.vocab_size = len(self.tokenizer)
        self.record_latent_readout = record_latent_readout
        self.readout_bank = self._make_readout_bank() if record_latent_readout else None
        self.tools = build_tool_executor(require_search=False)
        if "search" not in self.tools.tools:
            self.tools.register_tool(DuckDuckGoSearchTool())
        if "python" in self.tools.tools:
            self.tools.register_tool(PythonTool(
                python_path=sys.executable,
                timeout=60,
                memory_limit_gb=None,
            ))
        self.anchor_ids = {
            name: self._special_id(name)
            for name in ("<latent>", "</latent>", "<think>", "</think>")
        }
        eos = getattr(self.tokenizer, "eos_token_ids", None) or []
        self.eos_ids = {int(token) for token in eos}
        if not self.eos_ids:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self.eos_ids.add(int(eos_id))
        self._blocked_in_think = self._make_blocked_mask(
            (self.anchor_ids["</latent>"], self.anchor_ids["<think>"])
        )
        self._blocked_out_think = self._make_blocked_mask(
            (self.anchor_ids["</latent>"], self.anchor_ids["<latent>"], self.anchor_ids["</think>"])
        )

    def _special_id(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected {token} to be one special token, got {ids}")
        return int(ids[0])

    def _make_blocked_mask(self, blocked_ids: tuple[int, ...]) -> mx.array:
        mask = [False] * self.vocab_size
        for token_id in blocked_ids:
            if 0 <= token_id < self.vocab_size:
                mask[token_id] = True
        return mx.array(mask)

    def _make_readout_bank(self) -> mx.array:
        embedding = self.backbone.embed_tokens
        chunks = []
        for start in range(0, self.vocab_size, 2048):
            token_ids = mx.arange(start, min(start + 2048, self.vocab_size))
            vectors = embedding(token_ids).astype(mx.float32)
            norms = mx.sqrt(mx.sum(vectors * vectors, axis=-1, keepdims=True))
            chunks.append((vectors / mx.maximum(norms, 1e-12)).astype(mx.float16))
        bank = mx.concatenate(chunks, axis=0)
        mx.eval(bank)
        return bank

    def _nearest_token(self, vector: mx.array) -> int:
        value = vector.reshape(1, -1).astype(mx.float16)
        value = value / mx.maximum(mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True)), 1e-12)
        token_id = mx.argmax(value @ self.readout_bank.T, axis=-1)
        return int(token_id.item())

    def _forward_ids(self, token_ids: list[int], cache) -> mx.array:
        hidden = self.backbone(mx.array([token_ids]), cache=cache)
        logits = self.model.lm_head(hidden[:, -1:, :])[:, -1, :]
        mx.eval(hidden, logits)
        return hidden, logits

    def _forward_embeddings(self, embeddings: mx.array, cache) -> mx.array:
        hidden = self.backbone(None, cache=cache, input_embeddings=embeddings)
        logits = self.model.lm_head(hidden[:, -1:, :])[:, -1, :]
        mx.eval(hidden, logits)
        return hidden, logits

    def _sample(self, logits: mx.array, sampler, in_think: bool) -> int:
        blocked = self._blocked_in_think if in_think else self._blocked_out_think
        filtered = mx.where(blocked[None, :], mx.array(float("-inf")), logits)
        logprobs = filtered - mx.logsumexp(filtered, axis=-1, keepdims=True)
        return int(sampler(logprobs).item())

    def _render_output(self, output: str, spans: list[dict]) -> str:
        marker = "<latent></latent>"
        parts = output.split(marker)
        rendered = parts[0]
        for index, part in enumerate(parts[1:]):
            text = spans[index]["pseudo_text"] if index < len(spans) else ""
            rendered += f"<latent>{text}</latent>" + part
        return rendered

    def generate(self, question: str, params: dict | None = None, on_delta=None) -> dict:
        params = params or {}
        max_steps = int(params.get("max_tokens", params.get("max_response_tokens", 4096)))
        max_latent_steps = int(params.get("max_latent_steps", 200))
        min_latent_steps = int(params.get("min_latent_steps", 4))
        max_tool_calls = int(params.get("max_tool_calls", 5))
        temperature = float(params.get("temperature", 0.6))
        top_p = float(params.get("top_p", 0.95))
        top_k = int(params.get("top_k", 20))
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = [int(token) for token in prompt]
        prompt_tokens = len(prompt_ids)
        cache = make_prompt_cache(self.model)
        start = time.time()
        hidden, logits = self._forward_ids(prompt_ids, cache)

        generated_ids: list[int] = []
        output = ""
        decoded_len = 0
        tool_scan_position = 0
        in_think = False
        latent_mode = False
        latent_steps_in_span = 0
        latent_steps_total = 0
        generation_steps = 0
        spans: list[dict] = []
        open_span: list[int] = []
        turns = 1
        python_calls = 0
        search_calls = 0
        tool_requests = 0
        end_reason = "budget"

        while generation_steps < max_steps:
            if latent_mode:
                generation_steps += 1
                latent_steps_in_span += 1
                link_input = self.hybridlink(hidden[:, -1:, :])
                exit_id = int(mx.argmax(logits, axis=-1).item())
                should_exit = (
                    latent_steps_in_span >= max_latent_steps
                    or (latent_steps_in_span >= min_latent_steps and exit_id == self.anchor_ids["</latent>"])
                )
                if should_exit:
                    close_id = self.anchor_ids["</latent>"]
                    pseudo_text = "".join(self.tokenizer.decode([token]) for token in open_span)
                    generated_ids.append(close_id)
                    full_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    delta = full_decoded[decoded_len:]
                    output += delta
                    decoded_len = len(full_decoded)
                    hidden, logits = self._forward_ids([close_id], cache)
                    spans.append({
                        "steps": latent_steps_in_span,
                        "exit": "max_steps" if latent_steps_in_span >= max_latent_steps else "anchor",
                        "pseudo_text": pseudo_text,
                    })
                    if on_delta and delta:
                        on_delta(delta)
                    open_span = []
                    latent_mode = False
                    latent_steps_in_span = 0
                    continue

                if self.record_latent_readout:
                    readout_token = self._nearest_token(link_input[0, 0])
                    open_span.append(readout_token)
                    if on_delta:
                        on_delta(self.tokenizer.decode([readout_token]))
                hidden, logits = self._forward_embeddings(link_input, cache)
                latent_steps_total += 1
                continue

            generation_steps += 1
            token_id = self._sample(logits, sampler, in_think)
            generated_ids.append(token_id)
            full_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            delta = full_decoded[decoded_len:]
            output += delta
            decoded_len = len(full_decoded)
            if on_delta and delta:
                on_delta(delta)

            if token_id in self.eos_ids:
                end_reason = "eos"
                break

            if token_id == self.anchor_ids["<think>"]:
                in_think = True
            elif token_id == self.anchor_ids["</think>"]:
                in_think = False
            elif token_id == self.anchor_ids["<latent>"]:
                latent_mode = True
                latent_steps_in_span = 0

            hidden, logits = self._forward_ids([token_id], cache)
            if latent_mode:
                continue

            for tag in ("python", "search"):
                close_tag = f"</{tag}>"
                close_position = full_decoded.find(close_tag, tool_scan_position)
                if close_position < 0:
                    continue
                tool_scan_position = close_position + len(close_tag)
                tool_requests += 1
                if tool_requests > max_tool_calls:
                    end_reason = "tool_cap"
                    break
                content = ToolExecutor.extract_content(full_decoded[:tool_scan_position], tag)
                if tag == "python":
                    python_calls += 1
                else:
                    search_calls += 1
                try:
                    result = self.tools.execute(tag, content)
                except Exception as error:
                    result = f"The {tag} tool is unavailable: {error}"
                observation = f"<result>{result}</result>"
                output += observation
                if on_delta:
                    on_delta(observation)
                result_ids = [int(token) for token in self.tokenizer.encode(observation, add_special_tokens=False)]
                if result_ids:
                    hidden, logits = self._forward_ids(result_ids, cache)
                in_think = False
                turns += 1
                break
            if end_reason == "tool_cap":
                break

        if latent_mode:
            if open_span:
                spans.append({
                    "steps": len(open_span),
                    "exit": end_reason,
                    "pseudo_text": "".join(self.tokenizer.decode([token]) for token in open_span),
                })
            close_id = self.anchor_ids["</latent>"]
            generated_ids.append(close_id)
            full_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            delta = full_decoded[decoded_len:]
            output += delta
            if on_delta and delta:
                on_delta(delta)
        elif open_span:
            spans.append({
                "steps": len(open_span),
                "exit": end_reason,
                "pseudo_text": "".join(self.tokenizer.decode([token]) for token in open_span),
            })

        rendered = self._render_output(output, spans)
        return {
            "output": output,
            "output_rendered": rendered,
            "answer": extract_answer(output),
            "prompt_tokens": prompt_tokens,
            "end_reason": end_reason,
            "turns": turns,
            "generation_steps": generation_steps,
            "latent_steps": latent_steps_total,
            "latent_spans": len(spans),
            "latent_readouts": spans,
            "python_calls": python_calls,
            "search_calls": search_calls,
            "tool_requests": tool_requests,
            "wall_time_s": round(time.time() - start, 2),
        }

    def unload(self) -> None:
        self.readout_bank = None
        self.hybridlink = None
        self.model = None
        self.backbone = None
        self.tokenizer = None
        self.tools = None
        gc.collect()
        mx.clear_cache()
