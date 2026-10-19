"""Experimental one-step Qwen3.8 + HybridLink interface probe.

This checks that MLX-VLM's Qwen3.8 language model can accept a continuous
embedding from a HybridLink-shaped adapter. It deliberately does not claim
that the 14B adapter is trained for Qwen3.8 or that the resulting text quality
is meaningful; a Qwen3.8 adapter needs HybridSFT/HybridRL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_vlm import load


def to_mlx(tensor: torch.Tensor) -> mx.array:
    return mx.array(tensor.detach().float().cpu().numpy()).astype(mx.bfloat16)


class HybridLinkProbe:
    def __init__(self, path: Path, hidden_size: int):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.pre_ln.weight, self.pre_ln.bias = to_mlx(state["pre_ln.weight"]), to_mlx(state["pre_ln.bias"])
        self.proj1.weight, self.proj1.bias = to_mlx(state["proj1.weight"]), to_mlx(state["proj1.bias"])
        self.proj2.weight, self.proj2.bias = to_mlx(state["proj2.weight"]), to_mlx(state["proj2.bias"])
        self.post_ln.weight, self.post_ln.bias = to_mlx(state["post_ln.weight"]), to_mlx(state["post_ln.bias"])

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = hidden.astype(mx.bfloat16)
        return self.post_ln(hidden + self.proj2(nn.gelu(self.proj1(self.pre_ln(hidden)))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    parser.add_argument("--adapter", type=Path, required=True, help="HybridLink adapter.pt")
    parser.add_argument("--prompt", default="What is 2 plus 2?")
    args = parser.parse_args()

    model, processor = load(args.model)
    language_model = model.language_model
    tokenizer = processor.tokenizer
    prompt_ids = mx.array([tokenizer.encode(args.prompt, add_special_tokens=False)])
    cache = language_model.make_cache()
    hidden = language_model.model(prompt_ids, cache=cache)
    link = HybridLinkProbe(args.adapter, hidden.shape[-1])
    latent = link(hidden[:, -1:, :])
    after_latent = language_model.model(None, inputs_embeds=latent, cache=cache)
    logits = language_model.lm_head(after_latent[:, -1:, :])
    mx.eval(after_latent, logits)
    next_id = int(mx.argmax(logits[0, -1]).item())

    print("checkpoint:", args.model)
    print("hidden:", tuple(hidden.shape), hidden.dtype)
    print("latent input:", tuple(latent.shape), latent.dtype)
    print("post-latent hidden:", tuple(after_latent.shape), after_latent.dtype)
    print("post-latent logits:", tuple(logits.shape))
    print("greedy next token:", repr(tokenizer.decode([next_id])))
    print("This is interface validation only; the adapter is not trained for Qwen3.8.")


if __name__ == "__main__":
    main()
