# Qwen3.8-27B HybridAgent feasibility

## Findings

`Qwen/Qwen3.8-27B` is public and Apache-2.0 licensed. It is a **post-trained vision-language model**, not a plain Qwen3 causal LM. The published checkpoint has 27.8B BF16 parameters and about 55.6 GB of model files. Its language stack is 64 layers (48 Gated DeltaNet/linear-attention layers plus 16 full-attention layers), with a 5,120-wide hidden state. I did not find an official `Qwen3.8-27B-Base` checkpoint in the Hub search; the public Qwen repo is the post-trained model.

The 14B HybridAgent `adapter.pt` has the same 5,120 hidden width, but **matching tensor shapes does not make it a valid Qwen3.8 adapter**. Its layer activations and attention/recurrent states come from a different backbone. The adapter and HybridSFT stages need to be trained or validated on Qwen3.8; blindly attaching the 14B adapter would not be a responsible release.

The existing HybridAgent loader is built around `AutoModelForCausalLM` and Qwen3's ordinary transformer cache. Qwen3.8 is `Qwen3_5ForConditionalGeneration`, includes a vision encoder, and interleaves recurrent DeltaNet state with attention KV state. A Qwen3.8 HybridAgent therefore needs a new language-backbone wrapper and cache-aware generation path. For an initial prototype, the sensible scope is **text-only**: leave image/video features out, wrap the Qwen3.8 language model, then add the `<latent>`/`</latent>` transitions and a freshly trained HybridLink.

There is a promising Mac path: MLX-VLM already supports Qwen3.8 quantized checkpoints, and its Qwen3.5 language model accepts external input embeddings. That is the interface HybridLink needs. I ran a one-step compatibility probe with `mlx-community/Qwen3.8-27B-4bit`: the text backbone accepted the adapter vector and returned the next hidden state/logits (`[1, 1, 5120]` hidden, `[1, 1, 248320]` logits). I also loaded the Q4 checkpoint on the M5 Max and generated a short text-only continuation; it used about 16.4 GB peak memory. That token-capped run is only a load/smoke test, not a quality or throughput benchmark. The probe uses the released 14B adapter only to validate tensor/interface compatibility; it is **not trained for Qwen3.8**, so its generated result is not an evaluation. The reproducible probe is [`experiments/qwen38_compatibility_probe.py`](../experiments/qwen38_compatibility_probe.py). Qwen3.8's mixed recurrent/attention cache and trained latent behavior still need their own implementation and tests.

## Quantization and Mac fit

Available Apple-Silicon MLX checkpoints include `mlx-community/Qwen3.8-27B-4bit` and `mlx-community/Qwen3.8-27B-8bit`. Unsloth also publishes an NVFP4 compressed-tensors build and dynamic GGUF variants. The NVFP4 build is oriented toward NVIDIA kernels; the MLX checkpoints are the practical candidates for an M5 Max. A GGUF Q4 model is also usable through llama.cpp/LM Studio, but it does not run HybridLink by itself.

The full BF16 checkpoint exceeds this Mac's 48 GB unified memory before allocating the KV/recurrent cache or latent readout bank. A 4-bit MLX build is much more plausible for inference; an 8-bit build may fit with a carefully limited context, but leaves less room for cache and other applications. Neither quantized checkpoint is a HybridAgent until the Qwen3.8 HybridLink work is done.

## Proposed prototype and training sequence

1. Extend the compatibility probe into a text-only generation harness around the public Qwen3.8-27B MLX 4-bit checkpoint, including the correct mixed DeltaNet/KV cache handling and anchor-token transitions.
2. Port the HybridLink residual adapter to MLX and verify one latent transition, exit-anchor readout, and a short text-only trajectory. Start from a freshly initialized adapter; optionally compare a 14B adapter warm-start as an experiment, not as the presumed solution.
3. Adapt the HybridAgent trajectory formatter and model loader for Qwen3.8, add the anchor tokens, and run the three HybridSFT stages using the released HybridSFT-v2 trajectories. Evaluate at each stage against stock Qwen3.8.
4. Only after SFT shows stable hybrid trajectories, run HybridRL on the released math/search tasks and compare answer accuracy, tool success, latency, text-token usage, and latent-step usage.
5. Treat multimodal image/video hybrid reasoning as a later phase; it requires extending the trajectory format, processor, tool loop, and training/evaluation data.

The released HybridAgent project recommends 8 or 16 GPUs for SFT and 32 GPUs (4×8) for RL on its 8B/14B experiments. A full 27B adaptation would need an external multi-GPU training setup and likely more memory/compute than those baselines. The M5 Max is suitable for quantized inference experiments, not for reproducing the full 27B SFT+RL pipeline.

To reproduce the one-step interface check, install `mlx-vlm` in a Python 3.11 environment, then run from the HybridChat checkout:

```sh
python -m pip install mlx-vlm torch
python experiments/qwen38_compatibility_probe.py \
  --model mlx-community/Qwen3.8-27B-4bit \
  --adapter /path/to/HybridAgent-14B-Agentic/adapter.pt
```

This downloads the public MLX Q4 checkpoint if it is not cached and prints the hidden/latent tensor shapes. The 14B adapter is only a shape-compatible stand-in for this interface test; it is not a valid Qwen3.8 HybridLink checkpoint.

## References

- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8-27B MLX 4-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
- [Qwen3.8-27B MLX 8-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit)
- [Unsloth Qwen3.8-27B NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
- [HybridAgent repository](https://github.com/togethercomputer/HybridAgent)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
