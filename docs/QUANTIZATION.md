# HybridAgent quantization on Apple Silicon

## Choice used by HybridChat

The released checkpoints are **BF16**, rather than FP16. HybridChat converts them to MLX's 6-bit affine weight quantization (group size 64) on first install. MLX stores the packed quantized weights as files on disk and loads them with Apple Silicon Metal kernels. The separate `adapter.pt` HybridLink remains BF16, so the app can still inject continuous latent vectors.

| Checkpoint | BF16 source files | MLX Q6 installed files | Approx. reduction |
|---|---:|---:|---:|
| HybridAgent 8B | 16.46 GB | 6.73 GB | 59% |
| HybridAgent 14B | 29.65 GB | 12.11 GB | 59% |

The first-run space check includes the BF16 source, the converted output coexisting during conversion, and a runtime reserve. The app converts one checkpoint at a time, largest first, then removes the temporary BF16 source after validating the output. It retains the quantized checkpoint and HybridLink adapter.

## Why not simply download a GGUF?

Good 4-bit GGUFs already exist for both released models. For example, the 14B Q4_K_M is around 9.3 GB. Unsloth's Dynamic GGUFs are another strong quality/size option. These are excellent for llama.cpp or LM Studio **as ordinary text-generation models**, but those runtimes do not execute HybridAgent's custom HybridLink step: the latent vector must be injected as the next input embedding and the hidden state used to decide when to leave latent mode. A normal GGUF chat therefore loses the distinctive hybrid reasoning behavior.

The current app uses MLX quantized checkpoints instead. Its generation loop calls the model with external `input_embeddings`, runs the MLX HybridLink adapter, then continues over the same cache. This keeps the app's latent spans and tool loop intact.

## Why not NF4, INT8, or NVFP4 here?

- bitsandbytes NF4/LLM.int8 are popular for CUDA/NVIDIA and some CPU workflows, but its documented accelerator support does not include Apple MPS. They are not the straightforward M5 Max runtime format.
- Unsloth's Qwen3.8 NVFP4 checkpoint is mixed 4/8-bit compressed-tensors intended for NVIDIA-oriented kernels; it is not the native Metal choice.
- MLX's affine 6-bit format is supported by the same Apple-native runtime used by HybridChat, reduces the released weight files by more than half, and keeps the adapter interface available.

An 8-bit MLX conversion is also possible and is a little closer to BF16, but the tested outputs were about 8.78 GB (8B) and 15.81 GB (14B)—roughly 47% smaller, so it misses a strict 50% reduction. The app therefore defaults to Q6. A future advanced setting could offer Q8 for users who prefer the extra fidelity over the extra storage.

## Validation so far

The 6-bit 8B and 14B models both completed local HybridLink latent steps, pseudo-text readout, and Python tool calls on the M5 Max. Sanity prompts returned the same answers as the BF16 versions (8B: `19 × 23 = 437`; 14B: `2 + 2 = 4`). These are correctness smoke tests, not a broad benchmark; the random trajectories and latency vary between runs. A proper quality comparison should run the released HybridAgent evaluation sets for BF16 vs Q6 before calling the quantization quality equivalent.
