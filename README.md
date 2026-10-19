# HybridChat

HybridChat is a local desktop chat application for Together AI’s [HybridAgent](https://github.com/togethercomputer/HybridAgent) 8B and 14B models. It combines an OpenCode-inspired multi-session interface with the model’s distinctive HybridLink reasoning: normal text, continuous latent thought, and Python/tool observations in one trajectory.

The app manages the full local workflow: choose models, check disk space, download and quantize checkpoints, load one model into Apple Silicon unified memory, chat, then unload it to release memory. Model weights are **not** stored in this GitHub repository.

## What makes HybridChat specific to HybridAgent

- Shows latent spans, Python/search calls, tool results, and final boxed answers as separate styled blocks.
- Displays per-response step, latent, tool-call, and timing statistics.
- Keeps separate, persistent chat sessions in tabs; recent turns are included as compact conversation context.
- Uses HybridAgent’s `adapter.pt` HybridLink weights when generating latent inputs. The app does not replace this with ordinary text-only GGUF inference.
- Runs a local OpenAI-compatible API while the app is open, so other local clients such as OpenCode can use the loaded model.

## First launch and model downloads

On first launch, HybridChat opens a model setup dialog with checkboxes for either or both released checkpoints. It checks available space on the destination volume and blocks installation when the selected models and runtime cannot fit. Downloads resume from partial files and validate the large Hugging Face LFS files against their SHA-256 checksums.

HybridAgent publishes BF16 checkpoints. HybridChat downloads those official files, converts the backbone to a **Mac-native MLX 6-bit affine checkpoint** (group size 64), and retains the original HybridLink adapter and tokenizer. This quantized checkpoint is saved to disk before loading; model weights are not requantized on every app launch.

| Model | BF16 source download | Installed MLX 6-bit model | Peak free space for a fresh install* |
|---|---:|---:|---:|
| HybridAgent 8B Agentic | ~16.46 GB | ~6.73 GB | ~24.3 GB |
| HybridAgent 14B Agentic | ~29.65 GB | ~12.11 GB | ~42.8 GB |
| Both, installed sequentially | ~46.11 GB downloaded in total | ~18.85 GB | ~42.8 GB |

\*The peak-space estimate includes source and quantized files being present during conversion, plus about 1 GB reserved for the managed inference runtime. The app converts the larger model first and removes each temporary BF16 source after its quantized output has been verified. If a complete BF16 source is already present in a recognized local model directory, only the quantized output and runtime reserve are needed.

The model files live under HybridChat’s per-user application data directory (on macOS, `~/Library/Application Support/com.hybridagent.hybridchat/models`). The first install also prepares a private Python 3.11 environment and downloads the HybridAgent inference package pinned at [`1b41cebb`](https://github.com/togethercomputer/HybridAgent/commit/1b41cebbd2388d083ee1820af421b7f9e24affb5). If `uv` is not already installed, HybridChat obtains it and uses it to install Python and the required packages in app data. No external HybridAgent server is required.

## Model selection and memory

The toolbar model menu lists the available 8B and 14B checkpoints. Use **Load** to load the selected model, **Unload** to release its MLX weights and latent-readout memory, or select another model and load it to switch. Only one model is loaded at a time. On Apple Silicon, “VRAM” is shared unified memory; the app reports the active model and MLX device in the status area.

HybridChat currently targets **macOS on Apple Silicon**. It uses MLX for quantized inference and the custom latent-input loop. The Q6 files preserve HybridLink; the 8-bit experimental files are higher-fidelity but reduce disk storage by slightly less than half. See [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md) for the quantization trade-offs and [`docs/QWEN38_PLAN.md`](docs/QWEN38_PLAN.md) for the Qwen3.8-27B prototype and training plan. Qwen3.8 is not offered in the app until a HybridLink-trained checkpoint is validated.

## Run from source

Prerequisites: macOS on Apple Silicon, Rust, Tauri CLI v2, and an internet connection for the first-time runtime/model setup.

From the HybridChat checkout:

```sh
cargo tauri dev
```

Build a local macOS app bundle:

```sh
cargo tauri build --bundles app
```

The bundle is written to `src-tauri/target/release/bundle/macos/HybridChat.app`. A local unsigned build can be opened from the build folder; distribution outside your machine requires normal macOS code signing/notarization.

## OpenAI-compatible API

When the managed engine starts, HybridChat prefers `http://127.0.0.1:8100/v1`; if that port is occupied it chooses another local port. The active URL is shown under **settings** and can be copied. The local server binds only to loopback and exposes standard `/v1/models` and `/v1/chat/completions`, plus local load/unload management routes.

Example OpenCode provider configuration (replace the URL with the one shown by HybridChat if the port differs):

```json
{
  "provider": {
    "hybridchat-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "HybridChat (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8100/v1",
        "apiKey": "local"
      },
      "models": {
        "HybridAgent-8B-Agentic": { "name": "HybridAgent 8B" },
        "HybridAgent-14B-Agentic": { "name": "HybridAgent 14B" }
      }
    }
  }
}
```

Only the model loaded in HybridChat can serve requests. The current HybridAgent Python server computes a complete trajectory before sending its SSE response, so clients see a thinking interval followed by the rendered response stream. Stopping in HybridChat stops display; it cannot cancel an inference already running in the service.

## Search tool

The Python tool is available by default. Search is optional and requires a Tavily API key in the environment when HybridChat launches (`TAVILY_API_KEY`). Without one, a model-requested search returns an unavailable-tool observation and generation continues. The Python tool runs generated code as the current user in a subprocess; it is not a full operating-system sandbox.

## Technology

- **Rust + Tauri v2**: native desktop shell, disk-space inspection, resumable/verified Hugging Face downloads, private Python/MLX runtime management, model lifecycle, and OpenAI-compatible streaming bridge.
- **HTML, CSS, vanilla JavaScript**: the multi-tab, model-aware chat UI; no Node frontend toolchain is required.
- **MLX / mlx-lm**: Apple Silicon quantization and inference. HybridChat’s MLX engine ports HybridLink’s residual adapter and the text/latent/tool loop.
- **Transformers / PyTorch**: import-time support for the released adapter and checkpoints; the interactive model forward pass is MLX-native.
- **Hugging Face Hub**: pinned official BF16 HybridAgent model snapshots; the app converts them locally to the installed Q6 MLX format.

## Qwen3.8-27B exploration

HybridChat includes a research plan for a text-only HybridLink prototype on Qwen3.8-27B and the extra training needed before calling it a HybridAgent. Qwen3.8 is a multimodal, post-trained model with a different mixed recurrent/attention architecture; attaching the 14B adapter is not sufficient. See [`docs/QWEN38_PLAN.md`](docs/QWEN38_PLAN.md).

## Licensing and model files

HybridChat application code is MIT-licensed. The app downloads HybridAgent inference code from Together AI’s MIT-licensed repository at a pinned revision. HybridAgent model files remain on Hugging Face and are downloaded by the app; the released HybridAgent checkpoints are MIT-licensed. Qwen3.8-27B is Apache-2.0 licensed. No weights, runtime virtual environment, caches, or generated model files are committed here.
