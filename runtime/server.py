from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_engine import MLXHybridAgent

logging.basicConfig(
    level=os.environ.get("HYBRIDCHAT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hybridchat.engine")


class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.lock = threading.RLock()
        self.engine: MLXHybridAgent | None = None
        self.model_id: str | None = None
        self.loaded_at: float | None = None
        self.mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hybridchat-mlx")

    async def run_mlx(self, function, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.mlx_executor, function, *args)

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "model": self.model_id,
                "device": "mps",
                "loaded": self.engine is not None,
                "loadedAt": self.loaded_at,
                "modelsDirectory": str(self.models_dir),
                "tools": list(self.engine.tools.tools) if self.engine else [],
            }

    def load(self, model_id: str, model_path: str | None = None) -> dict[str, Any]:
        allowed = {"HybridAgent-8B-Agentic", "HybridAgent-14B-Agentic"}
        if model_id not in allowed:
            raise ValueError(f"Unsupported model: {model_id}")
        model_dir = Path(model_path) if model_path else self.models_dir / model_id
        if not (model_dir / "adapter.pt").is_file():
            raise FileNotFoundError(f"Quantized HybridAgent model not installed: {model_dir}")
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config is missing: {config_path}")

        with self.lock:
            if self.model_id == model_id and self.engine is not None:
                return self.status()
            self._unload_locked()
            logger.info("loading %s from %s", model_id, model_dir)
            self.engine = MLXHybridAgent(model_dir, record_latent_readout=True)
            self.model_id = model_id
            self.loaded_at = time.time()
            logger.info("loaded %s", model_id)
            return self.status()

    def unload(self) -> dict[str, Any]:
        with self.lock:
            self._unload_locked()
            logger.info("unloaded model and cleared MLX memory")
            return self.status()

    def _unload_locked(self) -> None:
        if self.engine is not None:
            self.engine.unload()
        self.engine = None
        self.model_id = None
        self.loaded_at = None

    def generate(self, body: dict, on_delta=None) -> dict[str, Any]:
        messages = body.get("messages") or []
        question = question_from_messages(messages)
        with self.lock:
            if self.engine is None:
                raise RuntimeError("No model is loaded. Select a model and press Load.")
            return self.engine.generate(question, body, on_delta=on_delta)


def build_openai_response(model_id: str, result: dict[str, Any]) -> dict[str, Any]:
    content = result.get("output_rendered") or result.get("output", "")
    meta = {
        key: result.get(key)
        for key in (
            "answer",
            "end_reason",
            "turns",
            "generation_steps",
            "latent_steps",
            "latent_spans",
            "python_calls",
            "search_calls",
            "wall_time_s",
        )
    }
    prompt_tokens = int(result.get("prompt_tokens", 0))
    completion_tokens = int(result.get("generation_steps", 0))
    return {
        "id": f"chatcmpl-hybrid-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "hybrid": meta,
    }


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return "" if content is None else str(content)


def question_from_messages(messages: list[dict]) -> str:
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages:
        raise ValueError("No user message in request")
    question = text_content(user_messages[-1].get("content")).strip()
    history = []
    for message in messages[:-1][-8:]:
        role = message.get("role")
        content = text_content(message.get("content")).strip()
        if role == "user" and content:
            history.append(f"User: {content}")
        elif role == "assistant" and content:
            answers = re.findall(r"<answer>([\s\S]*?)</answer>", content)
            summary = answers[-1].strip() if answers else re.sub(
                r"<latent>[\s\S]*?</latent>", "", content
            ).strip()[:1800]
            if summary:
                history.append(f"Assistant: {summary}")
    if history:
        history_text = "\n".join(history)
        return f"Conversation so far:\n{history_text}\n\nCurrent question:\n{question}"
    return question


def event_json(value: dict[str, Any]) -> str:
    return f"data: {json.dumps(value, ensure_ascii=False)}\n\n"


def create_app(manager: ModelManager) -> FastAPI:
    app = FastAPI(title="HybridChat MLX Engine")

    @app.get("/health")
    def health():
        return manager.status()

    @app.get("/v1/models")
    def models():
        loaded = manager.status()["model"]
        records = [] if loaded is None else [{"id": loaded, "object": "model", "owned_by": "local"}]
        return {"object": "list", "data": records}

    @app.post("/admin/models/load")
    async def load_model(request: dict):
        model_id = request.get("model_id") or request.get("modelId")
        try:
            return await manager.run_mlx(
                manager.load,
                str(model_id),
                request.get("model_path") or request.get("modelPath"),
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            logger.exception("model load failed")
            raise HTTPException(status_code=500, detail=f"model load failed: {error}") from error

    @app.post("/admin/models/unload")
    async def unload_model():
        try:
            return await manager.run_mlx(manager.unload)
        except Exception as error:
            logger.exception("model unload failed")
            raise HTTPException(status_code=500, detail=f"model unload failed: {error}") from error

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        status = await manager.run_mlx(manager.status)
        if not status["loaded"]:
            raise HTTPException(status_code=409, detail="Load a HybridAgent model before chatting")
        model_id = status["model"]
        if not request.get("stream"):
            try:
                result = await manager.run_mlx(manager.generate, request)
            except Exception as error:
                logger.exception("generation failed")
                raise HTTPException(status_code=500, detail=f"generation failed: {error}") from error
            response = build_openai_response(model_id, result)
            return JSONResponse(response)

        response_id = f"chatcmpl-hybrid-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async def stream_response():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            client_connected = threading.Event()
            client_connected.set()

            def on_delta(content: str) -> None:
                if not content or not client_connected.is_set():
                    return
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, ("delta", content))
                except RuntimeError:
                    client_connected.clear()

            async def generate():
                try:
                    result = await manager.run_mlx(manager.generate, request, on_delta)
                except Exception as error:
                    logger.exception("generation failed")
                    queue.put_nowait(("error", str(error)))
                else:
                    queue.put_nowait(("complete", result))

            generation = asyncio.create_task(generate())
            yield event_json({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            })

            try:
                while True:
                    kind, value = await queue.get()
                    if kind == "delta":
                        yield event_json({
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"content": value}, "finish_reason": None}],
                        })
                    elif kind == "error":
                        yield event_json({"error": {"message": f"generation failed: {value}", "type": "server_error"}})
                        yield "data: [DONE]\n\n"
                        break
                    else:
                        response = build_openai_response(model_id, value)
                        yield event_json({
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": response["usage"],
                        })
                        yield event_json({
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [],
                            "hybrid": response["hybrid"],
                            "usage": response["usage"],
                        })
                        yield "data: [DONE]\n\n"
                        break
            finally:
                client_connected.clear()

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="HybridChat managed MLX inference service")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    load_dotenv()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    manager = ModelManager(args.models_dir)
    uvicorn.run(create_app(manager), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
