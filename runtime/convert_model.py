from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mlx_lm.convert import convert


def attach_adapter(source: Path, output: Path, bits: int, group_size: int) -> None:
    required = ("adapter.pt", "adapter_config.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing HybridLink files in {source}: {', '.join(missing)}")
    for name in required:
        shutil.copy2(source / name, output / name)
    metadata = {
        "format": "mlx-affine",
        "weight_bits": bits,
        "group_size": group_size,
        "source_model": source.name,
        "hybridlink": "adapter.pt",
    }
    (output / "hybridchat_model.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_checkpoint(source: Path, output: Path, bits: int, group_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        config_path = output / "config.json"
        if not config_path.is_file():
            raise FileExistsError(f"Output directory exists but is incomplete: {output}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization", {})
        if int(quantization.get("bits", 0)) != bits:
            raise ValueError(f"Existing model at {output} is not {bits}-bit quantized")
        if not (output / "adapter.pt").is_file():
            attach_adapter(source, output, bits, group_size)
        return

    temporary = output.with_name(f"{output.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        convert(
            hf_path=str(source),
            mlx_path=str(temporary),
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
            q_mode="affine",
        )
        attach_adapter(source, temporary, bits, group_size)
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a HybridAgent checkpoint to MLX quantized weights")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=6, choices=(4, 6, 8))
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    convert_checkpoint(args.source, args.output, args.bits, args.group_size)


if __name__ == "__main__":
    main()
