from __future__ import annotations

import argparse
import io
import os
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


HYBRID_AGENT_COMMIT = "1b41cebbd2388d083ee1820af421b7f9e24affb5"


def ensure_engine_source(destination: Path, commit: str = HYBRID_AGENT_COMMIT) -> Path:
    package = destination / "hybrid_agent"
    if (package / "__init__.py").is_file():
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    archive_url = f"https://codeload.github.com/togethercomputer/HybridAgent/tar.gz/{commit}"
    request = urllib.request.Request(archive_url, headers={"User-Agent": "HybridChat"})
    print(f"Fetching HybridAgent inference source at {commit}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        archive_bytes = response.read()

    written = 0
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            try:
                package_index = parts.index("hybrid_agent")
            except ValueError:
                continue
            relative = PurePosixPath(*parts[package_index:])
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f"Invalid path in HybridAgent source archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as output:
                output.write(source.read())
            written += 1

    if not (package / "__init__.py").is_file() or written == 0:
        raise RuntimeError("HybridAgent source archive did not contain the inference package")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--commit", default=HYBRID_AGENT_COMMIT)
    args = parser.parse_args()
    path = ensure_engine_source(args.source_dir, args.commit)
    print(f"HybridAgent inference source ready at {path}", flush=True)


if __name__ == "__main__":
    main()
