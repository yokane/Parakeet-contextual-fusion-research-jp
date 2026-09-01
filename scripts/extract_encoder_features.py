#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from parakeet_context_fusion.model_io import restore_locked_asr_model


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_encoded(encoded: torch.Tensor, encoded_len: torch.Tensor | None) -> torch.Tensor:
    if encoded.ndim != 3 or encoded.shape[0] != 1:
        raise RuntimeError(f"expected encoder output [1,*,*], got {tuple(encoded.shape)}")
    value = encoded[0].detach().float().cpu()
    length = int(encoded_len.reshape(-1)[0].item()) if encoded_len is not None else None
    if length is not None:
        if value.shape[0] == length:
            return value[:length]
        if value.shape[1] == length:
            return value[:, :length].transpose(0, 1).contiguous()
    # FastConformer encoder convention is generally [B,D,T]. Prefer the smaller
    # axis as feature dimension only when the length tensor was unavailable.
    if value.shape[0] > value.shape[1]:
        return value
    return value.transpose(0, 1).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Parakeet encoder states for E05 phone scoring")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/encoder"))
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-lock", type=Path, default=Path("locks/hf-revisions.lock.json"))
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    model = restore_locked_asr_model(lock_path=args.model_lock, required_revision=args.model_revision)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        encoded = output
        encoded_len = None
        if isinstance(output, tuple):
            encoded = output[0]
            if len(output) > 1 and isinstance(output[1], torch.Tensor):
                encoded_len = output[1]
        if not isinstance(encoded, torch.Tensor):
            raise RuntimeError("encoder hook did not receive a tensor output")
        capture["encoded"] = encoded.detach()
        if encoded_len is not None:
            capture["encoded_len"] = encoded_len.detach()

    handle = model.encoder.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for row in rows:
                benchmark_id = str(row.get("benchmark_id") or row.get("id") or "")
                audio_path = str(row.get("audio_filepath") or "")
                if not benchmark_id or not audio_path:
                    raise RuntimeError("manifest rows must contain benchmark_id/id and audio_filepath")
                capture.clear()
                model.transcribe([audio_path], batch_size=1, return_hypotheses=False)
                encoded = capture.get("encoded")
                if encoded is None:
                    raise RuntimeError(f"encoder hook produced no output for {benchmark_id}")
                states = normalize_encoded(encoded, capture.get("encoded_len"))
                torch.save(
                    {
                        "encoder_states": states,
                        "benchmark_id": benchmark_id,
                        "audio_filepath": audio_path,
                        "model_revision": args.model_revision,
                    },
                    args.output_dir / f"{benchmark_id}.pt",
                )
    finally:
        handle.remove()

    print(f"wrote {len(rows)} encoder feature files to {args.output_dir}")


if __name__ == "__main__":
    main()
