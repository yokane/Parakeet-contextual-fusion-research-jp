#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from parakeet_context_fusion.ctc_local import ctc_sequence_logprob
from parakeet_context_fusion.phoneme import PhoneCTCHead


def load_states(path: Path) -> torch.Tensor:
    item = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(item, dict):
        item = item["encoder_states"]
    return item.float()


def benchmark_id(row: dict[str, object]) -> str:
    value = str(row.get("benchmark_id") or row.get("id") or "").strip()
    if not value:
        raise RuntimeError("E05 input row has neither benchmark_id nor id")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PhoneCTCHead(
        checkpoint["encoder_dim"],
        checkpoint["phone_vocab_size"],
        dropout=checkpoint.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    blank_id = int(checkpoint["blank_id"])
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), args.output.open("w", encoding="utf-8") as sink:
        for row in rows:
            sample_id = benchmark_id(row)
            feature_path = args.feature_dir / f"{sample_id}.pt"
            if not feature_path.is_file():
                raise RuntimeError(f"encoder feature not found for {sample_id}: {feature_path}")
            states = load_states(feature_path).to(args.device)
            for candidate in row["candidates"]:
                phone_ids = candidate.get("metadata", {}).get("phone_ids")
                if not phone_ids:
                    candidate["phone"] = 0.0
                    continue
                window = candidate.get("metadata", {}).get("ctc_window", [0, states.shape[0]])
                start, end = int(window[0]), int(window[1])
                local_states = states[max(0, start) : min(states.shape[0], end)]
                log_probs = model(local_states)
                score = ctc_sequence_logprob(
                    log_probs,
                    phone_ids,
                    blank_id=blank_id,
                    length_norm_power=1.0,
                )
                candidate["phone"] = float(score.cpu())
                base = float(candidate.get("fused_score", candidate.get("tdt", 0.0)))
                candidate["fused_score"] = base + args.alpha * candidate["phone"]
            row["candidates"].sort(
                key=lambda item: item.get("fused_score", item.get("tdt", 0.0)), reverse=True
            )
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
