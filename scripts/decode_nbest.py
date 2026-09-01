#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from parakeet_context_fusion.model_io import restore_locked_asr_model


def flatten_nbest(value: Any) -> list[Any]:
    if hasattr(value, "n_best_hypotheses"):
        return list(value.n_best_hypotheses or [])
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and hasattr(value[0], "n_best_hypotheses"):
            return list(value[0].n_best_hypotheses or [])
        return list(value)
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode Parakeet with an immutable locked .nemo checkpoint")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-lock", type=Path, default=Path("locks/hf-revisions.lock.json"))
    parser.add_argument("--strategy", choices=["greedy_batch", "malsd_batch"], default="malsd_batch")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ngram-lm-model", type=Path)
    parser.add_argument("--ngram-lm-alpha", type=float, default=0.0)
    parser.add_argument("--context-phrases", type=Path)
    parser.add_argument("--boosting-tree-alpha", type=float, default=0.0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audio_paths = [str(row["audio_filepath"]) for row in rows]
    model = restore_locked_asr_model(
        lock_path=args.model_lock,
        required_revision=args.model_revision,
    )
    model.eval()

    cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
    with open_dict(cfg):
        cfg.strategy = args.strategy
        if args.strategy == "malsd_batch":
            cfg.beam.beam_size = args.beam_size
            cfg.beam.return_best_hypothesis = False
            cfg.beam.pruning_mode = "late"
            cfg.beam.blank_lm_score_mode = "lm_weighted_full"
            cfg.beam.allow_cuda_graphs = True
            if args.ngram_lm_model:
                cfg.beam.ngram_lm_model = str(args.ngram_lm_model)
                cfg.beam.ngram_lm_alpha = args.ngram_lm_alpha
            if args.context_phrases:
                cfg.malsd.boosting_tree.key_phrases_file = str(args.context_phrases)
                cfg.malsd.boosting_tree.context_score = 1.0
                cfg.malsd.boosting_tree.depth_scaling = 2.0
                cfg.malsd.boosting_tree_alpha = args.boosting_tree_alpha
    model.change_decoding_strategy(cfg, decoder_type="rnnt")
    decoded = model.transcribe(audio_paths, batch_size=args.batch_size, return_hypotheses=True)
    if isinstance(decoded, tuple) and len(decoded) == 2:
        decoded = decoded[1] or decoded[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row, item in zip(rows, decoded, strict=True):
            hypotheses = flatten_nbest(item)
            out = dict(row)
            out["candidates"] = [
                {
                    "text": str(getattr(hyp, "text", hyp)),
                    "tdt": float(getattr(hyp, "score", 0.0)),
                    "metadata": {
                        "tokens": list(getattr(hyp, "tokens", []) or []),
                        "last_frame": getattr(hyp, "last_frame", None),
                    },
                }
                for hyp in hypotheses
            ]
            sink.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
