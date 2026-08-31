#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict


def flatten_nbest(value: Any) -> list[Any]:
    if hasattr(value, "n_best_hypotheses"):
        return list(value.n_best_hypotheses or [])
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and hasattr(value[0], "n_best_hypotheses"):
            return list(value[0].n_best_hypotheses or [])
        return list(value)
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="nvidia/parakeet-tdt_ctc-0.6b-ja")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ngram-lm-model", type=Path)
    parser.add_argument("--ngram-lm-alpha", type=float, default=0.0)
    parser.add_argument("--context-phrases", type=Path)
    parser.add_argument("--boosting-tree-alpha", type=float, default=0.0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    audio_paths = [str(row["audio_filepath"]) for row in rows]
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    model.eval()
    cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
    with open_dict(cfg):
        cfg.strategy = "malsd_batch"
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
    try:
        model.change_decoding_strategy(cfg, decoder_type="rnnt")
    except TypeError:
        model.change_decoding_strategy(cfg)
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
