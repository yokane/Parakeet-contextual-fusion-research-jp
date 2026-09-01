#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from parakeet_context_fusion.ctc_local import FrameWindow, expand_window, local_ctc_score
from parakeet_context_fusion.model_io import restore_locked_asr_model


def extract_ctc_hypothesis(item):
    if isinstance(item, tuple):
        item = item[0]
    if isinstance(item, list):
        item = item[0]
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank TDT N-best with the locked model's CTC branch")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-lock", type=Path, default=Path("locks/hf-revisions.lock.json"))
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--margin-frames", type=int, default=8)
    parser.add_argument("--frames-per-token", type=int, default=4)
    parser.add_argument("--min-window-frames", type=int, default=16)
    parser.add_argument("--full-utterance", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    audio_paths = [str(row["audio_filepath"]) for row in rows]
    model = restore_locked_asr_model(
        lock_path=args.model_lock,
        required_revision=args.model_revision,
    )
    model.eval()
    model.change_decoding_strategy(decoder_type="ctc")
    ctc_outputs = model.transcribe(audio_paths, batch_size=1, return_hypotheses=True)
    blank_id = int(getattr(model.ctc_decoding, "blank_id", model.ctc_decoder.num_classes_with_blank - 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as sink:
        for row, raw_hyp in zip(rows, ctc_outputs, strict=True):
            hyp = extract_ctc_hypothesis(raw_hyp)
            alignments = getattr(hyp, "alignments", None)
            if alignments is None:
                raise RuntimeError("CTC hypothesis did not expose frame alignments/log probabilities")
            log_probs = alignments if isinstance(alignments, torch.Tensor) else torch.as_tensor(alignments)
            if log_probs.ndim != 2:
                raise RuntimeError(f"CTC alignments must be [T,V], got {tuple(log_probs.shape)}")
            total_frames = log_probs.shape[0]
            for candidate in row["candidates"]:
                token_ids = model.tokenizer.text_to_ids(candidate["text"])
                if args.full_utterance:
                    window = FrameWindow(0, total_frames)
                else:
                    end = candidate.get("metadata", {}).get("last_frame")
                    if end is None:
                        window = FrameWindow(0, total_frames)
                    else:
                        width = max(args.min_window_frames, len(token_ids) * args.frames_per_token)
                        start = max(0, int(end) - width)
                        window = expand_window(
                            start,
                            min(total_frames, int(end) + 1),
                            margin=args.margin_frames,
                            total_frames=total_frames,
                        )
                score = local_ctc_score(
                    log_probs,
                    token_ids,
                    window=window,
                    blank_id=blank_id,
                    length_norm_power=1.0,
                )
                candidate["ctc_local"] = float(score.cpu())
                candidate.setdefault("metadata", {})["ctc_window"] = [window.start, window.end]
                base = float(candidate.get("fused_score", candidate.get("tdt", 0.0)))
                candidate["fused_score"] = base + args.alpha * float(score)
            row["candidates"].sort(key=lambda item: item["fused_score"], reverse=True)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
