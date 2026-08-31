from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fusion import CandidateScores, FusionWeights, rerank
from .phoneme import weighted_phone_distance


def cmd_phone_distance(args: argparse.Namespace) -> None:
    print(weighted_phone_distance(args.left.split(), args.right.split()))


def cmd_rerank(args: argparse.Namespace) -> None:
    weights = FusionWeights(
        lm=args.lm_alpha,
        phrase_boost=args.pb_alpha,
        ctc_local=args.ctc_alpha,
        phone=args.phone_alpha,
        entity_context=args.context_alpha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates = [CandidateScores(**candidate) for candidate in row["candidates"]]
            ranked = rerank(candidates, weights)
            row["ranked"] = [
                {"text": candidate.text, "fused_score": candidate.fused(weights)} for candidate in ranked
            ]
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parakeet-context-fusion")
    sub = parser.add_subparsers(required=True)
    phone = sub.add_parser("phone-distance")
    phone.add_argument("left")
    phone.add_argument("right")
    phone.set_defaults(func=cmd_phone_distance)
    rerank_parser = sub.add_parser("rerank")
    rerank_parser.add_argument("--input", type=Path, required=True)
    rerank_parser.add_argument("--output", type=Path, required=True)
    rerank_parser.add_argument("--lm-alpha", type=float, default=0.0)
    rerank_parser.add_argument("--pb-alpha", type=float, default=0.0)
    rerank_parser.add_argument("--ctc-alpha", type=float, default=0.0)
    rerank_parser.add_argument("--phone-alpha", type=float, default=0.0)
    rerank_parser.add_argument("--context-alpha", type=float, default=0.0)
    rerank_parser.set_defaults(func=cmd_rerank)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
