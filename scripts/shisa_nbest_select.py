#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from parakeet_context_fusion.llm_selector import (
    build_messages,
    context_from_row,
    parse_selection,
    prompt_sha256,
    stable_prompt_candidates,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E07a: deterministically select exactly one existing ASR N-best hypothesis with Shisa V2"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="shisa-ai/shisa-v2-qwen2.5-7b")
    parser.add_argument("--revision", default="2ba1a59")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--candidate-order", choices=["stable_shuffle", "asr"], default="stable_shuffle")
    parser.add_argument(
        "--context-field",
        help="Optional dotted field containing non-reference context. Gold/reference fields are rejected.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    dtype: str | torch.dtype
    dtype = args.dtype
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        device_map=args.device_map,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    model.eval()

    rows = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), args.output.open("w", encoding="utf-8") as sink:
        for row in rows:
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"row {row.get('id')!r} has no N-best candidates")

            row_id = str(row.get("benchmark_id") or row.get("id") or "")
            prompt_candidates = stable_prompt_candidates(
                candidates,
                row_id=row_id,
                top_k=args.top_k,
                seed=args.seed,
                order=args.candidate_order,
            )
            external_context = context_from_row(row, args.context_field)
            messages = build_messages(prompt_candidates, external_context=external_context)
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

            new_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
            raw_output = tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            parse_ok = True
            fallback = False
            parse_error: str | None = None
            try:
                prompt_index = parse_selection(raw_output, candidate_count=len(prompt_candidates))
                selected_original_index = prompt_candidates[prompt_index].original_index
            except ValueError as exc:
                # E07a is one deterministic model call per utterance. Do not retry or repair with another LLM call.
                parse_ok = False
                fallback = True
                parse_error = str(exc)
                selected_original_index = 0
                prompt_index = next(
                    (
                        candidate.prompt_id
                        for candidate in prompt_candidates
                        if candidate.original_index == selected_original_index
                    ),
                    0,
                )

            selected_text = str(candidates[selected_original_index].get("text") or "")
            source_top1 = str(candidates[0].get("text") or "")
            out = dict(row)
            out["selector_selected_text"] = selected_text
            out["selector"] = {
                "experiment": "E07a",
                "model": args.model,
                "revision": args.revision,
                "top_k": min(args.top_k, len(candidates)),
                "seed": args.seed,
                "candidate_order": args.candidate_order,
                "context_field": args.context_field,
                "prompt_sha256": prompt_sha256(messages),
                "selected_prompt_index": prompt_index,
                "selected_original_index": selected_original_index,
                "source_top1_text": source_top1,
                "changed_from_source_top1": selected_text != source_top1,
                "parse_ok": parse_ok,
                "fallback_to_source_top1": fallback,
                "parse_error": parse_error,
                "raw_output": raw_output,
                "runtime": {
                    "latency_ms": latency_ms,
                    "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                    "generated_tokens": int(new_tokens.shape[-1]),
                    "peak_vram_bytes": peak_vram_bytes,
                    "transformers_version": transformers.__version__,
                    "torch_version": torch.__version__,
                    "device_map": args.device_map,
                    "dtype": args.dtype,
                },
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            sink.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
