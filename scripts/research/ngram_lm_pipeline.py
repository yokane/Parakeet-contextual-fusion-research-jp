#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def encode(args: argparse.Namespace) -> None:
    # Lazy import: only the Vast/NeMo stage needs the heavyweight ASR stack.
    from nemo.collections.asr.parts.submodules.ngram_lm import kenlm_utils
    from nemo.collections.asr.parts.submodules.ngram_lm.constants import DEFAULT_TOKEN_OFFSET

    tokenizer, encoding_level, is_aggregate, vocab_size = kenlm_utils.setup_tokenizer(str(args.model_nemo))
    if encoding_level != "subword":
        raise SystemExit(f"expected a subword tokenizer for Parakeet, got {encoding_level!r}")
    if vocab_size is None:
        raise SystemExit("NeMo tokenizer did not expose vocab_size")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kenlm_utils.iter_files(
        source_path=[str(args.corpus)],
        dest_path=[str(args.output)],
        tokenizer=tokenizer,
        encoding_level=encoding_level,
        is_aggregate_tokenizer=is_aggregate,
        verbose=1,
    )
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise SystemExit("encoded corpus is empty")

    write_json(
        args.metadata,
        {
            "schema_version": 1,
            "stage": "encode",
            "model_revision": args.model_revision,
            "encoding_level": encoding_level,
            "vocab_size": int(vocab_size),
            "token_offset": int(DEFAULT_TOKEN_OFFSET),
            "source_corpus_sha256": sha256_file(args.corpus),
            "encoded_corpus_sha256": sha256_file(args.output),
        },
    )


def estimate(args: argparse.Namespace) -> None:
    metadata = json.loads(args.encoding_metadata.read_text(encoding="utf-8"))
    expected = str(metadata.get("encoded_corpus_sha256") or "")
    actual = sha256_file(args.encoded_corpus)
    if expected and actual != expected:
        raise SystemExit(f"encoded corpus hash mismatch: {actual} != {expected}")

    args.arpa.parent.mkdir(parents=True, exist_ok=True)
    args.binary.parent.mkdir(parents=True, exist_ok=True)
    lmplz = args.kenlm_bin_dir / "lmplz"
    build_binary = args.kenlm_bin_dir / "build_binary"
    if not lmplz.is_file() or not build_binary.is_file():
        raise SystemExit(f"KenLM binaries not found in {args.kenlm_bin_dir}")

    with args.encoded_corpus.open("rb") as source, args.arpa.open("wb") as sink:
        subprocess.run(
            [
                str(lmplz),
                "-o",
                str(args.order),
                "--discount_fallback",
                "--prune",
                "0",
            ],
            stdin=source,
            stdout=sink,
            check=True,
        )
    subprocess.run([str(build_binary), "trie", str(args.arpa), str(args.binary)], check=True)

    if args.arpa.stat().st_size == 0 or args.binary.stat().st_size == 0:
        raise SystemExit("KenLM estimation produced an empty artifact")
    write_json(
        args.metadata,
        {
            "schema_version": 1,
            "stage": "estimate",
            "order": args.order,
            "kenlm_revision": args.kenlm_revision,
            "encoded_corpus_sha256": actual,
            "arpa_sha256": sha256_file(args.arpa),
            "binary_sha256": sha256_file(args.binary),
            "encoding": metadata,
        },
    )


def pack(args: argparse.Namespace) -> None:
    # Keep the conversion tied to the exact NeMo runtime used by E02.
    from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel

    encoding = json.loads(args.encoding_metadata.read_text(encoding="utf-8"))
    estimation = json.loads(args.estimation_metadata.read_text(encoding="utf-8"))
    arpa_hash = sha256_file(args.arpa)
    expected = str(estimation.get("arpa_sha256") or "")
    if expected and arpa_hash != expected:
        raise SystemExit(f"ARPA hash mismatch: {arpa_hash} != {expected}")
    vocab_size = int(encoding["vocab_size"])

    model = NGramGPULanguageModel.from_arpa(
        lm_path=str(args.arpa),
        vocab_size=vocab_size,
        normalize_unk=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(args.output))
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise SystemExit("NGPU-LM package is empty")

    write_json(
        args.metadata,
        {
            "schema_version": 1,
            "stage": "pack",
            "model_revision": args.model_revision,
            "vocab_size": vocab_size,
            "arpa_sha256": arpa_hash,
            "ngpu_lm_sha256": sha256_file(args.output),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split E02 NGPU-LM preparation across Vast and GitHub-hosted CPU")
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode")
    p_encode.add_argument("--model-nemo", type=Path, required=True)
    p_encode.add_argument("--model-revision", required=True)
    p_encode.add_argument("--corpus", type=Path, required=True)
    p_encode.add_argument("--output", type=Path, required=True)
    p_encode.add_argument("--metadata", type=Path, required=True)
    p_encode.set_defaults(func=encode)

    p_estimate = sub.add_parser("estimate")
    p_estimate.add_argument("--encoded-corpus", type=Path, required=True)
    p_estimate.add_argument("--encoding-metadata", type=Path, required=True)
    p_estimate.add_argument("--kenlm-bin-dir", type=Path, default=Path("/opt/kenlm/bin"))
    p_estimate.add_argument("--kenlm-revision", required=True)
    p_estimate.add_argument("--order", type=int, default=6)
    p_estimate.add_argument("--arpa", type=Path, required=True)
    p_estimate.add_argument("--binary", type=Path, required=True)
    p_estimate.add_argument("--metadata", type=Path, required=True)
    p_estimate.set_defaults(func=estimate)

    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--arpa", type=Path, required=True)
    p_pack.add_argument("--encoding-metadata", type=Path, required=True)
    p_pack.add_argument("--estimation-metadata", type=Path, required=True)
    p_pack.add_argument("--model-revision", required=True)
    p_pack.add_argument("--output", type=Path, required=True)
    p_pack.add_argument("--metadata", type=Path, required=True)
    p_pack.set_defaults(func=pack)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
