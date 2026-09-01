#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def offers(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        value = payload.get("offers", payload.get("results", []))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Vast offers by predicted total research cost")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predicted-minutes", type=float, required=True)
    parser.add_argument("--max-cost-usd", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_cost_usd <= 0 or args.max_cost_usd > 0.5:
        raise SystemExit("Vast research budget must be > 0 and <= $0.50")
    if args.predicted_minutes <= 0:
        raise SystemExit("predicted-minutes must be positive")

    ranked: list[dict[str, Any]] = []
    for raw in offers(json.loads(args.input.read_text(encoding="utf-8"))):
        dph = float(raw.get("dph_total") or 0)
        if dph <= 0:
            continue
        cuda = float(raw.get("cuda_max_good") or 0)
        disk = float(raw.get("disk_space") or 0)
        if cuda < 13 or disk < 50 or raw.get("rentable") is False or raw.get("verified") is False:
            continue
        predicted_cost = round(dph * (args.predicted_minutes / 60.0), 4)
        if predicted_cost > args.max_cost_usd:
            continue
        reliability = float(raw.get("reliability") or 0)
        confidence = "high" if reliability >= 0.98 else ("medium" if reliability >= 0.95 else "low")
        offer_id = raw.get("id", raw.get("ask_contract_id"))
        if offer_id is None:
            continue
        ranked.append(
            {
                "offer_id": int(offer_id),
                "gpu_name": str(raw.get("gpu_name", "unknown")),
                "dph_total": dph,
                "predicted_minutes": args.predicted_minutes,
                "predicted_cost_usd": predicted_cost,
                "confidence": confidence,
                "reliability": reliability,
                "disk_space_gb": disk,
                "cuda_max_good": cuda,
            }
        )

    ranked.sort(key=lambda item: (item["predicted_cost_usd"], -item["reliability"], item["offer_id"]))
    result = {
        "schema_version": 1,
        "platform": "linux/amd64",
        "cuda_major": 13,
        "max_cost_usd": args.max_cost_usd,
        "storage_gb": 50,
        "offers": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("|Rank|Offer|GPU|$/h|Minutes|Predicted $|Confidence|")
    print("|---:|---:|---|---:|---:|---:|---|")
    for index, item in enumerate(ranked[:20], 1):
        print(
            f"|{index}|{item['offer_id']}|{item['gpu_name']}|{item['dph_total']:.4f}|"
            f"{item['predicted_minutes']:.2f}|{item['predicted_cost_usd']:.4f}|{item['confidence']}|"
        )
    if not ranked:
        print("|—|—|No eligible offer|—|—|—|—|")


if __name__ == "__main__":
    main()
