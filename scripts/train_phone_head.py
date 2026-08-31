#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from parakeet_context_fusion.phoneme import PhoneCTCHead


def load_feature(path: Path) -> torch.Tensor:
    item = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(item, dict):
        item = item["encoder_states"]
    if item.ndim != 2:
        raise ValueError(f"{path}: expected [T,D] encoder states, got {tuple(item.shape)}")
    return item.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phone-vocab-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise SystemExit("empty phone-head manifest")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    first = load_feature(Path(rows[0]["feature_path"]))
    encoder_dim = int(first.shape[-1])
    blank_id = args.phone_vocab_size - 1
    model = PhoneCTCHead(encoder_dim, args.phone_vocab_size, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        random.shuffle(rows)
        total_loss = 0.0
        seen = 0
        model.train()
        for row in rows:
            states = load_feature(Path(row["feature_path"])).to(args.device)
            target = torch.tensor(row["phone_ids"], dtype=torch.long, device=args.device)
            if target.numel() == 0 or states.shape[0] < target.numel():
                continue
            log_probs = model(states)
            loss = F.ctc_loss(
                log_probs.unsqueeze(1), target,
                input_lengths=torch.tensor([states.shape[0]], device=args.device),
                target_lengths=torch.tensor([target.numel()], device=args.device),
                blank=blank_id, reduction="mean", zero_infinity=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            seen += 1
        print(f"epoch={epoch + 1} examples={seen} mean_ctc_loss={total_loss / max(1, seen):.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "encoder_dim": encoder_dim,
        "phone_vocab_size": args.phone_vocab_size,
        "blank_id": blank_id,
        "dropout": args.dropout,
    }, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
