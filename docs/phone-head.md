# Frozen FastConformer phoneme CTC head

E05 trains only a small projection on top of cached, frozen Parakeet encoder states.

## Training cache contract

One target entity region per JSONL row:

```json
{"id":"utt-001-target-0","feature_path":"artifacts/encoder/utt-001-target-0.pt","phone_ids":[12,4,12,18,31]}
```

Each `feature_path` stores a `[T,D]` tensor or `{"encoder_states": tensor}` extracted from the same frozen `nvidia/parakeet-tdt_ctc-0.6b-ja` checkpoint used by the decoder. Reference/forced alignment may crop entity regions during training; do not require reference alignment at inference.

## Train

```bash
python scripts/train_phone_head.py \
  --manifest data/generated/phone_train.jsonl \
  --phone-vocab-size 64 \
  --output artifacts/phone_head.pt
```

## Apply

```bash
python scripts/rerank_phone.py \
  --input results/E04_ctc_rerank.jsonl \
  --output results/E05_phone_rerank.jsonl \
  --checkpoint artifacts/phone_head.pt \
  --feature-dir artifacts/encoder \
  --alpha 0.2
```

Report exact homophones and near-homophones separately. The phone head should help acoustically distinct near-homophones; identical phone strings need linguistic/entity context instead.
