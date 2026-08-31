# E06: NeMo TDT in-beam fusion contract

E06 is intentionally version-isolated. Do not vendor or silently monkey-patch an arbitrary NeMo nightly into the core benchmark package.

## Promotion criterion

Implement an in-beam driver only after E04/E05 show a reproducible N-best gain on held-out data. If N-best reranking does not improve the target metric, moving the same scorer into the beam adds complexity without evidence of value.

## Driver CLI

`experiments/E06_inbeam.sh` calls a user-supplied driver through `$E06_DRIVER` with:

```text
--manifest
--model
--beam-size
--ngram-lm-model
--ngram-lm-alpha
--context-phrases
--boosting-tree-alpha
--ctc-alpha
--phone-alpha
--output
```

## Required decoder behavior

Retain the normal TDT/NGPU-LM/GPU-PB state and add a small context state:

```python
@dataclass
class ContextState:
    phrase_tree_state: object
    entity_start_frame: int | None
    active_entity_ids: tuple[int, ...]
```

Use phrase boosting as the gate for expensive scoring:

1. Ordinary token: TDT + NGPU-LM + GPU-PB only.
2. Enter a context-phrase prefix: remember the encoder frame and active entity IDs.
3. Complete an entity: compute local CTC and optional phone score for only the small candidate set.
4. Add normalized scorer values before the next beam prune.

Target score:

```text
S = S_tdt
  + alpha * S_lm
  + beta * S_pb
  + I_entity * (gamma * S_ctc + delta * S_phone)
```

## Version-specific patches

When a NeMo commit is pinned, create:

```text
patches/nemo-<short-sha>/
├── README.md
├── inbeam_driver.py
└── nemo.patch
```

Record the exact NeMo commit SHA in every E06 result. Never invoke the neural phone/entity scorer for every vocabulary token and frame; candidate gating is part of the experiment hypothesis and runtime must be reported alongside accuracy.
