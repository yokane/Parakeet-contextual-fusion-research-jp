from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from .phone_distance import (
    DEFAULT_PHONE_COSTS,
    SPECIAL_MORAS,
    VOICING_PAIRS,
    PhoneCosts,
    insertion_deletion_cost,
    relation_for_phones,
    substitution_cost,
    weighted_phone_distance,
)

__all__ = [
    "DEFAULT_PHONE_COSTS",
    "SPECIAL_MORAS",
    "VOICING_PAIRS",
    "PhoneCosts",
    "PhoneCTCHead",
    "insertion_deletion_cost",
    "relation_for_phones",
    "substitution_cost",
    "weighted_phone_distance",
]


if nn is not None:

    class PhoneCTCHead(nn.Module):
        """Small trainable CTC projection intended for a frozen FastConformer encoder."""

        def __init__(self, encoder_dim: int, phone_vocab_size: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.projection = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim, phone_vocab_size),
            )

        def forward(self, encoder_states: torch.Tensor) -> torch.Tensor:
            """Return log probabilities; accepts [..., encoder_dim]."""
            return self.projection(encoder_states).log_softmax(dim=-1)

else:

    class PhoneCTCHead:
        """Placeholder that reports the optional torch dependency only when instantiated."""

        def __init__(self, encoder_dim: int, phone_vocab_size: int, dropout: float = 0.1) -> None:
            del encoder_dim, phone_vocab_size, dropout
            raise ModuleNotFoundError(
                "PhoneCTCHead requires the GPU/torch dependency set; run `mise run deps:sync-gpu`."
            )
