from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FrameWindow:
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def expand_window(start: int, end: int, *, margin: int, total_frames: int) -> FrameWindow:
    return FrameWindow(max(0, start - margin), min(total_frames, end + margin))


def ctc_sequence_logprob(
    log_probs: torch.Tensor,
    token_ids: list[int] | torch.Tensor,
    *,
    blank_id: int,
    length_norm_power: float = 1.0,
) -> torch.Tensor:
    if log_probs.ndim != 2:
        raise ValueError(f"expected [T,V] log_probs, got {tuple(log_probs.shape)}")
    targets = torch.as_tensor(token_ids, dtype=torch.long, device=log_probs.device)
    if targets.numel() == 0 or log_probs.shape[0] < targets.numel():
        return log_probs.new_tensor(float("-inf"))
    nll = F.ctc_loss(
        log_probs.unsqueeze(1),
        targets,
        input_lengths=torch.tensor([log_probs.shape[0]], device=log_probs.device),
        target_lengths=torch.tensor([targets.numel()], device=log_probs.device),
        blank=blank_id,
        reduction="sum",
        zero_infinity=True,
    )
    denom = float(targets.numel()) ** length_norm_power
    return -nll / denom


def local_ctc_score(
    full_log_probs: torch.Tensor,
    token_ids: list[int] | torch.Tensor,
    *,
    window: FrameWindow,
    blank_id: int,
    length_norm_power: float = 1.0,
) -> torch.Tensor:
    if window.length <= 0:
        return full_log_probs.new_tensor(float("-inf"))
    return ctc_sequence_logprob(
        full_log_probs[window.start : window.end],
        token_ids,
        blank_id=blank_id,
        length_norm_power=length_norm_power,
    )
