"""Custom learning rate schedulers for sleap-nn training.

This module provides learning rate schedulers with warmup phases that are commonly
used in deep learning for pose estimation and computer vision tasks.
"""

import math
from typing import List, Sequence, Union

from torch.optim.lr_scheduler import LRScheduler


def _per_group(
    value: Union[float, Sequence[float]], optimizer, name: str
) -> List[float]:
    """Broadcast a scalar LR setting to every param group (or check a per-group list).

    Args:
        value: One learning rate for every group, or one per param group.
        optimizer: The wrapped optimizer.
        name: Argument name, for error messages.

    Returns:
        One non-negative value per param group.
    """
    n = len(optimizer.param_groups)
    values = (
        [float(v) for v in value] if isinstance(value, Sequence) else [float(value)] * n
    )
    if len(values) != n:
        raise ValueError(
            f"{name} has {len(values)} values but the optimizer has {n} param groups"
        )
    if any(v < 0 for v in values):
        raise ValueError(f"{name} must be >= 0, got {value}")
    return values


def _warmup_alpha(step: int, warmup: int, skip_start: bool) -> float:
    """Fraction of the way from the warmup start LR to the base LR at ``step``.

    With ``skip_start`` the ramp leaves out its start point and runs
    ``1 / (warmup + 1), ..., warmup / (warmup + 1)``, so the first step does not
    train at ``warmup_start_lr`` itself. That is what the per-epoch schedules
    need: stepped once per epoch, the default ``warmup_start_lr=0`` otherwise
    trains the whole first epoch at a learning rate of 0.
    """
    if skip_start:
        return (step + 1) / (warmup + 1)
    return step / warmup


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler with linear warmup.

    The learning rate increases linearly from `warmup_start_lr` to the optimizer's
    base learning rate over `warmup_epochs`, then decreases following a cosine
    curve to `eta_min` over the remaining epochs.

    This schedule is widely used in vision transformers and modern CNN architectures
    as it provides stable early training (warmup) and smooth convergence (cosine decay).

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of epochs for the linear warmup phase.
        max_epochs: Total number of training epochs.
        warmup_start_lr: Learning rate at the start of warmup, or one per param
            group. Default: 0.0.
        eta_min: Minimum learning rate at the end of the schedule, or one per
            param group. Default: 0.0.
        last_epoch: The index of the last epoch. Default: -1.
        skip_warmup_start: Leave the start point out of the warmup ramp: step
            ``t < warmup_epochs`` uses ``(t + 1) / (warmup_epochs + 1)`` of the way
            from ``warmup_start_lr`` to the base learning rate instead of
            ``t / warmup_epochs``, so the first step does not train at
            ``warmup_start_lr`` (0 by default). Default: False.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = LinearWarmupCosineAnnealingLR(
        ...     optimizer, warmup_epochs=5, max_epochs=100, eta_min=1e-6
        ... )
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: Union[float, Sequence[float]] = 0.0,
        eta_min: Union[float, Sequence[float]] = 0.0,
        last_epoch: int = -1,
        skip_warmup_start: bool = False,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of epochs for the linear warmup phase.
            max_epochs: Total number of training epochs.
            warmup_start_lr: Learning rate at the start of warmup, or one per
                param group. Default: 0.0.
            eta_min: Minimum learning rate at the end of the schedule, or one per
                param group. Default: 0.0.
            last_epoch: The index of the last epoch. Default: -1.
            skip_warmup_start: Leave the start point out of the warmup ramp, so
                the first step trains above ``warmup_start_lr``. Default: False.
        """
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
        if warmup_epochs >= max_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be < max_epochs ({max_epochs})"
            )
        self._warmup_start_lrs = _per_group(
            warmup_start_lr, optimizer, "warmup_start_lr"
        )
        self._eta_mins = _per_group(eta_min, optimizer, "eta_min")

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.skip_warmup_start = skip_warmup_start
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate at the current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            if self.warmup_epochs == 0:
                return list(self.base_lrs)
            alpha = _warmup_alpha(
                self.last_epoch, self.warmup_epochs, self.skip_warmup_start
            )
            return [
                start + alpha * (base_lr - start)
                for start, base_lr in zip(self._warmup_start_lrs, self.base_lrs)
            ]
        else:
            # Cosine annealing phase
            decay_epochs = self.max_epochs - self.warmup_epochs
            if decay_epochs == 0:
                return list(self._eta_mins)
            progress = (self.last_epoch - self.warmup_epochs) / decay_epochs
            # Clamp progress to [0, 1] to handle epochs beyond max_epochs
            progress = min(1.0, progress)
            return [
                eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for eta_min, base_lr in zip(self._eta_mins, self.base_lrs)
            ]


class LinearWarmupLinearDecayLR(LRScheduler):
    """Linear warmup followed by linear decay learning rate scheduler.

    The learning rate increases linearly from `warmup_start_lr` to the optimizer's
    base learning rate over `warmup_epochs`, then decreases linearly to `end_lr`
    over the remaining epochs.

    This schedule provides a simple, interpretable learning rate trajectory and is
    commonly used in transformer-based models and NLP tasks.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of epochs for the linear warmup phase.
        max_epochs: Total number of training epochs.
        warmup_start_lr: Learning rate at the start of warmup, or one per param
            group. Default: 0.0.
        end_lr: Learning rate at the end of training, or one per param group.
            Default: 0.0.
        last_epoch: The index of the last epoch. Default: -1.
        skip_warmup_start: Leave the start point out of the warmup ramp: step
            ``t < warmup_epochs`` uses ``(t + 1) / (warmup_epochs + 1)`` of the way
            from ``warmup_start_lr`` to the base learning rate instead of
            ``t / warmup_epochs``, so the first step does not train at
            ``warmup_start_lr`` (0 by default). Default: False.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = LinearWarmupLinearDecayLR(
        ...     optimizer, warmup_epochs=5, max_epochs=100, end_lr=1e-6
        ... )
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: Union[float, Sequence[float]] = 0.0,
        end_lr: Union[float, Sequence[float]] = 0.0,
        last_epoch: int = -1,
        skip_warmup_start: bool = False,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of epochs for the linear warmup phase.
            max_epochs: Total number of training epochs.
            warmup_start_lr: Learning rate at the start of warmup, or one per
                param group. Default: 0.0.
            end_lr: Learning rate at the end of training, or one per param group.
                Default: 0.0.
            last_epoch: The index of the last epoch. Default: -1.
            skip_warmup_start: Leave the start point out of the warmup ramp, so
                the first step trains above ``warmup_start_lr``. Default: False.
        """
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
        if warmup_epochs >= max_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be < max_epochs ({max_epochs})"
            )
        self._warmup_start_lrs = _per_group(
            warmup_start_lr, optimizer, "warmup_start_lr"
        )
        self._end_lrs = _per_group(end_lr, optimizer, "end_lr")

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.skip_warmup_start = skip_warmup_start
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate at the current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            if self.warmup_epochs == 0:
                return list(self.base_lrs)
            alpha = _warmup_alpha(
                self.last_epoch, self.warmup_epochs, self.skip_warmup_start
            )
            return [
                start + alpha * (base_lr - start)
                for start, base_lr in zip(self._warmup_start_lrs, self.base_lrs)
            ]
        else:
            # Linear decay phase
            decay_epochs = self.max_epochs - self.warmup_epochs
            if decay_epochs == 0:
                return list(self._end_lrs)
            progress = (self.last_epoch - self.warmup_epochs) / decay_epochs
            # Clamp progress to [0, 1] to handle epochs beyond max_epochs
            progress = min(1.0, progress)
            return [
                base_lr + progress * (end_lr - base_lr)
                for end_lr, base_lr in zip(self._end_lrs, self.base_lrs)
            ]
