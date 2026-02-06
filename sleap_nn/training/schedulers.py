"""Custom learning rate schedulers for sleap-nn training.

This module provides learning rate schedulers with warmup phases that are commonly
used in deep learning for pose estimation and computer vision tasks.
"""

import math
from torch.optim.lr_scheduler import LRScheduler


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
        warmup_start_lr: Learning rate at the start of warmup. Default: 0.0.
        eta_min: Minimum learning rate at the end of the schedule. Default: 0.0.
        last_epoch: The index of the last epoch. Default: -1.

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
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of epochs for the linear warmup phase.
            max_epochs: Total number of training epochs.
            warmup_start_lr: Learning rate at the start of warmup. Default: 0.0.
            eta_min: Minimum learning rate at the end of the schedule. Default: 0.0.
            last_epoch: The index of the last epoch. Default: -1.
        """
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
        if warmup_epochs >= max_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be < max_epochs ({max_epochs})"
            )
        if warmup_start_lr < 0:
            raise ValueError(f"warmup_start_lr must be >= 0, got {warmup_start_lr}")
        if eta_min < 0:
            raise ValueError(f"eta_min must be >= 0, got {eta_min}")

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate at the current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            if self.warmup_epochs == 0:
                return list(self.base_lrs)
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            decay_epochs = self.max_epochs - self.warmup_epochs
            if decay_epochs == 0:
                return [self.eta_min for _ in self.base_lrs]
            progress = (self.last_epoch - self.warmup_epochs) / decay_epochs
            # Clamp progress to [0, 1] to handle epochs beyond max_epochs
            progress = min(1.0, progress)
            return [
                self.eta_min
                + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
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
        warmup_start_lr: Learning rate at the start of warmup. Default: 0.0.
        end_lr: Learning rate at the end of training. Default: 0.0.
        last_epoch: The index of the last epoch. Default: -1.

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
        warmup_start_lr: float = 0.0,
        end_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of epochs for the linear warmup phase.
            max_epochs: Total number of training epochs.
            warmup_start_lr: Learning rate at the start of warmup. Default: 0.0.
            end_lr: Learning rate at the end of training. Default: 0.0.
            last_epoch: The index of the last epoch. Default: -1.
        """
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
        if warmup_epochs >= max_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be < max_epochs ({max_epochs})"
            )
        if warmup_start_lr < 0:
            raise ValueError(f"warmup_start_lr must be >= 0, got {warmup_start_lr}")
        if end_lr < 0:
            raise ValueError(f"end_lr must be >= 0, got {end_lr}")

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate at the current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            if self.warmup_epochs == 0:
                return list(self.base_lrs)
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay phase
            decay_epochs = self.max_epochs - self.warmup_epochs
            if decay_epochs == 0:
                return [self.end_lr for _ in self.base_lrs]
            progress = (self.last_epoch - self.warmup_epochs) / decay_epochs
            # Clamp progress to [0, 1] to handle epochs beyond max_epochs
            progress = min(1.0, progress)
            return [
                base_lr + progress * (self.end_lr - base_lr)
                for base_lr in self.base_lrs
            ]
