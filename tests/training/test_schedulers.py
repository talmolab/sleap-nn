"""Tests for custom learning rate schedulers."""

import math

import pytest
import torch
from torch import nn

from sleap_nn.training.schedulers import (
    LinearWarmupCosineAnnealingLR,
    LinearWarmupLinearDecayLR,
)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 1)


@pytest.fixture
def optimizer(simple_model):
    """Create an optimizer for testing."""
    return torch.optim.Adam(simple_model.parameters(), lr=1e-3)


class TestLinearWarmupCosineAnnealingLR:
    """Tests for LinearWarmupCosineAnnealingLR scheduler."""

    def test_warmup_phase(self, optimizer):
        """Test that learning rate increases linearly during warmup."""
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=100,
            warmup_start_lr=0.0,
            eta_min=1e-6,
        )

        lrs = []
        for epoch in range(5):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should increase linearly from 0 to 1e-3 over 5 epochs
        expected_lrs = [0.0, 0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3]
        for actual, expected in zip(lrs, expected_lrs):
            assert abs(actual - expected) < 1e-9

    def test_cosine_phase(self, optimizer):
        """Test that learning rate follows cosine curve after warmup."""
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=15,
            warmup_start_lr=0.0,
            eta_min=0.0,
        )

        # Skip warmup phase
        for _ in range(5):
            scheduler.step()

        # Get LR at start of cosine phase (should be base_lr)
        lr_at_warmup_end = scheduler.get_last_lr()[0]
        assert abs(lr_at_warmup_end - 1e-3) < 1e-9

        # Step through cosine phase
        for _ in range(10):
            scheduler.step()

        # At end, LR should be eta_min (0.0)
        lr_at_end = scheduler.get_last_lr()[0]
        assert abs(lr_at_end - 0.0) < 1e-9

    def test_cosine_midpoint(self, optimizer):
        """Test LR at midpoint of cosine phase is half of base_lr."""
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=0,
            max_epochs=100,
            warmup_start_lr=0.0,
            eta_min=0.0,
        )

        # Step to midpoint
        for _ in range(50):
            scheduler.step()

        # At midpoint of cosine, LR should be base_lr / 2
        lr_at_midpoint = scheduler.get_last_lr()[0]
        assert abs(lr_at_midpoint - 0.5e-3) < 1e-9

    def test_eta_min(self, optimizer):
        """Test that LR doesn't go below eta_min."""
        eta_min = 1e-5
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20,
            warmup_start_lr=0.0,
            eta_min=eta_min,
        )

        # Step through all epochs
        for _ in range(25):  # Go past max_epochs
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        assert lr >= eta_min - 1e-10

    def test_validation_errors(self, optimizer):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="warmup_epochs must be >= 0"):
            LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=-1, max_epochs=100)

        with pytest.raises(ValueError, match="max_epochs must be > 0"):
            LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=0)

        with pytest.raises(ValueError, match="warmup_epochs.*must be < max_epochs"):
            LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=100, max_epochs=50)

        with pytest.raises(ValueError, match="warmup_start_lr must be >= 0"):
            LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=5, max_epochs=100, warmup_start_lr=-0.1
            )

        with pytest.raises(ValueError, match="eta_min must be >= 0"):
            LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=5, max_epochs=100, eta_min=-0.1
            )

    def test_no_warmup(self, optimizer):
        """Test scheduler with zero warmup epochs."""
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=0,
            max_epochs=10,
            warmup_start_lr=0.0,
            eta_min=0.0,
        )

        # First LR should be base_lr (no warmup)
        lr = scheduler.get_last_lr()[0]
        assert abs(lr - 1e-3) < 1e-9


class TestLinearWarmupLinearDecayLR:
    """Tests for LinearWarmupLinearDecayLR scheduler."""

    def test_warmup_phase(self, optimizer):
        """Test that learning rate increases linearly during warmup."""
        scheduler = LinearWarmupLinearDecayLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=100,
            warmup_start_lr=0.0,
            end_lr=1e-6,
        )

        lrs = []
        for epoch in range(5):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should increase linearly from 0 to 1e-3 over 5 epochs
        expected_lrs = [0.0, 0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3]
        for actual, expected in zip(lrs, expected_lrs):
            assert abs(actual - expected) < 1e-9

    def test_decay_phase(self, optimizer):
        """Test that learning rate decreases linearly after warmup."""
        scheduler = LinearWarmupLinearDecayLR(
            optimizer,
            warmup_epochs=0,
            max_epochs=10,
            warmup_start_lr=0.0,
            end_lr=0.0,
        )

        lrs = []
        for _ in range(11):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should decrease linearly from 1e-3 to 0 over 10 epochs
        for i, lr in enumerate(lrs[:11]):
            expected = 1e-3 * (1 - i / 10)
            assert abs(lr - expected) < 1e-9

    def test_end_lr(self, optimizer):
        """Test that LR reaches end_lr at max_epochs."""
        end_lr = 1e-5
        scheduler = LinearWarmupLinearDecayLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20,
            warmup_start_lr=0.0,
            end_lr=end_lr,
        )

        # Step to max_epochs
        for _ in range(20):
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        assert abs(lr - end_lr) < 1e-9

    def test_midpoint_decay(self, optimizer):
        """Test LR at midpoint of decay phase."""
        scheduler = LinearWarmupLinearDecayLR(
            optimizer,
            warmup_epochs=0,
            max_epochs=100,
            warmup_start_lr=0.0,
            end_lr=0.0,
        )

        # Step to midpoint
        for _ in range(50):
            scheduler.step()

        # At midpoint of linear decay, LR should be base_lr / 2
        lr = scheduler.get_last_lr()[0]
        assert abs(lr - 0.5e-3) < 1e-9

    def test_validation_errors(self, optimizer):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="warmup_epochs must be >= 0"):
            LinearWarmupLinearDecayLR(optimizer, warmup_epochs=-1, max_epochs=100)

        with pytest.raises(ValueError, match="max_epochs must be > 0"):
            LinearWarmupLinearDecayLR(optimizer, warmup_epochs=5, max_epochs=0)

        with pytest.raises(ValueError, match="warmup_epochs.*must be < max_epochs"):
            LinearWarmupLinearDecayLR(optimizer, warmup_epochs=100, max_epochs=50)

        with pytest.raises(ValueError, match="warmup_start_lr must be >= 0"):
            LinearWarmupLinearDecayLR(
                optimizer, warmup_epochs=5, max_epochs=100, warmup_start_lr=-0.1
            )

        with pytest.raises(ValueError, match="end_lr must be >= 0"):
            LinearWarmupLinearDecayLR(
                optimizer, warmup_epochs=5, max_epochs=100, end_lr=-0.1
            )

    def test_no_warmup(self, optimizer):
        """Test scheduler with zero warmup epochs."""
        scheduler = LinearWarmupLinearDecayLR(
            optimizer,
            warmup_epochs=0,
            max_epochs=10,
            warmup_start_lr=0.0,
            end_lr=0.0,
        )

        # First LR should be base_lr (no warmup)
        lr = scheduler.get_last_lr()[0]
        assert abs(lr - 1e-3) < 1e-9

    def test_multiple_param_groups(self):
        """Test scheduler works with multiple parameter groups."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
        optimizer = torch.optim.Adam(
            [
                {"params": model[0].parameters(), "lr": 1e-3},
                {"params": model[1].parameters(), "lr": 1e-4},
            ]
        )

        scheduler = LinearWarmupLinearDecayLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=100,
            warmup_start_lr=0.0,
            end_lr=0.0,
        )

        lrs = scheduler.get_last_lr()
        assert len(lrs) == 2
        assert abs(lrs[0] - 0.0) < 1e-9  # warmup_start_lr for group 1
        assert abs(lrs[1] - 0.0) < 1e-9  # warmup_start_lr for group 2

        # After warmup
        for _ in range(5):
            scheduler.step()

        lrs = scheduler.get_last_lr()
        assert abs(lrs[0] - 1e-3) < 1e-9  # base_lr for group 1
        assert abs(lrs[1] - 1e-4) < 1e-9  # base_lr for group 2


class TestSchedulerIntegration:
    """Integration tests for schedulers with training loop."""

    def test_cosine_full_training_loop(self, simple_model, optimizer):
        """Test cosine scheduler through a full training loop."""
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=2,
            max_epochs=10,
            warmup_start_lr=0.0,
            eta_min=1e-6,
        )

        lrs = []
        for epoch in range(10):
            # Simulate training step
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Verify LR trajectory
        assert lrs[0] < lrs[1]  # Warmup: increasing
        assert lrs[1] < lrs[2]  # Warmup: increasing
        assert lrs[2] > lrs[5]  # Cosine: decreasing
        assert lrs[5] > lrs[9]  # Cosine: decreasing

    def test_linear_full_training_loop(self, simple_model, optimizer):
        """Test linear scheduler through a full training loop."""
        scheduler = LinearWarmupLinearDecayLR(
            optimizer,
            warmup_epochs=2,
            max_epochs=10,
            warmup_start_lr=0.0,
            end_lr=1e-6,
        )

        lrs = []
        for epoch in range(10):
            # Simulate training step
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Verify LR trajectory
        assert lrs[0] < lrs[1]  # Warmup: increasing
        assert lrs[1] < lrs[2]  # Warmup: increasing
        assert lrs[2] > lrs[5]  # Decay: decreasing
        assert lrs[5] > lrs[9]  # Decay: decreasing
