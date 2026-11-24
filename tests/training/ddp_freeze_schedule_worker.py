"""2-process DDP (gloo, CPU) run of `EncoderFreezeSchedule`.

Run by ``test_weight_transfer.py::test_freeze_schedule_under_ddp`` as a script:
Lightning's DDP launcher re-runs this file once per rank, which is why it is not
a test module itself.

Usage: ``python ddp_freeze_schedule_worker.py OUT_DIR STRATEGY``. Rank 0 writes
``OUT_DIR/result.pt`` with the encoder weights of every rank after training.
"""

import sys
from pathlib import Path

import lightning as L
import torch
from torch import nn

from sleap_nn.architectures.common import FreezableEncoderMixin
from sleap_nn.training.callbacks import EncoderFreezeSchedule


class _Backbone(FreezableEncoderMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4))
        self.dec = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x):
        return self.dec(torch.relu(self.enc(x)))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.head_layers = nn.ModuleList([nn.Conv2d(4, 2, 1)])

    def forward(self, x):
        return self.head_layers[0](self.backbone(x))


class _Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = _Model()
        self.enc_snapshots = {}

    def training_step(self, batch, batch_idx):
        (x,) = batch
        loss = self.model(x).pow(2).mean()
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        w = self.model.backbone.enc[0].weight.detach().clone()
        self.enc_snapshots[self.global_step] = w

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.1)


def main(out_dir: Path, strategy: str):
    L.seed_everything(0)
    data = torch.utils.data.TensorDataset(torch.randn(16, 1, 8, 8))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    module = _Module()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=strategy,
        max_steps=4,
        callbacks=[EncoderFreezeSchedule(targets=["encoder"], unfreeze_at_step=2)],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(module, loader)
    w = module.model.backbone.enc[0].weight.detach().clone()
    gathered = trainer.strategy.all_gather(w)
    if trainer.global_rank == 0:
        torch.save(
            {
                "snapshots": {k: v for k, v in module.enc_snapshots.items()},
                "final_per_rank": gathered,
                "find_unused": trainer.strategy._ddp_kwargs.get(
                    "find_unused_parameters"
                ),
            },
            out_dir / "result.pt",
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
