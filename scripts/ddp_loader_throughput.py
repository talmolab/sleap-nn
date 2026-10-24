"""Measure multi-rank train-loader throughput and CPU use, without a model or GPU.

Starts ``--ranks`` processes that each build the train dataloader exactly as one DDP
rank of ``sleap-nn train`` would (its ``DistributedSampler`` shard, its rank-seeded
workers, ``--workers`` workers each), pulls ``--warmup`` batches, then times
``--batches`` batches with every rank in lockstep. The timed window is bracketed by
barriers, so the aggregate is what ``--ranks`` GPUs could at most be fed when the
loader is the bottleneck. CPU use is each rank's CPU time (its main process and
workers) over its timed window; ``--batches`` should be large next to each worker's
prefetch queue, which drains at the start of the window.

Not a test: run it on the training machine, e.g.

    uv run python scripts/ddp_loader_throughput.py config.yaml --ranks 4 --workers 8
    uv run python scripts/ddp_loader_throughput.py config.yaml --workers 4 8 16
    uv run python scripts/ddp_loader_throughput.py config.yaml --set KEY=VALUE

Each ``--workers`` value is a separate run; one line per run is printed, plus a JSON
record per run with ``--json``.
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import psutil
import torch.multiprocessing as mp


def _tree_cpu_seconds(proc: psutil.Process) -> float:
    """User + system CPU seconds of `proc` and its descendants (its workers)."""
    total = 0.0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.NoSuchProcess:
            pass
    return total


def _rank_main(rank, args, num_workers, barriers, results):
    """One rank: build its loader, warm up, then time `args.batches` batches."""
    import multiprocessing

    # This rank was started with `spawn`, which its dataloader would inherit; start
    # the workers the way a training rank does (the platform default).
    multiprocessing.set_start_method(None, force=True)

    from omegaconf import OmegaConf

    from sleap_nn.data.custom_datasets import get_train_val_dataloaders
    from sleap_nn.training.model_trainer import ModelTrainer

    built, start, stop = barriers
    config = OmegaConf.load(args.config)
    for item in args.set:
        key, value = item.split("=", 1)
        OmegaConf.update(config, key, OmegaConf.create({"v": value}).v, merge=False)
    OmegaConf.update(
        config, "trainer_config.train_data_loader.num_workers", num_workers
    )
    OmegaConf.update(config, "trainer_config.trainer_devices", args.ranks)

    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    # `_setup_datasets` only reads the rank and device count off the trainer.
    model_trainer.trainer = SimpleNamespace(
        global_rank=rank, num_devices=args.ranks, world_size=args.ranks
    )
    train_dataset, val_dataset = model_trainer._setup_datasets()
    train_loader, _ = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
        rank=rank,
        train_steps_per_epoch=args.ranks * (args.warmup + args.batches),
        trainer_devices=args.ranks,
    )
    batches = iter(train_loader)

    def n_crops(batch):
        for key in ("instance_image", "image"):
            if key in batch:
                return int(batch[key].shape[0])
        return int(next(iter(batch.values())).shape[0])

    # Warm up together, so no rank's workers bank batches while waiting for the
    # slowest rank's setup (a loader drained faster than it fills stays empty).
    built.wait()
    for _ in range(args.warmup):
        next(batches)
    start.wait()
    me = psutil.Process()
    cpu0, t0 = _tree_cpu_seconds(me), time.perf_counter()
    crops = sum(n_crops(next(batches)) for _ in range(args.batches))
    elapsed = time.perf_counter() - t0
    cpu = _tree_cpu_seconds(me) - cpu0
    stop.wait()
    results.put({"rank": rank, "crops": crops, "seconds": elapsed, "cpu": cpu})


def run(args, num_workers: int) -> dict:
    """Time one configuration; returns the aggregate record."""
    ctx = mp.get_context("spawn")
    barriers = tuple(ctx.Barrier(args.ranks + 1) for _ in range(3))
    results = ctx.Queue()
    procs = [
        ctx.Process(
            target=_rank_main, args=(rank, args, num_workers, barriers, results)
        )
        for rank in range(args.ranks)
    ]
    for p in procs:
        p.start()
    built, start, stop = barriers
    built.wait()
    start.wait()
    psutil.cpu_percent()  # start the machine-wide counter
    stop.wait()
    machine_pct = psutil.cpu_percent()
    per_rank = sorted((results.get() for _ in procs), key=lambda r: r["rank"])
    for p in procs:
        p.join()
    rates = [r["crops"] / r["seconds"] for r in per_rank]
    return {
        "ranks": args.ranks,
        "workers_per_rank": num_workers,
        "batches_per_rank": args.batches,
        "crops": sum(r["crops"] for r in per_rank),
        "crops_per_s": round(sum(rates), 1),
        "per_rank_crops_per_s": [round(x, 1) for x in rates],
        # Each rank's main process + its workers, over that rank's timed window.
        "cpu_cores_used": round(sum(r["cpu"] / r["seconds"] for r in per_rank), 2),
        "cpu_s_per_1k_crops": round(
            1000 * sum(r["cpu"] for r in per_rank) / sum(r["crops"] for r in per_rank),
            2,
        ),
        "machine_cpu_pct": machine_pct,
        "cpu_count": os.cpu_count(),
    }


def main():
    """Parse arguments and run each worker count."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="Training config (YAML).")
    parser.add_argument("--ranks", type=int, default=4, help="Simulated DDP ranks.")
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[8], help="Workers per rank."
    )
    parser.add_argument("--batches", type=int, default=100, help="Timed batches/rank.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed batches/rank.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override (dotted key), repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON records.")
    args = parser.parse_args()

    for num_workers in args.workers:
        rec = run(args, num_workers)
        print(
            f"{rec['ranks']} ranks x {rec['workers_per_rank']} workers: "
            f"{rec['crops_per_s']:.0f} crops/s aggregate "
            f"(per rank {rec['per_rank_crops_per_s']}), "
            f"{rec['cpu_cores_used']:.1f} cores of {rec['cpu_count']}, "
            f"{rec['cpu_s_per_1k_crops']:.2f} CPU-s per 1k crops "
            f"(machine {rec['machine_cpu_pct']:.0f}%)",
            flush=True,
        )
        if args.json:
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
