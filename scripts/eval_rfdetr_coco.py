r"""COCO val2017 box AP of an imported RF-DETR model, run through ``sleap-nn predict``.

The bbox_detr M1 check: import ``Roboflow/rf-detr-<size>`` (``sleap-nn
import-rfdetr``), predict COCO val2017 with ``sleap-nn predict`` (squash resize,
every detection kept: score threshold 0, RF-DETR's 300 per image) and score the
boxes with pycocotools:

* all-class box AP (RF-DETR-M reports 54.7);
* person-only box AP: person detections only, ``catIds=[1]`` (the protocol of
  the human project's ``coco_person_ap.py``).

Category ids: the imported head's class index is the COCO category id.

Usage (``--limit`` evaluates the first N labeled frames only, e.g. a CPU smoke)::

    CUDA_VISIBLE_DEVICES= uv run python scripts/eval_rfdetr_coco.py \
        --labels coco_val2017.slp --annotations instances_val2017.json \
        --model_dir models/rfdetr-medium --size medium --out_dir eval/rfdetr-m \
        --device cpu --limit 20

Needs ``pycocotools`` (in the dev dependency group).
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def import_model(size: str, model_dir: Path) -> None:
    """Import ``Roboflow/rf-detr-<size>`` into ``model_dir`` unless it exists."""
    if (model_dir / "best.ckpt").exists():
        print(f"Using the imported model in {model_dir}.")
        return
    from sleap_nn.convert.rfdetr import import_rfdetr

    import_rfdetr(size, model_dir)


def subset_labels(labels_path: Path, limit: int, out_path: Path) -> Path:
    """Write the first ``limit`` labeled frames of ``labels_path`` to ``out_path``."""
    import sleap_io as sio

    labels = sio.load_slp(labels_path.as_posix())
    frames = list(labels.labeled_frames[:limit])
    videos = []
    for lf in frames:
        if not any(lf.video is v for v in videos):
            videos.append(lf.video)
    sio.Labels(labeled_frames=frames, videos=videos, skeletons=labels.skeletons).save(
        out_path.as_posix()
    )
    return out_path


def run_predict(
    model_dir: Path, labels: Path, output: Path, device: str, batch_size: int
) -> None:
    """Run ``sleap-nn predict`` keeping every detection (as RF-DETR's eval does)."""
    cmd = [
        "sleap-nn",
        "predict",
        "-m",
        model_dir.as_posix(),
        "-i",
        labels.as_posix(),
        "-o",
        output.as_posix(),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--peak_threshold",
        "0",
        "--max_instances",
        "300",
    ]
    print("$ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def frame_image_path(lf) -> str:
    """The image file of a labeled frame of an image-sequence video."""
    filename = lf.video.filename
    return filename[lf.frame_idx] if isinstance(filename, list) else filename


def coco_results(predictions: Path, model_dir: Path):
    """``(results, image_ids)``: COCO results JSON entries and the predicted images."""
    import sleap_io as sio
    from omegaconf import OmegaConf

    cfg = OmegaConf.load((model_dir / "training_config.yaml").as_posix())
    categories = list(cfg.model_config.head_configs.bbox_detr.detr.categories)
    labels = sio.load_slp(predictions.as_posix())
    results, image_ids = [], []
    for lf in labels.labeled_frames:
        image_id = int(Path(frame_image_path(lf)).stem)
        image_ids.append(image_id)
        for box in lf.bboxes:
            if not isinstance(box, sio.PredictedBoundingBox):
                continue
            name = getattr(box.category, "name", box.category)
            results.append(
                {
                    "image_id": image_id,
                    "category_id": categories.index(name),
                    "bbox": [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1],
                    "score": float(box.score),
                }
            )
    return results, sorted(set(image_ids))


def coco_box_ap(annotations: Path, results, image_ids, person_only: bool) -> dict:
    """Box AP (pycocotools) over ``image_ids``, all classes or person only."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(annotations.as_posix())
    if person_only:
        results = [r for r in results if r["category_id"] == 1]
    # Detections of category ids COCO does not have (the unused N/A slots) are
    # dropped: loadRes would accept them, but they can never match.
    results = [r for r in results if r["category_id"] in gt.cats]
    if not results:
        return {"n_dets": 0}
    dt = gt.loadRes(results)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = image_ids
    if person_only:
        ev.params.catIds = [1]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100"]
    return {"n_dets": len(results), **dict(zip(names, ev.stats[:9].round(4).tolist()))}


def main(argv=None):
    """Import, predict and score."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True, help="COCO val2017 .slp")
    ap.add_argument(
        "--annotations", type=Path, required=True, help="instances_val2017.json"
    )
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument(
        "--size",
        default="medium",
        help="RF-DETR size to import when --model_dir has no model.",
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--limit", type=int, default=None, help="Only the first N labeled frames."
    )
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import_model(args.size, args.model_dir)
    labels = args.labels
    if args.limit is not None:
        labels = subset_labels(
            labels, args.limit, args.out_dir / f"labels_first{args.limit}.slp"
        )
    predictions = args.out_dir / "predictions.slp"
    run_predict(args.model_dir, labels, predictions, args.device, args.batch_size)

    results, image_ids = coco_results(predictions, args.model_dir)
    (args.out_dir / "coco_results.json").write_text(json.dumps(results))
    summary = {
        "model_dir": args.model_dir.as_posix(),
        "n_images": len(image_ids),
        "all_classes": coco_box_ap(args.annotations, results, image_ids, False),
        "person": coco_box_ap(args.annotations, results, image_ids, True),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
