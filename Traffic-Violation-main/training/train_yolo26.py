"""Train the YOLO26 comparison baseline (Ultralytics) on the same data as Option B.

  python training/train_yolo26.py --task helmet_plate
  python training/train_yolo26.py --task triple_riding --weights yolo26n.pt

Outputs go to training/runs/yolo26/<task>/ (weights/best.pt, results.csv).
YOLO26 is a comparison only, not the main model; see evaluate.py for head-to-head scoring.
"""
import argparse
import sys

import torch
from tqdm import tqdm
from ultralytics import YOLO

from config import DEFAULTS, RUNS_DIR, TASKS, YOLO_WEIGHTS, pick_device


def add_progress_bars(model):
    """One bar per epoch for the training batches, matching the bars in train_option_b.py."""
    bar = {}

    def on_epoch_start(trainer):
        bar["train"] = tqdm(total=len(trainer.train_loader), desc=f"epoch {trainer.epoch + 1}/{trainer.epochs} train",
                            leave=False, file=sys.stdout, mininterval=5)

    def on_batch_end(trainer):
        bar["train"].update(1)
        loss = getattr(trainer, "tloss", None)  # running loss; a dict in some Ultralytics versions, a tensor in others
        if loss is not None:
            values = loss.values() if isinstance(loss, dict) else [loss]
            bar["train"].set_postfix(loss=f"{sum(float(torch.as_tensor(v).sum()) for v in values):.3f}")

    def on_epoch_end(trainer):
        bar["train"].close()

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_train_epoch_end", on_epoch_end)
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--weights", default=YOLO_WEIGHTS, help="yolo26s.pt for helmet_plate, yolo26n.pt is fine for triple_riding")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--device", default="", help="'' auto, '0' first GPU, or 'cpu'")
    p.add_argument("--data", help="override data.yaml (tiny subsets for smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="continue from runs/yolo26/<task>/weights/last.pt after a dropped session")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    run_dir = RUNS_DIR / "yolo26" / args.task
    last = run_dir / "weights" / "last.pt"
    if args.resume:
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} not found")
        # Finished runs have their optimizer stripped; Ultralytics would silently start a new run from COCO instead.
        if torch.load(last, map_location="cpu", weights_only=False).get("optimizer") is None:
            print(f"{last} is from a finished run; nothing to resume.")
        else:
            # Ultralytics restores weights, optimizer, scheduler and epoch from last.pt; other flags are ignored.
            add_progress_bars(YOLO(str(last))).train(resume=True)
        print(f"best weights: {run_dir / 'weights' / 'best.pt'}")
        return
    if last.exists():
        raise SystemExit(f"{last} exists. Pass --resume to continue it, or remove the old run folder.")
    model = add_progress_bars(YOLO(args.weights))
    model.train(
        data=args.data or str(TASKS[args.task]["data"]),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        device=0 if device.type == "cuda" else "cpu",
        project=str(RUNS_DIR / "yolo26"),
        name=args.task,
        exist_ok=True,
    )
    print(f"best weights: {RUNS_DIR / 'yolo26' / args.task / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
