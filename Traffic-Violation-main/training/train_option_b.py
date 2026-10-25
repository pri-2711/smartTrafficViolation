
"""Train Option B (our own detector) on one task.

  python training/train_option_b.py --task helmet_plate
  python training/train_option_b.py --task triple_riding --epochs 60 --patience 15

Outputs go to training/runs/<task>_optionb/: weights/best.pt, weights/last.pt,
log.csv (per-epoch losses and validation mAP), results.json (final test scores).
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEFAULTS, RUNS_DIR, TASKS, pick_device
from evaluate import evaluate
from option_b import OptionBDetector, detection_loss, postprocess, scale_ranges_for
from option_b.grids import grids_for
from yolo_data import YoloDetectionDataset, collate, load_data_yaml


def predict(model, imgs, imgsz):
    return postprocess(model(imgs), imgsz)


def save_checkpoint(state, path):
    """Write to a temp file, then swap in. A session dropped mid-write leaves the previous checkpoint intact."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def lr_factor(epoch, epochs):
    """Cosine decay from 1.0 down to 5% of the base learning rate."""
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * epoch / max(epochs, 1)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--name", help="run name (default: <task>_optionb)")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--freeze-epochs", type=int, default=DEFAULTS["freeze_epochs"], help="backbone frozen for the first N epochs")
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--device", default="", help="'' auto, '0' first GPU, or 'cpu'")
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline smoke tests)")
    p.add_argument("--limit", type=int, help="use only the first N images per split (smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="continue from runs/<name>/weights/last.pt after a dropped session (required if last.pt exists)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    use_amp = device.type == "cuda"
    name = args.name or f"{args.task}_optionb"
    run_dir = RUNS_DIR / name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)

    names, splits = load_data_yaml(TASKS[args.task]["data"])
    nc = len(names)
    train_ds = YoloDetectionDataset(splits["train"], args.imgsz, augment=True, limit=args.limit)
    val_ds = YoloDetectionDataset(splits["val"], args.imgsz, augment=False, limit=args.limit)
    dl_kw = dict(num_workers=args.workers, collate_fn=collate, pin_memory=use_amp, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **dl_kw)

    model = OptionBDetector(nc, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    scale_ranges = scale_ranges_for(args.imgsz)
    warmup_iters = min(500, len(train_dl))

    print(f"[{name}] device={device} classes={names} train={len(train_ds)} val={len(val_ds)} amp={use_amp}")

    last_path = run_dir / "weights" / "last.pt"
    best_fitness, best_epoch, step, start_epoch = -1.0, -1, 0, 0
    log_path = run_dir / "log.csv"
    if last_path.exists():
        if not args.resume:
            raise SystemExit(f"{last_path} exists. Pass --resume to continue it, or use a new --name.")
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step = state["epoch"] + 1, state["step"]
        best_fitness, best_epoch = state["best_fitness"], state["best_epoch"]
        if state.get("finished"):  # the earlier run already stopped (early stop or last epoch)
            start_epoch = args.epochs
        print(f"resumed from epoch {start_epoch} (best so far: epoch {best_epoch + 1}, fitness {best_fitness:.4f})")
        if start_epoch >= args.epochs:
            print("training already complete; skipping to final test evaluation")

    # Append when resuming so the log keeps the epochs from the dropped session.
    log_f = open(log_path, "a" if start_epoch > 0 else "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if start_epoch == 0:
        log.writerow(["epoch", "lr", "loss", "cls", "obj", "box", "val_mAP50", "val_mAP50_95", "seconds"])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        frozen = epoch < args.freeze_epochs
        model.freeze_backbone(frozen)
        model.train()
        if frozen:
            model.backbone.eval()  # keep BN statistics of the pretrained backbone fixed

        base_lr = args.lr * lr_factor(epoch, args.epochs)
        sums = {"loss": 0.0, "cls": 0.0, "obj": 0.0, "box": 0.0}
        # Progress bar: shows the batch rate and running loss, so a slow epoch is visibly moving.
        bar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{args.epochs} train", leave=False, file=sys.stdout, mininterval=5)
        for imgs, targets in bar:
            warm = min(1.0, (step + 1) / warmup_iters)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * warm
            imgs = imgs.to(device, non_blocking=True)

            with torch.autocast(device.type, enabled=use_amp):
                outputs = model(imgs)
            outputs = [(c.float(), r.float(), o.float()) for c, r, o in outputs]
            grids = grids_for(outputs)
            loss, parts = detection_loss(outputs, targets, grids, nc, scale_ranges)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            sums["loss"] += loss.item()
            for k in ("cls", "obj", "box"):
                sums[k] += parts[k]
            step += 1
            bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{base_lr * warm:.1e}")

        n = max(len(train_dl), 1)
        model.eval()
        val = evaluate(lambda x: predict(model, x, args.imgsz), val_dl, device,
                       desc=f"epoch {epoch + 1}/{args.epochs} val")
        val_map50, val_map = val["mAP50"], val["mAP50_95"]
        fitness = 0.1 * val_map50 + 0.9 * val_map  # same weighting Ultralytics uses for best.pt
        seconds = round(time.time() - t0, 1)

        log.writerow([epoch, round(base_lr, 6), round(sums["loss"] / n, 4), round(sums["cls"] / n, 4),
                      round(sums["obj"] / n, 4), round(sums["box"] / n, 4), val_map50, val_map, seconds])
        log_f.flush()
        print(f"epoch {epoch + 1}/{args.epochs} loss={sums['loss'] / n:.3f} mAP50={val_map50} mAP50-95={val_map} "
              f"frozen={frozen} {seconds}s")

        if fitness > best_fitness:
            best_fitness, best_epoch = fitness, epoch
        # last.pt is the resume point (full training state); best.pt is the model we report.
        stop = epoch >= args.freeze_epochs and epoch - best_epoch >= args.patience
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "names": names, "imgsz": args.imgsz, "epoch": epoch, "step": step, "val": val,
                 "best_fitness": best_fitness, "best_epoch": best_epoch,
                 "finished": stop or epoch + 1 >= args.epochs}
        save_checkpoint(state, last_path)
        if best_epoch == epoch:
            save_checkpoint(state, run_dir / "weights" / "best.pt")
        if stop:
            print(f"early stop: no validation improvement for {args.patience} epochs")
            break

    log_f.close()

    # Final, unbiased score: reload the best checkpoint and score the held-out test split once.
    best = torch.load(run_dir / "weights" / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    model.eval()
    test_ds = YoloDetectionDataset(splits["test"], args.imgsz, augment=False, limit=args.limit)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, **dl_kw)
    test = evaluate(lambda x: predict(model, x, args.imgsz), test_dl, device)
    results = {"task": args.task, "best_epoch": best["epoch"] + 1, "val": best["val"], "test": test,
               "test_images": len(test_ds)}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
