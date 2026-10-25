"""Evaluation: one metric, one data path, used for both Option B and YOLO26.

Both models see the same letterboxed 640px tensors and are scored against the same
ground truth with the same mAP code, so the comparison is apples to apples.

CLI:
  python training/evaluate.py --task helmet_plate --split test \
      --option-b training/runs/helmet_plate_optionb/weights/best.pt \
      --yolo runs/yolo26/helmet_plate/weights/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from config import DEFAULTS, RUNS_DIR, TASKS, pick_device
from option_b import OptionBDetector, postprocess
from yolo_data import YoloDetectionDataset, collate, load_data_yaml


@torch.no_grad()
def evaluate(predict_fn, loader, device, desc=None):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.warn_on_many_detections = False  # YOLO can output up to 300 boxes; the metric keeps the top 100 by score anyway
    batches = tqdm(loader, desc=desc, leave=False, file=sys.stdout, mininterval=5) if desc else loader
    for imgs, targets in batches:
        preds = predict_fn(imgs.to(device))
        preds = [
            {"boxes": p["boxes"].float().cpu(), "scores": p["scores"].float().cpu(), "labels": p["labels"].long().cpu()}
            for p in preds
        ]
        gts = [{"boxes": t[:, 1:].cpu(), "labels": t[:, 0].long().cpu()} for t in targets]
        metric.update(preds, gts)
    r = metric.compute()
    return {
        "mAP50": round(float(r["map_50"]), 4),
        "mAP50_95": round(float(r["map"]), 4),
        "recall_100": round(float(r["mar_100"]), 4),
    }


def option_b_predictor(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model = OptionBDetector(len(ckpt["names"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    imgsz = ckpt["imgsz"]

    def predict(imgs):
        return postprocess(model(imgs), imgsz)

    return predict


def yolo_predictor(weights, device):
    from ultralytics import YOLO

    yolo = YOLO(weights)

    def predict(imgs):
        res = yolo.predict(imgs, conf=0.001, iou=0.6, max_det=300, device=device, verbose=False)
        return [{"boxes": r.boxes.xyxy, "scores": r.boxes.conf, "labels": r.boxes.cls} for r in res]

    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--option-b", help="Option B checkpoint (.pt) written by train_option_b.py")
    p.add_argument("--yolo", help="YOLO26 weights (.pt)")
    p.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    p.add_argument("--device", default="")
    p.add_argument("--out", help="Optional JSON file to write results to")
    args = p.parse_args()

    if not (args.option_b or args.yolo):
        p.error("give --option-b and/or --yolo")

    device = pick_device(args.device)
    _, splits = load_data_yaml(TASKS[args.task]["data"])
    ds = YoloDetectionDataset(splits[args.split], args.imgsz, augment=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)

    results = {"task": args.task, "split": args.split, "images": len(ds), "device": str(device)}
    if args.option_b:
        results["option_b"] = evaluate(option_b_predictor(args.option_b, device), loader, device)
    if args.yolo:
        results["yolo26"] = evaluate(yolo_predictor(args.yolo, device), loader, device)

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
