"""Draw both models' detections on test images, so the boxes can be checked by eye.

  python training/show_predictions.py --task helmet_plate
  python training/show_predictions.py --task triple_riding --n 20 --conf 0.3

--require keeps only images whose labels contain every listed class (e.g. number-plate). Each run writes to
its own tag folder, so different selections do not mix:

  python training/show_predictions.py --task helmet_plate --require number-plate --tag plates --n 20
  python training/show_predictions.py --task helmet_plate --require bike,no-helmet --tag no_helmet_on_bike --n 20

--pick skips the test-split sample entirely: it opens a native file-picker dialog so you can choose any
single image (from the dataset or anywhere else on disk), runs both models on just that image, and opens
the result:

  python training/show_predictions.py --task helmet_plate --pick
  python training/show_predictions.py --task triple_riding --pick --conf 0.3

For each image, one PNG with Option B on the left and YOLO26 on the right. Ground truth is drawn in green
on both sides when a label file exists for the image; predictions are drawn in colour with their class and
score. Output: analysis/predictions/<task>/. Without --pick, images are sampled evenly across the test
split, so the sample covers the whole set.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from config import ANALYSIS_DIR, DEFAULTS, RUNS_DIR, TASKS, pick_device
from evaluate import option_b_predictor, yolo_predictor
from yolo_data import PAD_VALUE, YoloDetectionDataset, load_data_yaml

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180)]
GROUND_TRUTH = (0, 255, 0)
VIOLATION_COLOR = (200, 0, 0)
OK_COLOR = (0, 130, 0)

# task -> {class name whose presence means a violation: message to show}
VIOLATION_RULES = {
    "helmet_plate": {"no-helmet": "Helmet violation found"},
    "triple_riding": {"tripleriding": "Triple riding violation found"},
}


def violation_text(task, labels, names):
    """Violation message for one model's detections on one image, or an all-clear message."""
    detected = {names[int(lab)] for lab in labels}
    for cls_name, message in VIOLATION_RULES.get(task, {}).items():
        if cls_name in detected:
            return message, True
    return "No violation found", False


def tensor_to_image(t):
    arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def draw(img, boxes, labels, scores, names, color_of, width=2):
    """Draw boxes on a copy of img. boxes are (x1, y1, x2, y2) in the same 640 px coordinates as the model input."""
    out = img.copy()
    draw_ = ImageDraw.Draw(out)
    for box, lab, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        color = color_of(int(lab))
        draw_.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = f"{names[int(lab)]} {score:.2f}" if score is not None else names[int(lab)]
        draw_.text((x1 + 2, max(0, y1 - 12)), text, fill=color)
    return out


def ground_truth_boxes(target):
    """target rows are (cls, x1, y1, x2, y2) in model input pixels."""
    return target[:, 1:].numpy(), target[:, 0].long().numpy()


def classes_in(img_path):
    """Set of class ids in an image's label file (empty for negative images)."""
    label = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if not label.exists():
        return set()
    return {int(float(line.split()[0])) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()}


def run(predict, tensor, conf):
    """Model output for one image, keeping only detections above conf."""
    det = predict(tensor.unsqueeze(0))[0]
    keep = det["scores"].cpu().numpy() >= conf
    return (det["boxes"].cpu().numpy()[keep], det["labels"].cpu().numpy()[keep], det["scores"].cpu().numpy()[keep])


def pick_image_file():
    """Open a native file-picker dialog and return the chosen path, or None if cancelled."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an image to run the models on",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def load_single_image(path, imgsz):
    """Letterbox an arbitrary image file the same way YoloDetectionDataset does at eval time (centered, no augment).
    Ground truth is included if a YOLO label file exists next to it under a sibling labels/ folder; otherwise empty."""
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    S = imgsz
    r = S / max(w0, h0)
    nw, nh = max(1, round(w0 * r)), max(1, round(h0 * r))
    img = img.resize((nw, nh), Image.BILINEAR)
    ox, oy = (S - nw) // 2, (S - nh) // 2
    canvas = Image.new("RGB", (S, S), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    canvas.paste(img, (ox, oy))

    boxes = []
    for cls, x1, y1, x2, y2 in YoloDetectionDataset._read_labels(path):
        x1, x2 = min(max(x1 * w0 * r + ox, 0), S), min(max(x2 * w0 * r + ox, 0), S)
        y1, y2 = min(max(y1 * h0 * r + oy, 0), S), min(max(y2 * h0 * r + oy, 0), S)
        if x2 - x1 > 2 and y2 - y1 > 2:
            boxes.append((cls, x1, y1, x2, y2))

    image = TF.to_tensor(canvas)
    target = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)
    return image, target


def make_sheet(tensor, target, option_b, yolo, names, conf, device, task):
    """One side-by-side PNG (Option B | YOLO26) for a single image tensor, green ground truth on both sides.
    Each panel is also labelled with that model's violation call for this image. Returns (sheet, {title: message})."""
    base = tensor_to_image(tensor)
    gt_boxes, gt_labels = ground_truth_boxes(target)
    gt_img = draw(base, gt_boxes, gt_labels, [None] * len(gt_labels), names, lambda _: GROUND_TRUTH, width=1)

    panels, messages = [], {}
    for title, predict in (("Option B", option_b), ("YOLO26", yolo)):
        boxes, labels, scores = run(predict, tensor.to(device), conf)
        message, is_violation = violation_text(task, labels, names)
        messages[title] = message
        panel = draw(gt_img, boxes, labels, scores, names, lambda c: PALETTE[c % len(PALETTE)])
        canvas = Image.new("RGB", (panel.width, panel.height + 40), (255, 255, 255))
        canvas.paste(panel, (0, 40))
        d = ImageDraw.Draw(canvas)
        d.text((4, 4), f"{title}: {len(boxes)} detections (conf >= {conf})", fill=(0, 0, 0))
        d.text((4, 22), message, fill=VIOLATION_COLOR if is_violation else OK_COLOR)
        panels.append(canvas)

    sheet = Image.new("RGB", (panels[0].width * 2 + 10, panels[0].height), (255, 255, 255))
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (panels[0].width + 10, 0))
    return sheet, messages


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--pick", action="store_true", help="open a file picker and run on one chosen image instead of sampling the test split")
    p.add_argument("--n", type=int, default=12, help="number of test images to draw (ignored with --pick)")
    p.add_argument("--conf", type=float, default=0.25, help="show detections with score at or above this value")
    p.add_argument("--require", help="comma-separated class names; keep only images whose labels contain all of them")
    p.add_argument("--tag", default="all", help="sub-folder name under analysis/predictions/<task>/ (default 'manual' with --pick)")
    p.add_argument("--option-b", help="Option B checkpoint (default: the trained best.pt)")
    p.add_argument("--yolo", help="YOLO26 weights (default: the trained best.pt)")
    p.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--device", default="", help="'' auto, '0' first GPU, or 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()
    option_b_path = Path(args.option_b or RUNS_DIR / f"{args.task}_optionb" / "weights" / "best.pt")
    yolo_path = Path(args.yolo or RUNS_DIR / "yolo26" / args.task / "weights" / "best.pt")
    for path in (option_b_path, yolo_path):
        if not path.exists():
            raise SystemExit(f"weights not found: {path}")

    device = pick_device(args.device)
    names, splits = load_data_yaml(TASKS[args.task]["data"])
    option_b = option_b_predictor(str(option_b_path), device)
    yolo = yolo_predictor(str(yolo_path), device)

    if args.pick:
        image_path = pick_image_file()
        if image_path is None:
            raise SystemExit("no image selected")
        tensor, target = load_single_image(image_path, args.imgsz)
        sheet, messages = make_sheet(tensor, target, option_b, yolo, names, args.conf, device, args.task)

        out_dir = ANALYSIS_DIR / "predictions" / args.task / (args.tag if args.tag != "all" else "manual")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{image_path.stem}.png"
        sheet.save(out_file)
        print(f"saved {out_file}")
        for title, message in messages.items():
            print(f"{title}: {message}")
        print("green boxes = ground truth (if a label file was found); coloured boxes = predictions")
        if sys.platform == "win32":
            os.startfile(out_file)  # open it immediately so the result is visible without hunting for the file
        return

    ds = YoloDetectionDataset(splits["test"], args.imgsz, augment=False)
    if len(ds) == 0:
        raise SystemExit(f"no test images in {splits['test']}")

    candidates = list(range(len(ds)))
    if args.require:
        wanted = [c.strip() for c in args.require.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in names]
        if unknown:
            raise SystemExit(f"unknown class(es) {unknown}; this task has {names}")
        need = {names.index(c) for c in wanted}
        candidates = [i for i in candidates if need <= classes_in(ds.images[i])]
        print(f"{len(candidates)} of {len(ds)} test images contain {wanted}")
        if not candidates:
            raise SystemExit("no test images match --require")

    count = min(args.n, len(candidates))
    picks = [candidates[j] for j in np.linspace(0, len(candidates) - 1, count).round().astype(int)]  # spread across the matches
    out_dir = ANALYSIS_DIR / "predictions" / args.task / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in picks:
        tensor, target = ds[int(idx)]
        sheet, _ = make_sheet(tensor, target, option_b, yolo, names, args.conf, device, args.task)
        sheet.save(out_dir / f"{Path(ds.images[int(idx)]).stem}.png")
        print(f"saved {Path(ds.images[int(idx)]).name}")

    print(f"\n{count} images written to {out_dir}")
    print("green boxes = ground truth; coloured boxes = predictions")


if __name__ == "__main__":
    main()
