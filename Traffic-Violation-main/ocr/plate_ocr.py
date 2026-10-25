"""Plate OCR, step 1: crop plates from the test split and read them with EasyOCR.

  python ocr/plate_ocr.py crops --n 60               # cut plate crops >=100px/side, write ocr/labels.csv
  python ocr/plate_ocr.py crops --n 60 --min-size 60 # allow smaller (more crops, more of them unreadable)
  python ocr/plate_ocr.py read                       # run EasyOCR on the crops, write ocr/results.csv

Crops come from the ground-truth plate boxes, so OCR is measured on its own, without detection errors.
--min-size filters out plate boxes smaller than that on their shorter side, measured in the *original*
image before any upscaling: most plate boxes in this dataset are tiny (median ~22px) and are not legible
to any OCR engine (or a human) no matter how they're preprocessed, so testing against them just produces
noise. The default of 100px keeps roughly the best 10% of plate boxes in the test split -- see
AI_Implementation_Plan.md's OCR findings for the measurements behind this.

Fill in the plate_text column of ocr/labels.csv by hand (the true text as printed on the plate, no spaces);
`read` then reports exact-match accuracy.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OCR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OCR_DIR.parent / "training"))  # reuse the dataset loader and task paths

from config import TASKS  # noqa: E402
from yolo_data import IMG_EXTS, load_data_yaml  # noqa: E402
CROP_DIR = OCR_DIR / "crops"
LABELS = OCR_DIR / "labels.csv"
RESULTS = OCR_DIR / "results.csv"
PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
DIGIT_FIX = str.maketrans({"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2"})  # letters often misread for digits
PLATE_CLASS = "number-plate"


def preprocess(bgr):
    """Upscale 3x, grayscale, boost contrast. The plan's steps: cubic resize, grayscale, CLAHE."""
    big = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def crop_box(img, box, pad=0.10):
    """Crop an (x1, y1, x2, y2) box with 10% padding on each side, clipped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, x2 = int(max(0, x1 - px)), int(min(w, x2 + px))
    y1, y2 = int(max(0, y1 - py)), int(min(h, y2 + py))
    return img[y1:y2, x1:x2]


def make_crops(n, min_size=100):
    names, splits = load_data_yaml(TASKS["helmet_plate"]["data"])
    plate_id = names.index(PLATE_CLASS)
    images = sorted(p for p in splits["test"].iterdir() if p.suffix.lower() in IMG_EXTS)

    # Every plate box in the test split whose shorter side is >= min_size px in the original image.
    # Collected across the whole split first, THEN spread n picks evenly across it -- filtering per
    # image (the old behaviour) would have let a handful of early images with many tiny boxes crowd
    # out the picks before size was ever considered.
    candidates = []
    for img_path in images:
        label = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if not label.exists():
            continue
        lines = [l for l in label.read_text(encoding="utf-8").splitlines()
                 if l.split() and int(float(l.split()[0])) == plate_id]
        if not lines:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in lines:
            xc, yc, bw, bh = [float(v) for v in line.split()[1:5]]
            box = ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)
            if min(box[2] - box[0], box[3] - box[1]) >= min_size:
                candidates.append((img_path, box))

    if not candidates:
        raise SystemExit(f"no plate boxes >= {min_size}px (shorter side) in the test split; try a smaller --min-size")
    picks = [candidates[j] for j in np.linspace(0, len(candidates) - 1, min(n, len(candidates))).round().astype(int)]
    print(f"{len(candidates)} plate boxes >= {min_size}px; picking {len(picks)} spread evenly across them")

    for stale in CROP_DIR.glob("*.png"):  # clear any crops from a previous run/filter so nothing stale lingers
        stale.unlink()
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    rows, cache = [], {}
    for img_path, box in picks:
        if img_path not in cache:
            cache[img_path] = cv2.imread(str(img_path))
        crop_name = f"{img_path.stem[:40]}_{len(rows):03d}.png"
        cv2.imwrite(str(CROP_DIR / crop_name), preprocess(crop_box(cache[img_path], box)))
        rows.append({"crop": crop_name, "plate_text": ""})

    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "plate_text"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} crops written to {CROP_DIR}")
    print(f"type the true plate text into the plate_text column of {LABELS}")


def correct(raw):
    """Clean EasyOCR output: keep letters and digits, then fix the characters the format says must be digits or letters."""
    text = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if PLATE_RE.match(text):
        return text
    # Try the confusable map on the digit positions only: Indian plates are 2 letters, 1-2 digits, 1-3 letters, 4 digits.
    m = re.match(r"^([A-Z0-9]{2})([A-Z0-9]{1,2})([A-Z0-9]{1,3})([A-Z0-9]{4})$", text)
    if m:
        state, district, series, number = m.groups()
        fixed = state + district.translate(DIGIT_FIX) + series + number.translate(DIGIT_FIX)
        if PLATE_RE.match(fixed):
            return fixed
    return text


def read_all():
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    crops = sorted(CROP_DIR.glob("*.png"))
    if not crops:
        raise SystemExit("no crops found; run `plate_ocr.py crops` first")
    truth = {}
    if LABELS.exists():
        with open(LABELS, encoding="utf-8") as f:
            truth = {r["crop"]: r["plate_text"].strip().upper() for r in csv.DictReader(f)}

    rows, hits, labelled = [], 0, 0
    for crop in crops:
        image = cv2.imread(str(crop), cv2.IMREAD_GRAYSCALE)  # pass the array, not the path: EasyOCR's path loader needs an extra plugin
        raw = " ".join(t for _, t, _ in reader.readtext(image, detail=1, paragraph=False))
        text = correct(raw)
        valid = bool(PLATE_RE.match(text))
        true = truth.get(crop.name, "")
        if true:
            labelled += 1
            hits += text == true
        rows.append({"crop": crop.name, "raw": raw, "read": text, "valid_format": valid, "true": true,
                     "match": (text == true) if true else ""})

    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"read {len(rows)} crops; {sum(r['valid_format'] for r in rows)} match the Indian plate format")
    if labelled:
        print(f"exact match against typed labels: {hits}/{labelled} = {hits / labelled:.1%}")
    else:
        print("no labels typed yet; accuracy will be reported once plate_text is filled in")
    print(f"wrote {RESULTS}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--n", type=int, default=60)
    c.add_argument("--min-size", type=int, default=100,
                    help="skip plate boxes smaller than this (px, shorter side, original image)")
    sub.add_parser("read")
    args = p.parse_args()
    if args.cmd == "crops":
        make_crops(args.n, args.min_size)
    else:
        read_all()


if __name__ == "__main__":
    main()
