"""Aggregator: the one place that combines everything in the AI section into a single decision.

For one image, runs the helmet/plate model, the triple-riding model, and plate OCR, combines their
output (plus a speed reading, if one is known) into a single violation record, saves one annotated
evidence image with every box drawn on it, and logs a row to violations.db at the project root.
`dashboard/app.py` reads that same violations.db.

  python pipeline/aggregator.py --pick
  python pipeline/aggregator.py --image path\\to\\snapshot.jpg --speed 42.5

By default both tasks use their YOLO26 checkpoint, since it scored higher than Option B on both
tasks' test split (see analysis/summary.txt). Pass --helmet-model option_b / --triple-model option_b
to run our own detector instead.

This module is also meant to be imported directly, e.g. by cam_receiver_code.py once the IoT side is
wired up (see IOT_INTEGRATION_HANDOFF.md): build one Pipeline() up front, then call .process(path,
speed_kmph=...) per captured snapshot.
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PIPELINE_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "training"))
sys.path.insert(0, str(PROJECT_DIR / "ocr"))

from config import RUNS_DIR, TASKS, pick_device  # noqa: E402
from evaluate import option_b_predictor, yolo_predictor  # noqa: E402
from plate_ocr import PLATE_CLASS, PLATE_RE, correct, crop_box, preprocess  # noqa: E402
from show_predictions import PALETTE, draw, load_single_image, pick_image_file, run, tensor_to_image, violation_text  # noqa: E402
from yolo_data import load_data_yaml  # noqa: E402

DB_PATH = PROJECT_DIR / "violations.db"
EVIDENCE_DIR = PROJECT_DIR / "violations" / "evidence"
OVERSPEED_THRESHOLD_KMPH = 40.0 # same placeholder as the firmware's trafficViolationSystem.ino; recalibrate both together in Phase 7


def init_db(db_path):
    con = sqlite3.connect(db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            plate_number TEXT,
            plate_valid_format INTEGER,
            violation_types TEXT,
            speed_kmph REAL,
            raw_image_path TEXT,
            annotated_image_path TEXT,
            confidences TEXT,
            helmet_message TEXT,
            triple_message TEXT
        )"""
    )
    con.commit()
    con.close()


def log_violation(db_path, record):
    init_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.execute(
        "INSERT INTO violations (timestamp, plate_number, plate_valid_format, violation_types, speed_kmph, "
        "raw_image_path, annotated_image_path, confidences, helmet_message, triple_message) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["timestamp"], record["plate_number"], int(record["plate_valid_format"]),
            ",".join(record["violation_types"]), record["speed_kmph"], record["raw_image_path"],
            record["annotated_image_path"], json.dumps(record["confidences"]),
            record["helmet_message"], record["triple_message"],
        ),
    )
    con.commit()
    row_id = cur.lastrowid
    con.close()
    return row_id


def resolve_weights(task, model_choice):
    path = (RUNS_DIR / f"{task}_optionb" / "weights" / "best.pt" if model_choice == "option_b"
            else RUNS_DIR / "yolo26" / task / "weights" / "best.pt")
    if not path.exists():
        raise SystemExit(f"weights not found: {path}")
    return path


class Pipeline:
    """Loads both detectors (and, lazily, the OCR reader) once. Build one instance and reuse it —
    model loading takes a few seconds, so never construct this inside a per-image loop."""

    def __init__(self, helmet_model="yolo26", triple_model="yolo26", conf=0.25, device="", db_path=None):
        self.device = pick_device(device)
        self.conf = conf
        self.db_path = Path(db_path) if db_path else DB_PATH
        init_db(self.db_path)

        self.names, self.predictors = {}, {}
        for task, choice in (("helmet_plate", helmet_model), ("triple_riding", triple_model)):
            names, _ = load_data_yaml(TASKS[task]["data"])
            weights = resolve_weights(task, choice)
            predictor = (option_b_predictor(str(weights), self.device) if choice == "option_b"
                         else yolo_predictor(str(weights), self.device))
            self.names[task] = names
            self.predictors[task] = predictor

        self._reader = None  # EasyOCR: loaded on first use, not at startup

    def _ocr_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=(self.device.type == "cuda"), verbose=False)
        return self._reader

    def process(self, image_path, speed_kmph=None, imgsz=640, save_evidence=True):
        image_path = Path(image_path)
        tensor, _ = load_single_image(image_path, imgsz)
        canvas = tensor_to_image(tensor)  # the same 640px letterboxed image both models see

        results = {}
        for task in ("helmet_plate", "triple_riding"):
            boxes, labels, scores = run(self.predictors[task], tensor.to(self.device), self.conf)
            names = self.names[task]
            message, is_violation = violation_text(task, labels, names)
            results[task] = {"boxes": boxes, "labels": labels, "scores": scores, "names": names,
                              "message": message, "is_violation": is_violation}
        hp, tr = results["helmet_plate"], results["triple_riding"]

        plate_text, plate_valid = "", False
        if PLATE_CLASS in hp["names"]:
            plate_id = hp["names"].index(PLATE_CLASS)
            plate_idxs = [i for i, lab in enumerate(hp["labels"]) if int(lab) == plate_id]
            if plate_idxs:
                best = max(plate_idxs, key=lambda i: hp["scores"][i])
                canvas_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
                crop = crop_box(canvas_bgr, hp["boxes"][best])
                if crop.size:
                    raw = " ".join(t for _, t, _ in self._ocr_reader().readtext(preprocess(crop), detail=1, paragraph=False))
                    plate_text = correct(raw)
                    plate_valid = bool(PLATE_RE.match(plate_text))

        # one evidence image with both tasks' boxes on it; offset triple-riding's palette index so its
        # single class never lands on the same colour as one of helmet_plate's four classes
        evidence = draw(canvas, hp["boxes"], hp["labels"], hp["scores"], hp["names"], lambda c: PALETTE[c % len(PALETTE)])
        evidence = draw(evidence, tr["boxes"], tr["labels"], tr["scores"], tr["names"], lambda c: PALETTE[(c + 4) % len(PALETTE)])

        violation_types = []
        if hp["is_violation"]:
            violation_types.append("no_helmet")
        if tr["is_violation"]:
            violation_types.append("triple_riding")
        if speed_kmph is not None and speed_kmph > OVERSPEED_THRESHOLD_KMPH:
            violation_types.append("overspeed")

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "plate_number": plate_text,
            "plate_valid_format": plate_valid,
            "violation_types": violation_types,
            "speed_kmph": speed_kmph,
            "raw_image_path": str(image_path),
            "annotated_image_path": "",
            "helmet_message": hp["message"],
            "triple_message": tr["message"],
            "confidences": {
                "helmet_plate_max": float(max(hp["scores"], default=0.0)),
                "triple_riding_max": float(max(tr["scores"], default=0.0)),
            },
        }

        if save_evidence:
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ev_path = EVIDENCE_DIR / f"{ts}_{image_path.stem}.png"
            evidence.save(ev_path)
            record["annotated_image_path"] = str(ev_path)

        record["id"] = log_violation(self.db_path, record)
        record["evidence_image"] = evidence
        return record


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="path to an image to process")
    p.add_argument("--pick", action="store_true", help="open a file picker instead of --image")
    p.add_argument("--speed", type=float, help="measured speed in km/h, if known (manual testing)")
    p.add_argument("--helmet-model", choices=["yolo26", "option_b"], default="yolo26")
    p.add_argument("--triple-model", choices=["yolo26", "option_b"], default="yolo26")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="", help="'' auto, '0' first GPU, or 'cpu'")
    p.add_argument("--db", default=None, help="path to violations.db (default: project_root/violations.db)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.pick:
        image_path = pick_image_file()
        if image_path is None:
            raise SystemExit("no image selected")
    elif args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise SystemExit(f"image not found: {image_path}")
    else:
        raise SystemExit("give --image PATH or --pick")

    pipeline = Pipeline(helmet_model=args.helmet_model, triple_model=args.triple_model,
                         conf=args.conf, device=args.device, db_path=args.db)
    record = pipeline.process(image_path, speed_kmph=args.speed)

    print(f"\n{image_path.name}")
    print(f"  helmet/plate: {record['helmet_message']}")
    print(f"  triple riding: {record['triple_message']}")
    plate_note = "" if (record["plate_valid_format"] or not record["plate_number"]) else "  [does not match the Indian plate format]"
    print(f"  plate text: {record['plate_number'] or '(none read)'}{plate_note}")
    if record["speed_kmph"] is not None:
        print(f"  speed: {record['speed_kmph']:.1f} km/h")
    print(f"  violation types: {', '.join(record['violation_types']) or 'none'}")
    print(f"  evidence image: {record['annotated_image_path']}")
    print(f"  logged as violations.db row {record['id']}")

    if args.pick and sys.platform == "win32":
        os.startfile(record["annotated_image_path"])


if __name__ == "__main__":
    main()
