# Two-Wheeler Violation Detection — AI Layer

**AIoT CSE712 project.** This repo is the AI/vision side of a system that watches a checkpoint,
triggers a camera when a two-wheeler is detected, and should flag helmet violations, triple riding,
and read the number plate. The IoT/hardware side (ESP32 sensors, trigger logic, camera firmware)
is a separate piece that still needs to be wired into this one — see **"If you're combining this
with the IoT side"** below, that's most of what's left.

New here? Read this file top to bottom once (10 minutes), then go to whichever doc below matches
what you're about to do.

| Doc | Read it when... |
|---|---|
| **This file** | You just opened the repo and need to know what's where and how to run it |
| [`AI_Implementation_Plan.md`](AI_Implementation_Plan.md) | You want the full history: every dataset decision, every finding, every phase's status, with dates and evidence |
| [`IOT_INTEGRATION_HANDOFF.md`](IOT_INTEGRATION_HANDOFF.md) | You're doing the IoT/firmware side and need to know exactly how to call into this code |
| [`training/README.md`](training/README.md) | You're training or re-evaluating the models specifically |

## What actually works right now (short version)

- ✅ Two detectors, two tasks, each trained two ways (our own model + a YOLO26 baseline) — all four trained, scored, checkpoints saved.
- ✅ A combined pipeline (`pipeline/aggregator.py`) that runs both detectors + plate OCR on one image and logs a violation record.
- ✅ A small dashboard (`dashboard/app.py`) to run an image and browse the log.
- 🟡 Plate OCR: the engine works, but the dataset doesn't have enough large, legible, correctly-scripted plate images to get a trustworthy accuracy number yet.
- ⬜ The IoT side (firmware, trigger, camera) isn't wired into any of this yet — that's the actual remaining work, and it's on the IoT/hardware teammate, not this codebase.

Full detail, numbers, and the reasoning behind every one of those: `AI_Implementation_Plan.md`.

## Repository layout

```
.
├── dataset/                     the two training-ready datasets (gitignored -- see "Getting the data" below)
│   ├── helmet_and_plate/        4 classes: bike, helmet, no-helmet, number-plate -- 24,936 / 3,810 / 2,678 train/valid/test
│   └── triple_riding/           1 class: tripleriding -- 3,475 / 331 / 165 train/valid/test
├── dataset_structuring/         the 6 scripts (run order 01-06) that built dataset/ from backup/'s raw downloads.
│                                 Historical/reproducibility record -- don't re-run unless intentionally rebuilding.
├── backup/                      raw Roboflow downloads, the original project PPT, the IoT firmware (gitignored,
│                                 same reason as dataset/). smartTrafficViolation-main.zip here is the IoT firmware
│                                 + cam_receiver_code.py -- still zipped, not yet extracted; see the handoff doc.
│
├── training/                    trains and evaluates both detectors. Its own README.md has the full how-to.
│   ├── config.py                 paths, task list, hyperparameter defaults, device selection
│   ├── yolo_data.py              dataset loader, letterboxing, augmentation (shared by both models)
│   ├── option_b/                 our own detector: MobileNetV3-Small backbone, anchor-free decoupled head
│   ├── train_option_b.py         trains Option B
│   ├── train_yolo26.py           trains the YOLO26 comparison baseline (Ultralytics)
│   ├── evaluate.py               scores both models on the same test split, same metric code
│   ├── analyze.py                per-epoch charts + summary.txt from finished runs -> project_root/analysis/
│   ├── show_predictions.py       draws both models' boxes on test images, or on one image YOU pick (--pick)
│   ├── train_models.ipynb        the whole training pipeline in one notebook (local or Google Colab)
│   ├── runs/                     trained checkpoints + logs (best.pt tracked in git; last.pt is not -- see .gitignore)
│   └── requirements.txt          install this first -- see "Setup" below
│
├── ocr/                         plate OCR: crop -> preprocess -> EasyOCR -> Indian-plate regex + character fix
│   ├── plate_ocr.py              `crops` cuts plate crops from the test split; `read` runs OCR on them
│   ├── crops/, labels.csv, results.csv   the current crop set and its (partial) ground truth / OCR output
│
├── pipeline/                    combines everything above into one decision
│   └── aggregator.py             Pipeline class: both detectors + OCR -> one violation record, logged to violations.db
│
├── dashboard/                   the UI
│   └── app.py                    Streamlit: run an image, browse the violation log
│
├── analysis/                    charts, per-epoch logs, summary.txt, test_comparison.png -- from training/analyze.py
├── violations.db                SQLite log written by the aggregator (gitignored -- runtime state, not shared)
├── violations/evidence/         annotated evidence images, one per processed snapshot (gitignored, same reason)
│
├── AI_Implementation_Plan.md    the full project history and current status -- see the table above
├── IOT_INTEGRATION_HANDOFF.md   exact next steps for wiring the IoT side into pipeline/aggregator.py
├── AI Model Architecture, Feature Engineering & Dataset Design.pptx   presentation deck (model/dataset design)
└── .venv/                       Python 3.13 virtual environment (gitignored)
```

## Setup

```powershell
# from the project root
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r training\requirements.txt
```

That one `requirements.txt` covers training, OCR, the aggregator, and the dashboard -- there's only one
venv for the whole project.

## Getting the data

`dataset/` and `backup/` are **not** in git -- `dataset/` alone is close to 2 GB and `backup/`'s raw
downloads are several GB more, impractical for a normal GitHub repo. To get them:

1. Ask whoever has the project folder locally (or the shared Drive link, if one exists) for `dataset/`
   and `backup/` directly -- this is the fast path and what you want for just running or demoing things.
2. Or, if you only have `backup/`'s raw Roboflow downloads: `dataset_structuring/01_merge_triple_riding.py`
   and `02_merge_helmet_plate.py` rebuild `dataset/` from them (see `AI_Implementation_Plan.md` section 02
   for exactly what they do and why).

You do **not** need either of these just to run the aggregator or dashboard on your own image --
only the trained checkpoints in `training/runs/` (which *are* in git) are needed for that.

## How to run things

All commands assume the project root as your working directory and the venv active.

**Train** (needs `dataset/`; see `training/README.md` for the full guide, including Colab):
```powershell
.venv\Scripts\python.exe training\train_option_b.py --task helmet_plate
.venv\Scripts\python.exe training\train_yolo26.py --task helmet_plate --weights yolo26s.pt
```

**Check a model's predictions by eye** -- either sampled from the test split, or on any image you pick yourself:
```powershell
.venv\Scripts\python.exe training\show_predictions.py --task helmet_plate --pick
```

**Run the combined pipeline on one image** (this is the main thing a teammate will want):
```powershell
.venv\Scripts\python.exe pipeline\aggregator.py --pick
.venv\Scripts\python.exe pipeline\aggregator.py --image path\to\snapshot.jpg --speed 42.5
```
Prints the violation call, saves an annotated evidence image to `violations/evidence/`, and logs a row
to `violations.db`. See `pipeline/aggregator.py`'s own docstring for every flag.

**Dashboard** (run an image through a browser UI, browse everything ever logged):
```powershell
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

**OCR investigation** (crop plates from the test split, run EasyOCR on them):
```powershell
.venv\Scripts\python.exe ocr\plate_ocr.py crops --n 40 --min-size 100
.venv\Scripts\python.exe ocr\plate_ocr.py read
```

## If you're combining this with the IoT side

This is almost certainly why you're reading this. Short version: the IoT side (ESP32 firmware +
`cam_receiver_code.py`, zipped at `backup/smartTrafficViolation-main.zip`) needs to (1) fix a known
camera-resolution bug, (2) send its measured speed value over the wire, and (3) call
`pipeline.aggregator.Pipeline.process(image_path, speed_kmph)` right after it saves a snapshot.
Everything after that point -- running both models, OCR, saving evidence, logging to `violations.db`
-- is already built and already works.

**Full step-by-step, with exact line numbers, code to paste in, and a testing checklist:
[`IOT_INTEGRATION_HANDOFF.md`](IOT_INTEGRATION_HANDOFF.md).** It assumes zero prior context on this
AI codebase.

## Known limitations (be upfront about these in the report/demo)

- **Plate OCR accuracy is unmeasured.** The engine works (confirmed), but the test split doesn't have
  enough large + legible + correctly-scripted (Indian, not the Nepali plates folded in for detection
  diversity) plate crops to get a real number yet. See `AI_Implementation_Plan.md`'s OCR findings.
- **The 20 km/h overspeed threshold is a placeholder** in both the firmware and
  `pipeline/aggregator.py`'s `OVERSPEED_THRESHOLD_KMPH` -- recalibrate both together once real speeds
  are measured at the mounted rig.
- **Option B (our own detector) trails YOLO26** by design expectation -- it's a small model built from
  scratch for this course; YOLO26 is a mature, pretrained architecture. The gap is the point of the
  comparison, not a bug. See `analysis/summary.txt` for exact numbers.
