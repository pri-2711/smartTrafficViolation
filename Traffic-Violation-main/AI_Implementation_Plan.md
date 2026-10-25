# Two-Wheeler Vision Stack

**AIoT Smart Helmet & Traffic Violation Monitoring — AI Layer**

Implementation plan for the three vision tasks that sit on top of the finished IoT trigger & capture rig: helmet detection, triple-riding detection, and automatic number-plate OCR — plus the glue that turns detections into logged violations.

New to the repo? Start at the project root's **`README.md`** instead — it's the quick map of what's
where and how to run things. This file is the detailed history: every dataset decision, every
finding, every phase's status, with dates and evidence.

Live artifact: https://claude.ai/code/artifact/5b563b3c-1095-4457-82d5-ce235d726955

---

## Table of contents

1. [Current status & deliverables](#current-status--deliverables)
2. [Critical findings](#01-critical-findings)
3. [Dataset audit](#02-dataset-audit)
4. [System architecture](#03-system-architecture)
5. [Models & datasets](#04-models--datasets)
6. [Phased implementation plan](#05-phased-implementation-plan)
7. [Risks & mitigations](#06-risks--mitigations)
8. [Code reference](#07-code-reference)
9. [Sources](#sources)

---

## Current status & deliverables

Quick snapshot of what actually exists on disk right now, kept up to date as the project progresses.

### Project folder layout

```
Artificial Intelligence of Things [CSE712]/
├── dataset/
│   ├── helmet_and_plate/   — training-ready: 24,936 / 3,810 / 2,678 (train/valid/test), 4 classes
│   └── triple_riding/      — training-ready: 3,475 / 331 / 165, 1 class, ~20% negatives
├── dataset_structuring/    — the 6 scripts that built & audited the datasets above (run order 01–06)
├── training/               — Option B + YOLO26 training, evaluation, and the show_predictions.py viewer (both tasks trained, see below)
├── analysis/               — charts, per-epoch logs, summary.txt and test_comparison from analyze.py
├── ocr/                    — plate_ocr.py (crop → preprocess → EasyOCR → regex fix); not yet working, see below
├── pipeline/               — aggregator.py: combines both detectors + OCR into one violation record, logs to violations.db
├── dashboard/              — app.py: Streamlit UI — run an image, browse the violation log
├── violations.db           — SQLite log written by pipeline/aggregator.py, read by dashboard/app.py
├── violations/evidence/    — annotated evidence images, one per processed snapshot
├── backup/                 — original Roboflow downloads, source zips, the project PPT, and the IoT firmware
│                             (still zipped as smartTrafficViolation-main.zip — never extracted into the repo)
├── AI Model Architecture, Feature Engineering & Dataset Design.pptx  — 17-slide presentation (see below)
├── AI_Implementation_Plan.md  — this file
├── IOT_INTEGRATION_HANDOFF.md — next steps for whoever wires the IoT side into the aggregator
└── .venv/                  — Python environment used for dataset scripts, training, OCR, and the dashboard
```

### Done so far

*Last verified against the actual files on disk: 2026-10-19.*

- ✅ **Dataset selection & audit** — 4 public Roboflow datasets evaluated, 2 kept as primary sources, 1 repurposed as a negatives source, 1 folded in for plate detection only. Full writeup: [Dataset audit](#02-dataset-audit).
- ✅ **Datasets built, merged, and fixed** — `dataset/helmet_and_plate` and `dataset/triple_riding` exist, are leakage-free, and passed a full byte-level integrity audit (one real defect — 328 polygon-format labels — found and fixed). Training-ready as-is.
- ✅ **Dataset pipeline code organized** — every script that built and verified the datasets lives in `dataset_structuring/01`–`06`, with paths corrected to match the current folder layout and re-verified to still run correctly.
- ✅ **Architecture decision made** — two separate detectors (helmet+plate, triple-riding), each trained two ways for comparison — see below.
- ✅ **Presentation deck built** — `AI Model Architecture, Feature Engineering & Dataset Design.pptx` (17 slides): AI model selection, feature engineering across the pipeline, and the dataset selection/audit story, structured to match a reference deck's depth.
- ✅ **Model training (Phases 3–4) — done.** Both tasks trained twice: **Option B** (our own MobileNetV3-Small + anchor-free detector, built from scratch in `training/option_b/`) and **YOLO26** (Ultralytics, pretrained) as a comparison baseline. All 4 checkpoints exist (`training/runs/{helmet_plate,triple_riding}_optionb/weights/best.pt`, `training/runs/yolo26/{helmet_plate,triple_riding}/weights/best.pt`). Test-split scores in `analysis/summary.txt`: helmet/plate mAP50 0.860 (Option B) / 0.945 (YOLO26); triple-riding mAP50 0.953 (Option B) / 0.963 (YOLO26).
- ✅ **Evaluation & visual QA tooling built** — `training/evaluate.py` (head-to-head mAP scoring), `training/analyze.py` (charts, `analysis/`), `training/show_predictions.py` (side-by-side detection viewer; now also supports `--pick` to manually choose any image via a file-picker dialog and run both models on it, printing a helmet/triple-riding violation call per model).
- 🟡 **OCR — built but not working on the plates that matter, and never actually measured.** `ocr/plate_ocr.py` (crop plate box → 3x upscale → grayscale → CLAHE → EasyOCR → Indian-plate regex + confusable-character fix) runs end to end, but: (1) `ocr/labels.csv` ground-truth column is still 100% blank — nobody has typed in the real plate text for any of the 59 crops, so accuracy has never actually been computed; (2) on inspecting `ocr/results.csv`, EasyOCR returns **completely empty text** on all 47 crops from the helmet+plate (Indian-plate) dataset — the only ones OCR needs to work on. The only crops that returned any text are 12 Nepali-plate crops, which the dataset audit itself already flagged as not valid for OCR testing, and none of those matched the Indian format either. **OCR is effectively non-functional right now** — see [Critical findings](#01-critical-findings).
- ✅ **Aggregator + dashboard built (2026-10-19).** `pipeline/aggregator.py` runs both detectors and plate OCR on one image, combines them (plus an optional speed value) into one violation record, saves an annotated evidence image to `violations/evidence/`, and logs a row to `violations.db`. `dashboard/app.py` (Streamlit) gives a simple local UI: pick/upload an image and run it, or browse the violation log with counts and an evidence-image viewer. Both tested end-to-end against real test-split images — detection, logging, and the dashboard boot all confirmed working. See `IOT_INTEGRATION_HANDOFF.md` for exactly how this plugs into the real capture rig once the IoT side is ready.
- ⬜ **ESP32-CAM firmware fix — confirmed still NOT applied.** Re-checked directly inside `backup/smartTrafficViolation-main.zip` (the firmware has never been extracted into the repo, only kept zipped): `CameraWebServer.ino` line 93 still reads `s->set_framesize(s, FRAMESIZE_QVGA);`, so every `/capture` frame is still forced to 320×240. Single highest-priority remaining item; see [Critical findings](#01-critical-findings).
- ⬜ **Pipeline integration / aggregator (Phase 6) — not started.** No code anywhere in the repo combines the helmet/plate model + triple-riding model + OCR + speed into one violation decision or log. `cam_receiver_code.py` (inside the same zip) only saves the raw JPEG snapshot — it has no call into any Python inference code at all. Also: the ESP32's TCP trigger message is literally `CAPTURE|date|time` — the measured speed value is computed on the ESP32 and printed to Serial but **never sent** to the receiver, so the aggregator won't have a speed number to combine until the firmware message is extended.
- ⬜ **Field testing & dashboard (Phases 7–8)** — not started; depend on the above.

---

## 01. Critical findings

*From reading the firmware, receiver script, and actual sample snapshots before writing this plan.*

### 🔴 Fix before training anything — ESP32-CAM is capturing at 320×240 (QVGA), not the resolution the firmware appears to configure

`CameraWebServer.ino` sets `FRAMESIZE_UXGA` (or `SVGA` without PSRAM) during `esp_camera_init()` — but a later line, `s->set_framesize(s, FRAMESIZE_QVGA)`, drops the live sensor back down to 320×240 "for higher initial frame rate." That line affects the `/capture` endpoint too, since it's the same sensor handle streaming and single-shot both read from. Confirmed directly: all six sample images in the `snapshots/` folder are exactly 320×240 and 3–4 KB.

At 320×240, a number plate photographed from a few metres away can be well under 100px wide — too few pixels per character for OCR to work reliably, and marginal for a small-object detector to localize confidently. This one setting is the single biggest lever on final accuracy, for both plate OCR and helmet detection, and it's a one-line fix.

**Fix:** remove/guard that downgrade line, or explicitly call `s->set_framesize(s, FRAMESIZE_SVGA)` (800×600) right before the `/capture` handler runs. The AI-Thinker board has PSRAM, so UXGA (1600×1200) is available too — test SVGA first since larger JPEGs take longer to transfer over Wi-Fi and the `CAMERA_DELAY_MS` budget is only 1000 ms.

**Status: confirmed still broken (re-checked 2026-10-19).** The firmware was never extracted into the repo — it sits zipped at `backup/smartTrafficViolation-main.zip`. Opened it directly and line 93 of `FINAL SYSTEM/CameraWebServer/CameraWebServer.ino` is unchanged: `s->set_framesize(s, FRAMESIZE_QVGA);`.

### 🔴 New — the ESP32 trigger message never carries the measured speed

`trafficViolationSystem.ino` computes `speedKMPH` and prints it to Serial for debugging, but the TCP message it actually sends to the Python receiver (`cam_receiver_code.py`) is just `CAPTURE|<date>|<time>` — no speed field. Phase 6's aggregator is specified to combine "speed (from the ESP32 trigger) + helmet status + rider count + plate text," but as written there is no speed value for it to read on the receiver side.

**Fix:** extend the trigger message format to `CAPTURE|<date>|<time>|<speed>` and parse the extra field in `cam_receiver_code.py`.

**Status: not yet fixed — found during the 2026-10-19 combination-readiness audit.**

### 🔴 Plate OCR: the "no text found" bug is fixed, but reveals a deeper dataset problem

Originally, `ocr/plate_ocr.py read` returned completely empty `raw`/`read` output for every real Indian-plate crop. Root cause found: plate boxes in `dataset/helmet_and_plate`'s test split are **tiny** — median 22px on the shorter side, only 10% reach 100px — too small for any OCR engine (or a human) to read, upscaling or not. Confirmed by eye: a sampled crop at ~34×22px pre-upscale was genuinely illegible static.

**Fix applied (2026-10-19):** `ocr/plate_ocr.py crops` now takes `--min-size` (default 100px, measured on the original image before upscaling) and only crops plate boxes that clear it — see the script's docstring. Re-ran it: EasyOCR now returns real text on 35 of 40 crops (`3184I`, `UEEU8033`, `J06NARTOO8UTBQQ`, etc.) — the "zero text" bug is gone.

**But:** 39 of the 40 size-qualifying crops turned out to be from the **Nepali plate dataset** (`Number Plate.v3i.yolo26`, folded in only for plate *detection* diversity), not the Indian one — its source photos are simply much higher-resolution. Looked at several directly: legible, but Devanagari script (not the Latin/Indian format the regex expects) and some are blank embossed metal with no paint contrast. So **0/40 match the Indian plate format**, but that's now a dataset-selection problem, not an OCR-engine failure. `ocr/labels.csv`'s ground truth is still unfilled, so a real accuracy number still doesn't exist — now it's genuinely blocked on finding Indian-plate crops that are both ≥100px *and* actually from the right dataset, which this test split may simply not have many of.

**Status: partially resolved.** OCR engine confirmed working; the remaining blocker is that this dataset doesn't have enough large, legible, correctly-scripted plates to test against — see the PaddleOCR note below for why that avenue is also closed for now, and [Models & datasets](#04-models--datasets) for the dataset mix.

### 🔴 Answers "do we need another dataset" — Yes, the helmet + number-plate Roboflow dataset almost certainly has no triple-riding class

"Triple riding" isn't a visual attribute like "helmet" — it's a count of people associated with one motorcycle, so a dataset built for helmet/plate detection won't carry the class needed for it. Solved via public datasets — see [Models & datasets](#04-models--datasets).

**Status: resolved** — see [Dataset audit](#02-dataset-audit) and [Current status](#current-status--deliverables).

### Other notes

- **The overspeed threshold (20 km/h) is a placeholder.** Reasonable for debugging the Wi-Fi/trigger chain, but needs recalibrating against real measured speeds once the rig is mounted at its actual test location — a single HC-SR04 sampled every 150 ms is a coarse speed estimate, so expect noise.
- **Small naming mismatch between the PPT and firmware.** The project PPT describes an "IR Obstacle Sensor" as the checkpoint trigger; the actual code in `trafficViolationSystem.ino` arms on two PIR motion sensors instead. Not an AI-layer concern, but worth reconciling before the report/viva.

---

## 02. Dataset audit

Four public Roboflow datasets were evaluated in two rounds: split sizes, class distribution, label format, source-image leakage across splits, and a visual spot-check of sample images (not just the READMEs). All four raw downloads are preserved under `backup/` for reference.

### Round 1 — original two datasets

| Dataset | Images | Classes | Format check | License |
|---|---|---|---|---|
| **helmet and number plate dataset** (`helmet-and-number-plate-...-iityz`, v3) | 20,287 (17,742 / 1,690 / 855) | `bike` 25,550 boxes · `helmet` 43,643 · `no-helmet` 19,102 · `number-plate` 20,134 | ✅ standard YOLO txt, trains directly with `ultralytics` | CC BY 4.0 |
| **triple dataset** (`ayuwidyaa/deteksi-triple-riding`, v3) | 3,177 (2,780 / 265 / 132) | `tripleriding` — 3,589 boxes, every image has ≥1 box | ✅ standard YOLO txt | CC BY 4.0 |

**Helmet dataset — fix before trusting val/test mAP.** ~11–13% of validation and test images were near-duplicates of training images. Grouping images by source filename (stripping Roboflow's per-copy hash suffix) found: **528 source images in both train and valid**, **306 in both train and test**, **107 in both valid and test** — out of 4,623 / 1,567 / 823 unique sources per split. These are rotated/blurred augmented copies of the same photo split across sets (confirmed genuine via UUID-suffixed filenames that can't collide by chance). Effect: reported mAP would have read **higher than real-world accuracy**. **Fixed** by re-grouping all three splits by source stem and reassigning whole groups to one split only — see `dataset_structuring/02_merge_helmet_plate.py`.

**Helmet dataset — spot-check before training.** At least one clearly off-domain, mislabeled image was found (filename stem `-1x-1`): a photo of a crowd wearing COVID face masks on an escalator in a Thai mall, labeled with 10 `no-helmet` boxes. Looks like contamination from a merged sub-source, not systemic — a wider spot-check of ~10 other images were all genuine on-topic photos.

**Triple dataset — needed negative examples.** 0 of 3,177 label files were empty — the model would never have seen a contrastive single/double-rider example labeled as background, so a detector trained purely on positives would tend to over-fire on any populated motorcycle. **Fixed** by adding negatives sourced from the Round 2 violation dataset (below), reaching ~20% negatives per split.

- Secondary note: this dataset's augmentation included a 50% chance of *vertical* flip — a physically-impossible road orientation. Not a blocker, not addressed.
- Also worth knowing: nearly all 2,780 training images trace back to just 3 physical CCTV cameras (Nipa Nipa, Borong Antang, Tinggi Mae Merapi in Makassar, Indonesia), all overhead/top-down — very different from the ESP32-CAM's roadside mounting height/angle. This is why Phase 2's real-capture step matters more for this model than the helmet one.

### Round 2 — two larger datasets downloaded later

| Dataset | Images | Classes | Splits | License |
|---|---|---|---|---|
| **helmet and triple riding dataset** (`msuiit-esuvy/helmet-dataset-jahk2`, v3) | 27,010 (25,503 / 996 / 499) | `No Violation` 75,226 boxes · `Violation` 68,075 boxes | ✅ zero leakage across splits | CC BY 4.0 |
| **Number Plate.v3i.yolo26** (`sujalkishwor/number-plate-wevp8`, v3) | 11,137 (7,793 / 2,223 / 1,121) | `license_plate` — 10,961 boxes, 841 negative (no-plate) images included | ✅ zero leakage, no augmentation | CC BY 4.0 |

**Bigger ≠ better here — a structural problem, not a quality one.** "helmet and triple riding dataset" only has 2 classes: `No Violation` / `Violation` — no separate helmet class, no separate triple-riding class. The project requires Helmet Detection and Triple Riding Detection as two independently-graded features, and the violation log needs a **type** field a binary flag can't supply. Content is genuinely good (real RPi-camera highway/urban footage from multiple Myanmar locations — Bago, Mandalay, NyaungU, Pakokku — well balanced 52%/48%, zero split leakage), but the label scheme doesn't fit. **Repurposed instead of discarded:** its 75,226 `No Violation` boxes became the negative-example source for the triple-riding model.

**Domain check — script mismatch for OCR, fine for detection.** "Number Plate.v3i.yolo26" is Nepali plates (red background, Devanagari script), not Indian. Fine to use for **plate detection** (visual diversity helps a small detector generalize). **Not usable for OCR validation** — EasyOCR/PaddleOCR configured for English can't read Devanagari, and the Indian-plate regex `^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$` will never match this dataset's ground truth. Used only for training the plate detector; OCR must be validated against real captured Indian plates (Phase 7).

### ✅ Final training-ready datasets — `dataset/`

- **`dataset/triple_riding/`** — the 3,177 original `tripleriding` positives plus 794 negatives pulled from the violation dataset's zero-Violation images, deduped by source photo before assignment. **3,475 train / 331 valid / 165 test**, ~20% negative in every split, zero cross-split leakage.
- **`dataset/helmet_and_plate/`** — the original helmet+plate dataset re-split by source image (fixing the leakage) plus all 11,137 Nepali plate images folded in as extra `number-plate`-class examples. **24,936 train / 3,810 valid / 2,678 test**, zero cross-split leakage. Box counts for bike/helmet/no-helmet are byte-for-byte unchanged from the original (25,550 / 43,643 / 19,102 — confirms nothing was lost in the re-split); number-plate boxes grew from 20,134 to **31,095**.

Both folders have a correct `data.yaml` and a `README.merge.txt` documenting exact provenance. Built and verified by `dataset_structuring/01_merge_triple_riding.py` and `02_merge_helmet_plate.py`.

Final task → source mapping:
- **Helmet detection** → helmet and number plate dataset (has the class granularity needed)
- **Triple riding** → triple dataset, with negatives pulled from the helmet+triple-riding dataset's `No Violation` images
- **Number plate detection** → helmet+plate dataset's plate boxes (~20K) combined with Number Plate.v3i.yolo26 (~11K)
- **Not used as a standalone model** → the binary Violation/No-Violation dataset — kept only as a negative-example source. (`harshas-workspace/triple-ride-detection`, a 5-class unified dataset, remains an unexplored option if a larger correctly-granular triple-riding set is wanted later.)

### ✅ Final integrity audit — training-ready, defect found and fixed

Full byte-level pass over all 35,395 images across both datasets (not a sample) — decoded every image, parsed every label line, hashed every file: **0 corrupt/unreadable images, 0 orphan images or labels, 0 cross-split leakage** (checked two independent ways — source-stem regex and exact byte-hash — both zero on both datasets). Run via `dataset_structuring/04_audit_final_datasets.py`.

**Found and fixed:** 328 label lines in the helmet+plate dataset (199 files: 176 train / 7 valid / 16 test) were in polygon/segmentation format (`class_id x1 y1 x2 y2 ... xn yn`) instead of plain bounding boxes — a leftover from the original Roboflow source. A plain YOLO *detect*-task loader expects exactly 5 fields per line, so these would have errored or been silently mis-parsed. Converted each to `class_id x_center y_center width height` via the polygon's min/max extent (`dataset_structuring/05_fix_polygon_labels.py`); total box count unchanged before/after (119,390) — nothing lost, only reformatted. Re-verified by `06_verify_labels_after_fix.py`: 0 invalid lines remaining.

**Also checked, no action needed:** 563 exact-duplicate image groups in the helmet+plate set, all within the same split (harmless); class imbalance ratio 2.3× (fine); image resolution 360×182–640×640, already Roboflow-preprocessed so no resizing needed before `model.train()`. Triple-riding dataset was clean on the first pass, zero issues.

**Both datasets are training-ready as-is.**

---

## 03. System architecture

The IoT layer (sensors → trigger → snapshot) is done and working. Everything after "snapshot saved" is the AI layer this plan builds.

**Flow:**

1. **PIR ×2 + HC-SR04** (ESP32 DevKit) continuously monitor the checkpoint. When measured speed exceeds the threshold (20 km/h placeholder) →
2. **TCP trigger → Python receiver**, which issues an **HTTP GET `/capture`** to the ESP32-CAM →
3. **ESP32-CAM** captures a frame (should be SVGA/800×600 once the QVGA bug is fixed) and the receiver saves it as a snapshot.
4. That single snapshot fans out **in parallel** to three vision components:
   - **Helmet / plate model** — fine-tuned YOLO26s → helmet status + plate box
   - **Rider-count model** — person+motorcycle detections → occupancy count
   - **Plate detector → crop → OCR** — YOLO26n plate box, cropped and passed to OCR
5. All three outputs converge on an **aggregator**, which combines them with the speed value already known from the trigger to decide whether a violation occurred.
6. The aggregator writes a **violation record + evidence image** to the log, optionally surfaced on a **dashboard** (stretch goal).

---

## 04. Models & datasets

### ✅ Architecture decision — two detectors, one reused OCR engine, not three custom models

**Model A (helmet + plate)** and **Model B (triple-riding)** stay separate: different label granularity (many small per-head boxes vs. one coarse per-scene box), different camera domains, and Model B's much smaller class count would get drowned out if trained jointly against Model A's ~120K boxes. Two nano/small YOLO models run sequentially still comfortably fit the ~1s capture-to-log budget, so there's no latency reason to force a merge.

**OCR is reused, never custom-trained.** A text-recognition model is a research-scale undertaking (sequence architectures trained on millions of text crops) already solved by EasyOCR/PaddleOCR — building a custom one would be reinventing a solved problem for no grading benefit. Pipeline: Model A finds the plate box → crop it → feed the crop to OCR, only once a violation is confirmed by Model A or B.

### Why YOLO, and which version

"CNN model" in the project brief means an object *detector*, not a plain classifier — bounding boxes are needed (where the helmet/plate/riders are), not just a whole-image label. Ultralytics' YOLO family is the standard choice: mature, Roboflow exports natively to its label format, trains end-to-end with a few lines of Python.

**YOLO26** (released Jan 2026) is Ultralytics' current stable release, alongside YOLO11 as the older mature option. Two YOLO26 features matter specifically here: **STAL** (Small-Target-Aware Label Assignment) improves detection of small objects — exactly what a helmet-on-a-head or a distant plate is — and Ultralytics reports **up to 43% faster CPU inference** than YOLO11n, relevant if the inference PC has no GPU. YOLO11 is a drop-in fallback with an identical Python API if any tooling snag comes up with YOLO26 — the dataset format is version-agnostic either way.

Use **nano/small** variants (`yolo26n.pt` / `yolo26s.pt`), not medium/large — Roboflow-scale custom datasets are typically hundreds to a few thousand images per class, so a bigger model overfits and trains/infers slower with no accuracy gain at that data volume.

### Model & data table

| Task | Model | Data source |
|---|---|---|
| **Helmet / rider status + plate localization** | Fine-tuned `yolo26s.pt` | `dataset/helmet_and_plate` — 4 classes (bike, helmet, no-helmet, number-plate), 24,936 / 3,810 / 2,678, training-ready. |
| **Triple riding (occupancy count)** | Fine-tuned `yolo26n.pt` (primary); pretrained `yolo26n.pt` zero-shot person+motorcycle counting rule (fallback/sanity-check) | `dataset/triple_riding` — 1 class, 3,475 / 331 / 165, ~20% negatives, training-ready. Fallback needs no dataset — COCO already has person/motorcycle. |
| **Plate text (OCR)** | **EasyOCR** first, benchmark **PaddleOCR** for the accuracy pass | Not a detector — a pretrained text recognizer run on the cropped plate region from Model A. EasyOCR is a one-line pip install and the fastest path to a working end-to-end pipeline. PaddleOCR generally scores higher on messy real-world crops but has a heavier/fussier install. Start with EasyOCR, swap in PaddleOCR once the rest is proven. |

Reference benchmark for the triple-riding task: [kashishparmar02/triple-rider-detection](https://github.com/kashishparmar02/triple-rider-detection) (YOLOv8s, 6,000+ images, 81.7% mAP) — a realistic accuracy target, not a drop-in dataset.

---

## 05. Phased implementation plan

*Sequenced by dependency, not calendar — adjust week estimates to the actual deadline. Phases 2–4 can run in parallel across three teammates.*

### Phase 0 — Environment setup (~1 day)

- **Training happens on Colab** (confirmed) — no local GPU needed. `Runtime > Change runtime type > T4 GPU`, then `!pip install -q ultralytics`.
- `dataset/helmet_and_plate` and `dataset/triple_riding` are already built, merged, and audited locally — **zip and upload them directly to Colab (or a Google Drive mount)** rather than re-pulling raw sources from Roboflow and re-running the merge scripts there.
- Locally: the project `.venv` already has what the dataset scripts need; add `opencv-python easyocr pandas` when Phase 5/6 work starts (that part runs on the laptop next to `cam_receiver_code.py` — only training happens on Colab).

### Phase 1 — Dataset audit & acquisition ✅ done

See [Dataset audit](#02-dataset-audit) and [Current status](#current-status--deliverables) for the full breakdown. Decision: two separate models — `dataset/helmet_and_plate` trains the helmet+plate detector, `dataset/triple_riding` trains a dedicated single-class triple-riding detector. Both leakage fixes and the negative-example fix are implemented and verified (`dataset_structuring/01`–`06`).

### Phase 2 — Fix camera capture resolution (~1–2 days) — ⬜ not started (confirmed still broken, 2026-10-19)

- Edit `CameraWebServer.ino`: remove/guard `s->set_framesize(s, FRAMESIZE_QVGA)`, or explicitly set `FRAMESIZE_SVGA` right before the `/capture` handler serves a frame. (The firmware currently lives zipped at `backup/smartTrafficViolation-main.zip` — extract it into a working location before editing.)
- Also extend the TCP trigger message in `trafficViolationSystem.ino` from `CAPTURE|date|time` to include the measured `speedKMPH`, and update `cam_receiver_code.py` to parse it — otherwise Phase 6's aggregator has no speed value to combine. (Found during the 2026-10-19 audit; see [Critical findings](#01-critical-findings).)
- Re-flash, re-run `cam_receiver_code.py`, confirm new snapshots are ≥800×600.
- While at the rig: capture 50–100 *real* photos from the actual mounting angle/height/lighting — helmet/no-helmet, 1/2/3 riders, day and dusk. Becomes the Phase 7 domain-gap fine-tuning set and an honest final accuracy check.

### Phase 3 — Train the helmet/rider detector — ✅ done

Trained both as **Option B** (our own MobileNetV3-Small + anchor-free detector) and as **YOLO26** (`yolo26s.pt`, Ultralytics) for comparison, on `dataset/helmet_and_plate`. Test-split: Option B mAP50 0.860 / mAP50-95 0.570; YOLO26 mAP50 0.945 / mAP50-95 0.707 (full numbers in `analysis/summary.txt`, checkpoints in `training/runs/helmet_plate_optionb/` and `training/runs/yolo26/helmet_plate/`). The 81.7% mAP kashishparmar02 reference was for triple-riding, not this task, but both runs land in the same ballpark.

### Phase 4 — Triple riding detection — ✅ done (fine-tuned); zero-shot fallback not built

- **Primary — done:** fine-tuned both Option B and `yolo26n.pt` on `dataset/triple_riding`. Test-split: Option B mAP50 0.953 / mAP50-95 0.667; YOLO26 mAP50 0.963 / mAP50-95 0.786.
- **Zero-shot fallback — not implemented.** The person-in-expanded-motorcycle-box counting rule described below is still just a plan snippet, not code. Low priority now that the fine-tuned models work well, but still worth having as a sanity cross-check if time allows.
- Still undecided as a team: does a small child count toward the rider limit?

### Phase 5 — Number plate detection + OCR — 🟡 engine works; blocked on the dataset, not the code

- Plate detection — done (it's the `number-plate` class in the helmet_plate model above).
- `ocr/plate_ocr.py` crops the plate box (~10% padding, `--min-size` filter added 2026-10-19), preprocesses (3x `cv2.resize`/`INTER_CUBIC`, grayscale, CLAHE), runs EasyOCR, and applies the regex + confusable-character correction (O↔0, I↔1, B↔8, S↔5, Z↔2) — all implemented, runs without errors, and (after the size filter) actually returns text.
- **But:** the plate boxes big enough to be legible are almost all from the Nepali plate dataset (wrong script), not the Indian one, so a real accuracy number still doesn't exist. Ground truth in `ocr/labels.csv` is still unfilled. See [Critical findings](#01-critical-findings) for the detail.
- **PaddleOCR — tried, ruled out on this machine (2026-10-19).** Installed cleanly (`paddlepaddle`, `paddleocr`, `paddlex`), but *importing* it crashed OS-level resources twice (process fork failure, then `OSError: [WinError 1450] Insufficient system resources` / "paging file too small") — two independent failures from the same action is enough signal not to retry on this hardware. **Uninstalled** (it was also silently shadowing the project's `opencv-python` with `opencv-contrib-python`, which has been fixed — see `pip check`). If PaddleOCR needs to be retried, do it on a machine with more headroom, in isolation from everything else running.
- Next real option: find or add Indian-plate images that are both large enough to read and actually in the right script — the current `dataset/helmet_and_plate` test split may just not have enough of them (see [Models & datasets](#04-models--datasets)).

### Phase 6 — Pipeline integration — ✅ AI side done (2026-10-19); IoT-side wiring remains

- **Done:** `pipeline/aggregator.py`'s `Pipeline` class runs both detectors + plate OCR on one image, combines them (plus an optional speed value) into one violation record, saves an annotated evidence image, and logs a row to `violations.db` — tested end-to-end on real test-split images (see [Current status](#current-status--deliverables)). The evidence drawing reuses `training/show_predictions.py`'s `draw()`/`violation_text()` rather than duplicating it, exactly as originally planned.
- **Remaining — IoT side, handed off:** `cam_receiver_code.py` still needs to call `Pipeline.process(...)` after saving each snapshot, and the ESP32 trigger message needs to start sending the measured speed. Both are spelled out step by step, with exact line numbers and a code snippet to paste in, in **`IOT_INTEGRATION_HANDOFF.md`** (new doc, separate from this plan) — written so whoever does the IoT fixes doesn't need to read the AI code first.
- `violations.db` schema matches what was planned: `id, timestamp, plate_number, violation_types, speed_kmph, raw_image_path, annotated_image_path, confidences` (plus `helmet_message`/`triple_message` for convenience).

### Phase 7 — Field testing & calibration (~3–5 days) — ⬜ not started

- Real runs with an actual two-wheeler at the mounted rig: helmet/no-helmet, 1/2/3 riders, varied speed.
- Recalibrate the 20 km/h overspeed threshold against real measured speeds — in **both** `trafficViolationSystem.ino` and `pipeline/aggregator.py`'s `OVERSPEED_THRESHOLD_KMPH`, together.
- Feed misclassified real captures back into training and retrain if time allows.

### Phase 8 — Dashboard — ✅ done (2026-10-19), ahead of schedule

No longer a stretch goal — built alongside the aggregator since the two are only useful together for the demo. `dashboard/app.py` (Streamlit, `.venv\Scripts\python.exe -m streamlit run dashboard\app.py`): one page, two tabs — "Run on an image" (upload or point at a file, run both models + OCR, see the annotated result) and "Violation log" (every row in `violations.db`, newest first, with total/no-helmet/triple-riding counts and an evidence-image viewer). Tested: boots cleanly, serves HTTP 200, reads the same `violations.db` the aggregator writes.

### Phase 9 — Documentation & report (final days) — ⬜ not started

Final metrics per model (mAP, OCR accuracy %), the architecture diagram, a limitations section (lighting, camera-angle sensitivity, Indian-plate-format assumption, single-ultrasonic speed estimation noise), and future work. (Distinct from the already-built `AI Model Architecture, Feature Engineering & Dataset Design.pptx`, which covers model/dataset design, not final results.)

---

## 06. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Roboflow dataset's camera angle/distance differs from the rig's real mounting position | Phase 2's 50–100 real captures feed a small fine-tuning round in Phase 7 to close the gap. |
| Plate OCR accuracy still weak even after the resolution fix (motion blur at speed, night lighting) | Regex + confusable-character correction as a safety net (Phase 5); consider a short exposure/gain tweak on the OV2640. |
| Helmet dataset's train/valid/test leakage inflates reported mAP | **Resolved** — re-split by source stem, verified zero leakage. |
| Triple-riding model over-fires on any populated motorcycle (no negatives in the dataset) | **Resolved** — ~20% negative images added before training. |
| PaddleOCR install friction on Windows | **Materialized** — installed fine, but crashed OS resources on import (2026-10-19) and was uninstalled. Shipping with EasyOCR only; retry PaddleOCR on different hardware if it's still wanted. |
| Single ultrasonic sensor gives noisy speed readings | Out of scope for the AI plan, but recalibrate the threshold on-site (Phase 7) rather than trusting the current 20 km/h placeholder. |

---

## 07. Code reference

### Dataset pipeline (implemented, in `dataset_structuring/`)

The dataset-building code originally sketched in this plan has since been fully implemented, corrected, and verified — the authoritative versions live in `dataset_structuring/`, not here:

| Script | Purpose |
|---|---|
| `01_merge_triple_riding.py` | Builds `dataset/triple_riding` from `backup/triple dataset` (positives) + `backup/helmet and triple riding dataset` (negatives), deduped by source photo. |
| `02_merge_helmet_plate.py` | Builds `dataset/helmet_and_plate` — re-splits `backup/helmet and number plate dataset` by source image, folds in `backup/Number Plate.v3i.yolo26`. |
| `03_verify_helmet_plate_merge.py` | Independent leakage + class-count check, re-derived from disk. |
| `04_audit_final_datasets.py` | Full byte-level integrity audit (corruption, orphans, label validity, duplicates, resolution, class balance). |
| `05_fix_polygon_labels.py` | Converts the 328 polygon-format label lines the audit found into standard boxes. Idempotent. |
| `06_verify_labels_after_fix.py` | Fast final check that 0 invalid label lines remain. |

`01`/`02` are deterministic (seeded) but will delete and rebuild their target folder if re-run — don't run them unless intentionally rebuilding a dataset from its raw `backup/` source.

### Not yet implemented — reference snippets for upcoming phases

**Train (Phase 3):**

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")            # or yolo26n.pt for the triple-riding model
results = model.train(
    data="dataset/helmet_and_plate/data.yaml",
    epochs=100,
    imgsz=640,
    patience=20,                      # early stop if val mAP plateaus
    device=0,                         # Colab GPU — Runtime > Change runtime type > T4 GPU
)
# best weights land in runs/detect/train/weights/best.pt — download this file, that's the deliverable
```

**Zero-shot rider counting — fallback path (Phase 4):**

```python
from ultralytics import YOLO

coco_model = YOLO("yolo26n.pt")       # pretrained, no fine-tuning
res = coco_model(image, classes=[0, 3])  # 0=person, 3=motorcycle in COCO

riders = 0
for moto_box in boxes_of(res, "motorcycle"):
    expanded = expand_box(moto_box, 1.2)
    riders = sum(1 for p in boxes_of(res, "person") if center_in(p, expanded))

triple_riding = riders >= 3
```

**Plate OCR with format validation (Phase 5):**

```python
import re, easyocr

reader = easyocr.Reader(["en"])
CONFUSABLE = str.maketrans({"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2"})
PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")

def read_plate(crop):
    raw = "".join(t for _, t, _ in reader.readtext(crop)).upper().replace(" ", "")
    if PLATE_RE.match(raw):
        return raw
    fixed = raw.translate(CONFUSABLE)
    return fixed if PLATE_RE.match(fixed) else raw  # fall back, flag low-confidence
```

---

## Sources

Sources checked while writing this plan:

- [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/) — model variants, STAL, CPU benchmark vs YOLO11n
- [Ultralytics — models supported](https://docs.ultralytics.com/models)
- [Roboflow Universe — triple-ride-detection (5-class unified)](https://universe.roboflow.com/harshas-workspace/triple-ride-detection)
- [Roboflow Universe — triple-riding-detection-kbulg](https://universe.roboflow.com/pran/triple-riding-detection-kbulg/dataset/1)
- [GitHub — kashishparmar02/triple-rider-detection](https://github.com/kashishparmar02/triple-rider-detection)
