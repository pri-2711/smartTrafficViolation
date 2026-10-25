# IoT Integration — Next Steps

**For whoever is fixing the IoT/firmware side.** This is everything needed to plug the finished AI
section (two trained detectors + OCR + the aggregator + the dashboard) into the real capture rig.
You don't need to understand the AI code internals — just the contract described below.

Written 2026-10-19, updated 2026-10-20. The AI-side pieces this doc assumes exist: `pipeline/aggregator.py` (the
combined inference pipeline) and `dashboard/app.py` (the log viewer) — both already built and
tested as of this date. If you haven't already, read the project root's **`README.md`** first — it
maps out the whole repo in two minutes. Full background and history on the AI side: `AI_Implementation_Plan.md`.

---

## 1. What you're connecting to what

```
ESP32 trigger  →  TCP  →  cam_receiver_code.py  →  HTTP GET /capture  →  ESP32-CAM
                                    │
                                    ▼ (after saving the snapshot)
                          pipeline.aggregator.Pipeline.process(image_path, speed_kmph)
                                    │
                                    ▼
                    violations.db  +  violations/evidence/<name>.png
                                    │
                                    ▼
                   dashboard/app.py  (streamlit run — reads the same violations.db)
```

Everything left of "`cam_receiver_code.py`" is the IoT side. Everything from `Pipeline.process(...)`
onward is already built and already works — you're only wiring the receiver script into it.

## 2. The three things to fix, in order

### 2a. Fix the ESP32-CAM resolution bug

The firmware currently lives zipped at `backup/smartTrafficViolation-main.zip` (never extracted into
the repo) — extract it first, e.g. into a new `firmware/` folder, so it's actually editable and
version-controlled.

In `FINAL SYSTEM/CameraWebServer/CameraWebServer.ino`, line ~93:

```cpp
s->set_framesize(s, FRAMESIZE_QVGA);   // <- remove this line, or change it to FRAMESIZE_SVGA
```

This line silently drops every `/capture` frame to 320×240 regardless of the UXGA/SVGA setup earlier
in the file. Re-flash, re-run `cam_receiver_code.py`, confirm new snapshots are ≥800×600 (check file
size and open one). This matters a lot: at 320×240 a number plate is often under 100px wide, which is
part of why OCR is currently failing (see the OCR section of `AI_Implementation_Plan.md`).

### 2b. Send the measured speed to the receiver

Right now the trigger message is just `CAPTURE|date|time` — the speed value the ESP32 already
computes (`speedKMPH` in `FINAL SYSTEM/trafficViolationSystem/trafficViolationSystem.ino`, around
line 445) is only printed to Serial, never sent over TCP.

Extend the message format (same file, around lines 169 and 178) to:

```
CAPTURE|<date>|<time>|<speed_kmph>
```

e.g. `CAPTURE|19_10_2026|15_04_45|42.5`.

### 2c. Wire the receiver into the aggregator

In `cam_receiver_code.py`:

1. Parse the new 4th field from the trigger message as a float (`speed_kmph`). Fall back to `None` if
   it's missing, so the script doesn't break on an old-format message during testing.
2. Add these two lines **once, before the `while True:` loop** (not inside it — loading the models
   takes a few seconds, so only do it once per process):

   ```python
   import sys
   sys.path.insert(0, r"C:\Parin\Navrachna\7th Sem\Artificial Intelligence of Things [CSE712]\pipeline")
   from aggregator import Pipeline

   pipeline = Pipeline()   # defaults: both tasks on their YOLO26 checkpoint, conf 0.25
   ```

3. Right after the existing `print("IMAGE SAVED:")` block (i.e. once `filepath` holds the saved
   snapshot), add:

   ```python
   record = pipeline.process(filepath, speed_kmph=speed_kmph)
   print("violations:", record["violation_types"] or "none")
   print("plate:", record["plate_number"] or "(none read)")
   ```

That's the entire integration. `Pipeline.process(...)` already does everything else: runs both
detectors, runs OCR on the plate crop, saves an annotated evidence image to
`violations/evidence/`, and inserts a row into `violations.db` at the project root. Nothing else
needs to be written — the dashboard reads that same `violations.db` automatically.

## 3. The contract (what you can rely on, without reading the rest of the AI code)

```python
from aggregator import Pipeline

pipeline = Pipeline(
    helmet_model="yolo26",    # or "option_b"
    triple_model="yolo26",    # or "option_b"
    conf=0.25,
    device="",                 # "" = auto-detect GPU/CPU, "cpu" to force CPU
    db_path=None,               # None = project_root/violations.db
)                               # construct ONCE — loads both models (a few seconds)

record = pipeline.process(image_path, speed_kmph=None)   # call PER snapshot
```

`record` is a dict with (at least) these keys:

| Key | Type | Meaning |
|---|---|---|
| `violation_types` | `list[str]` | any of `"no_helmet"`, `"triple_riding"`, `"overspeed"` — empty list means no violation |
| `plate_number` | `str` | OCR'd plate text, `""` if no plate was read |
| `plate_valid_format` | `bool` | whether `plate_number` matches the Indian plate regex |
| `speed_kmph` | `float \| None` | echoes whatever you passed in |
| `annotated_image_path` | `str` | path to the saved evidence PNG |
| `id` | `int` | the row id just written to `violations.db` |

You don't need to do anything else with `record` — it's already logged. Printing a couple of fields
(as in step 2c above) is just for your own sanity-checking at the rig.

## 4. Testing checklist once wired up

- [ ] Trigger a real capture (walk/ride past the rig). Confirm the saved snapshot is ≥800×600.
- [ ] Confirm a new row appears in `violations.db` (`dashboard/app.py`'s "Violation log" tab, or
      `sqlite3 violations.db "select * from violations order by id desc limit 1;"`).
- [ ] Confirm `violations/evidence/<timestamp>_<name>.png` exists and has visible, correctly labelled
      boxes.
- [ ] Run `.venv\Scripts\python.exe -m streamlit run dashboard\app.py` and confirm the new row shows
      up there too, with a working evidence-image preview.
- [ ] Trigger a few more passes with different speeds, with and without a helmet, to confirm
      `violation_types` reflects what actually happened each time.

## 5. One thing to recalibrate together, not separately

`OVERSPEED_THRESHOLD_KMPH = 20.0` is a placeholder in **two** places: the firmware
(`trafficViolationSystem.ino`) and `pipeline/aggregator.py`. Once real speeds are measured at the
mounted rig (Phase 7 in `AI_Implementation_Plan.md`), update both constants to the same value —
don't just change one.
