# Training

*Part of the [project root README](../README.md) — read that first for the full repo map.*

Two detectors, each trained with our own model (**Option B**) and with **YOLO26** as the comparison baseline.

| Task | Dataset | Classes | Option B | YOLO26 |
|---|---|---|---|---|
| `helmet_plate` (Model A) | `dataset/helmet_and_plate` | bike, helmet, no-helmet, number-plate | yes | `yolo26s.pt` |
| `triple_riding` (Model B) | `dataset/triple_riding` | tripleriding | yes | `yolo26n.pt` |

## Layout

```
training/
├── config.py            paths, task list, default hyperparameters, device selection
├── yolo_data.py         YOLO-format dataset loader, letterbox, augmentation
├── option_b/            Option B network and training math
│   ├── model.py         MobileNetV3-Small backbone, PAN-lite neck, decoupled head
│   ├── grids.py         grid centers, strides, object-size ranges, box decoding
│   ├── assign.py        static center-sampling label assignment
│   ├── loss.py          BCE cls + BCE objectness + CIoU box loss
│   └── postprocess.py   score threshold + NMS for inference
├── train_option_b.py    trains Option B (writes runs/<task>_optionb/)
├── train_yolo26.py      trains the YOLO26 baseline (writes runs/yolo26/<task>/)
├── evaluate.py          scores both models on the same test split and metric
├── analyze.py           per-epoch charts and summary.txt from finished runs (writes project_root/analysis/)
├── show_predictions.py  draws both models' boxes on test images, or on one image you pick yourself (--pick)
├── train_models.ipynb   the whole pipeline in one notebook (local or Colab)
└── requirements.txt
```

Downstream of this folder: `pipeline/aggregator.py` runs both trained detectors plus plate OCR together on one
image and logs a violation record; `dashboard/app.py` is the Streamlit UI for that. See the project root's
`AI_Implementation_Plan.md`.

## Run locally

From the project root, with the project venv active:

```powershell
.venv\Scripts\python.exe -m pip install -r training\requirements.txt
.venv\Scripts\python.exe training\train_option_b.py --task helmet_plate
.venv\Scripts\python.exe training\train_option_b.py --task triple_riding --epochs 60
.venv\Scripts\python.exe training\train_yolo26.py --task helmet_plate --weights yolo26s.pt
.venv\Scripts\python.exe training\train_yolo26.py --task triple_riding --weights yolo26n.pt
.venv\Scripts\python.exe training\evaluate.py --task helmet_plate --split test `
    --option-b training\runs\helmet_plate_optionb\weights\best.pt `
    --yolo training\runs\yolo26\helmet_plate\weights\best.pt
```

No CUDA GPU? Everything still runs on CPU (`--device cpu`), but a full 25k-image run takes days. Use a GPU for real training, or run a quick check with `--limit 200 --epochs 2`.

For a GPU locally, install the CUDA build of PyTorch first (see https://pytorch.org/get-started/locally/), then the rest of `requirements.txt`.

## Run on Google Colab

1. Zip the dataset first (there's no standing `dataset.zip` in the repo -- it's a one-time upload artifact, not something to keep around locally): from the project root,
   `Compress-Archive -Path dataset\helmet_and_plate, dataset\triple_riding -DestinationPath dataset.zip` (PowerShell), or `zip -r dataset.zip dataset/helmet_and_plate dataset/triple_riding` (bash). Upload `training/` and that `dataset.zip` to `My Drive/Parin/`. Everything for the project lives in that one folder. The notebook unzips the dataset to the Colab disk on first run, since reading images straight from Drive is very slow. Delete the local `dataset.zip` again once it's uploaded -- `dataset/` is already there unzipped and is what everything else in the repo actually reads.
2. Open `training/train_models.ipynb` in Colab.
3. Runtime → Change runtime type → **T4 GPU**.
4. Run the cells top to bottom. Section 1 mounts Drive and sets the project path.
5. Before the session ends, download `training/runs/` back to your laptop.

## Outputs

- `runs/<task>_optionb/weights/best.pt`: Option B best checkpoint (chosen by validation mAP)
- `runs/<task>_optionb/log.csv`: per-epoch loss, learning rate, validation mAP
- `runs/<task>_optionb/results.json`: final validation and test scores
- `runs/yolo26/<task>/weights/best.pt`: YOLO26 best checkpoint
- `runs/compare_<task>.json`: head-to-head test scores from `evaluate.py`
- `analysis/` (project root): from `analyze.py` (also run by the last notebook section): `epoch_log.csv` (per-epoch LR, loss, mAP, error rate, seconds for both models), `summary.txt`, and PNG charts (`error_rate`, `val_map`, `loss`, `learning_rate`, `epoch_time`, `test_comparison`)

## Method notes

- Both models use the same 640 px letterbox, the same train/val/test split, and the same mAP code. The only differences are the architecture and the training recipe.
- Option B: backbone frozen for the first 10 epochs, then fine-tuned end to end. AdamW, lr 1e-3 with warmup and cosine decay. Gradient clipping at 10. Mixed precision on GPU.
- YOLO26 uses Ultralytics' default recipe (mosaic augmentation, its own schedule). It is a strong baseline, not a tuned competitor, so the comparison reports both models as trained with their default settings.
