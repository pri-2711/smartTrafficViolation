"""Shared paths, defaults and device selection. Edit values here, not in every script."""
import os
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")  # keep the console output to progress bars and results

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent
# The Colab notebook copies dataset/ to local disk (much faster to read than Drive) and sets this variable.
DATASET_DIR = Path(os.environ.get("AIOT_DATASET_DIR") or PROJECT_DIR / "dataset")
RUNS_DIR = TRAINING_DIR / "runs"
ANALYSIS_DIR = PROJECT_DIR / "analysis"  # charts and logs from analyze.py, kept apart from the raw runs

# The two detectors we train. Each one is a separate model (see AI_Implementation_Plan.md).
TASKS = {
    "helmet_plate": {
        "data": DATASET_DIR / "helmet_and_plate" / "data.yaml",
        "description": "Model A: helmet / no-helmet / bike / number-plate",
    },
    "triple_riding": {
        "data": DATASET_DIR / "triple_riding" / "data.yaml",
        "description": "Model B: triple-riding occupancy",
    },
}

YOLO_WEIGHTS = "yolo26s.pt"  # comparison baseline; Option B is the main model

DEFAULTS = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "freeze_epochs": 10,
    "patience": 20,
    "workers": 4,
    "seed": 0,
}


def pick_device(requested=""):
    """'' = auto (CUDA GPU if present, otherwise CPU). Accepts '0', 'cpu', 'cuda:0', etc."""
    if requested in ("", None):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.isdigit():
        return torch.device(f"cuda:{requested}")
    return torch.device(requested)
