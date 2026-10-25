"""Analysis of finished runs: per-epoch log, error-rate charts, and test comparison.

  python training/analyze.py

Reads training/runs/ (Option B log.csv and results.json, YOLO26 results.csv, compare_<task>.json).
Writes project_root/analysis/: epoch_log.csv (one row per model per epoch), summary.txt, and PNG charts.
Error rate here means 1 - mAP50 (the share of objects the detector gets wrong at IoU 0.5).
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ANALYSIS_DIR, RUNS_DIR, TASKS

OUT = ANALYSIS_DIR
MODELS = {"Option B": "optionb", "YOLO26": "yolo26"}
COLORS = {"Option B": "tab:blue", "YOLO26": "tab:orange"}


def _rows(path):
    """CSV rows with stripped header names (Ultralytics pads its column names with spaces)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        return [dict(zip(header, [v.strip() for v in row])) for row in reader if row]


def _float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_option_b(task):
    path = RUNS_DIR / f"{task}_optionb" / "log.csv"
    if not path.exists():
        return []
    out = []
    for r in _rows(path):
        out.append({
            "epoch": int(r["epoch"]) + 1, "lr": _float(r["lr"]), "train_loss": _float(r["loss"]),
            "val_mAP50": _float(r["val_mAP50"]), "val_mAP50_95": _float(r["val_mAP50_95"]),
            "epoch_seconds": _float(r["seconds"]),
        })
    return out


def load_yolo26(task):
    path = RUNS_DIR / "yolo26" / task / "results.csv"
    if not path.exists():
        return []
    out, prev_time = [], 0.0
    for r in _rows(path):
        t = _float(r.get("time"))
        seconds = None
        if t is not None:
            # Ultralytics' time counter restarts from 0 when a run is resumed, so the epoch after a resume is just t.
            seconds = round(t - prev_time, 1) if t >= prev_time else round(t, 1)
            prev_time = t
        out.append({
            "epoch": int(float(r["epoch"])), "lr": _float(r.get("lr/pg0")),
            "train_loss": sum(_float(r.get(k), 0.0) for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss")),
            "val_mAP50": _float(r.get("metrics/mAP50(B)")), "val_mAP50_95": _float(r.get("metrics/mAP50-95(B)")),
            "epoch_seconds": seconds,
        })
    return out


def load_test(task):
    path = RUNS_DIR / f"compare_{task}.json"
    if not path.exists():
        return {}
    res = json.loads(path.read_text(encoding="utf-8"))
    return {label: res[key] for label, key in (("Option B", "option_b"), ("YOLO26", "yolo26")) if key in res}


def write_epoch_log(data):
    """One CSV for both models, so the per-epoch numbers can be opened in Excel or pasted into the report."""
    path = OUT / "epoch_log.csv"
    fields = ["task", "model", "epoch", "lr", "train_loss", "val_mAP50", "val_mAP50_95", "error_rate", "epoch_seconds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (task, model), rows in data.items():
            for r in rows:
                err = None if r["val_mAP50"] is None else round(1 - r["val_mAP50"], 4)
                w.writerow({"task": task, "model": model, "error_rate": err, **r})
    return path


def plot_curves(data):
    """Per-epoch curves: validation mAP50 (error = 1 - mAP50), loss, learning rate and epoch time."""
    tasks = list(TASKS)
    charts = {
        "error_rate": ("Error rate (1 - val mAP50), lower is better", "error rate"),
        "val_map": ("Validation mAP50-95, higher is better", "mAP50-95"),
        "loss": ("Training loss", "loss"),
        "learning_rate": ("Learning rate", "lr"),
        "epoch_time": ("Seconds per epoch", "seconds"),
    }
    for key, (title, ylabel) in charts.items():
        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            for model in MODELS:
                rows = [r for r in data.get((task, model), []) if r["epoch"] is not None]
                if key == "error_rate":
                    pts = [(r["epoch"], 1 - r["val_mAP50"]) for r in rows if r["val_mAP50"] is not None]
                elif key == "val_map":
                    pts = [(r["epoch"], r["val_mAP50_95"]) for r in rows if r["val_mAP50_95"] is not None]
                elif key == "loss":
                    pts = [(r["epoch"], r["train_loss"]) for r in rows if r["train_loss"] is not None]
                elif key == "learning_rate":
                    pts = [(r["epoch"], r["lr"]) for r in rows if r["lr"] is not None]
                else:
                    pts = [(r["epoch"], r["epoch_seconds"]) for r in rows if r["epoch_seconds"] is not None]
                if pts:
                    ax.plot(*zip(*pts), label=model, color=COLORS[model])
            ax.set_title(f"{task}: {title}", fontsize=10)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"{key}.png", dpi=150)
        plt.close(fig)


def plot_test(tests):
    """Bar chart of test-split scores: mAP50, mAP50-95, recall@100, plus the error rate (1 - mAP50)."""
    metrics = [("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"), ("recall_100", "recall@100")]
    tasks = [t for t in TASKS if tests.get(t)]
    if not tasks:
        return None
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        width = 0.35
        xs = range(len(metrics))
        for i, (label, res) in enumerate(tests[task].items()):
            vals = [res[k] for k, _ in metrics]
            bars = ax.bar([x + (i - 0.5) * width for x in xs], vals, width, label=label, color=COLORS[label])
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(list(xs), [name for _, name in metrics])
        ax.set_ylim(0, 1)
        ax.set_title(f"{task}: test split", fontsize=10)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "test_comparison.png", dpi=150)
    plt.close(fig)


def write_summary(data, tests):
    lines = []
    for task in TASKS:
        for model in MODELS:
            rows = [r for r in data.get((task, model), []) if r["val_mAP50"] is not None]
            if not rows:
                continue
            first, best = rows[0], max(rows, key=lambda r: r["val_mAP50"])
            e0, e1 = 1 - first["val_mAP50"], 1 - best["val_mAP50"]
            drop = (e0 - e1) / e0 * 100 if e0 > 0 else 0.0
            times = [r["epoch_seconds"] for r in rows if r["epoch_seconds"] is not None]
            lines.append(f"{task} / {model}")
            lines.append(f"  epochs run: {len(rows)}")
            lines.append(f"  error rate (1 - mAP50): epoch 1 {e0:.3f} -> best {e1:.3f} (epoch {best['epoch']}), "
                         f"down {drop:.1f}%")
            if times:
                lines.append(f"  time per epoch: mean {sum(times) / len(times) / 60:.1f} min, "
                             f"total {sum(times) / 3600:.2f} h")
            lines.append("")
    for task, res in tests.items():
        for label, m in res.items():
            lines.append(f"test {task} / {label}: mAP50 {m['mAP50']:.3f}, mAP50-95 {m['mAP50_95']:.3f}, "
                         f"error rate {1 - m['mAP50']:.3f}, recall@100 {m['recall_100']:.3f}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, tests = {}, {}
    for task in TASKS:
        data[(task, "Option B")] = load_option_b(task)
        data[(task, "YOLO26")] = load_yolo26(task)
        tests[task] = load_test(task)
    if not any(data.values()):
        raise SystemExit(f"no training logs found under {RUNS_DIR}")
    write_epoch_log(data)
    plot_curves(data)
    plot_test(tests)
    print(write_summary(data, tests))
    print(f"\nwrote charts and logs to {OUT}")


if __name__ == "__main__":
    main()
