"""
Step 1 of the dataset pipeline.

Builds dataset/triple_riding from two raw sources:
  - backup/triple dataset                     (positives, copied as-is)
  - backup/helmet and triple riding dataset    (negatives: its 'No Violation' images only)

Negatives are deduplicated by source photo (Roboflow's per-copy hash stripped)
before being assigned to a split, so no single source image's copies end up
spread across train/valid/test.

Run from anywhere; paths are hardcoded below. Re-running this deletes and
rebuilds dataset/triple_riding from scratch (random.seed(42) makes the split
assignment deterministic, so the result is reproducible).
"""
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\Parin\Navrachna\7th Sem\Artificial Intelligence of Things [CSE712]")
TRIPLE_SRC = ROOT / "backup" / "triple dataset"
VIOLATION_SRC = ROOT / "backup" / "helmet and triple riding dataset"
OUT = ROOT / "dataset" / "triple_riding"

SPLITS = ["train", "valid", "test"]
STEM_RE = re.compile(r"_jpg\.rf\.[0-9a-f]+$", re.IGNORECASE)

random.seed(42)


def stem_of(path: Path) -> str:
    return STEM_RE.sub("", path.stem)


def reset_output():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_positives():
    """Copy every triple-riding image+label as-is, split preserved."""
    counts = {}
    for split in SPLITS:
        img_dir = TRIPLE_SRC / split / "images"
        lbl_dir = TRIPLE_SRC / split / "labels"
        n = 0
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / f"{img.stem}.txt"
            shutil.copy2(img, OUT / split / "images" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, OUT / split / "labels" / lbl.name)
            else:
                (OUT / split / "labels" / f"{img.stem}.txt").write_text("")
            n += 1
        counts[split] = n
    return counts


def collect_negative_candidates():
    """
    From the violation dataset, find images with NO 'Violation' (class 1) box —
    i.e. images that are safe to use as 'not triple riding' background examples.
    Group by source stem first so we never pull multiple augmented copies of the
    same source photo into different splits (that would reintroduce leakage).
    """
    groups = defaultdict(list)  # stem -> list of (image_path)
    for split in SPLITS:
        img_dir = VIOLATION_SRC / split / "images"
        lbl_dir = VIOLATION_SRC / split / "labels"
        for lbl in lbl_dir.glob("*.txt"):
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            has_violation = any(l.split()[0] == "1" for l in lines)
            if has_violation:
                continue
            img = img_dir / f"{lbl.stem}.jpg"
            if img.exists():
                groups[stem_of(img)].append(img)
    return groups


def pick_negatives(groups, targets):
    """One representative image per unique source stem, assigned to one split each."""
    stems = list(groups.keys())
    random.shuffle(stems)

    assigned = {s: [] for s in SPLITS}
    idx = 0
    for split in SPLITS:
        need = targets[split]
        got = 0
        while got < need and idx < len(stems):
            stem = stems[idx]
            idx += 1
            img = groups[stem][0]  # just one representative copy of this source photo
            assigned[split].append(img)
            got += 1
    return assigned


def write_negatives(assigned):
    counts = {}
    for split, imgs in assigned.items():
        n = 0
        for img in imgs:
            dst_name = f"neg_{img.name}"
            shutil.copy2(img, OUT / split / "images" / dst_name)
            (OUT / split / "labels" / f"neg_{img.stem}.txt").write_text("")
            n += 1
        counts[split] = n
    return counts


def write_data_yaml():
    (OUT / "data.yaml").write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "\n"
        "nc: 1\n"
        "names: ['tripleriding']\n"
    )


def write_readme(pos_counts, neg_counts):
    lines = [
        "Merged triple-riding dataset",
        "=============================",
        "",
        "Positives copied as-is from: triple dataset (ayuwidyaa/deteksi-triple-riding, v3)",
        "Negatives sourced from: helmet and triple riding dataset (msuiit-esuvy/helmet-dataset-jahk2, v3),",
        "  restricted to images with zero 'Violation' (class 1) boxes, deduplicated by source photo",
        "  (only one representative copy per unique source image was pulled, and each source photo",
        "  was assigned to exactly one split, so no leakage was reintroduced).",
        "",
        "Split counts (positives + negatives = total):",
    ]
    for split in SPLITS:
        pos = pos_counts[split]
        neg = neg_counts[split]
        total = pos + neg
        pct = (neg / total * 100) if total else 0
        lines.append(f"  {split}: {pos} positive + {neg} negative = {total} total ({pct:.1f}% negative)")
    lines.append("")
    lines.append("Label format: standard YOLO txt (class x_center y_center width height, normalized).")
    lines.append("Negative images have an empty .txt label file (Ultralytics treats this as background).")
    (OUT / "README.merge.txt").write_text("\n".join(lines))


def verify_no_leakage():
    stems = {s: set() for s in SPLITS}
    for split in SPLITS:
        for img in (OUT / split / "images").glob("*.jpg"):
            stems[split].add(stem_of(img))
    tv = stems["train"] & stems["valid"]
    tt = stems["train"] & stems["test"]
    vt = stems["valid"] & stems["test"]
    return stems, tv, tt, vt


def main():
    reset_output()
    pos_counts = copy_positives()
    print("Positives copied:", pos_counts)

    groups = collect_negative_candidates()
    print(f"Unique candidate negative source images found: {len(groups)}")

    targets = {
        "train": round(pos_counts["train"] * 0.25),
        "valid": round(pos_counts["valid"] * 0.25),
        "test": round(pos_counts["test"] * 0.25),
    }
    print("Negative targets (~20% of final split):", targets)

    assigned = pick_negatives(groups, targets)
    neg_counts = write_negatives(assigned)
    print("Negatives written:", neg_counts)

    write_data_yaml()
    write_readme(pos_counts, neg_counts)

    stems, tv, tt, vt = verify_no_leakage()
    print("\nFinal split sizes (unique source images):",
          {k: len(v) for k, v in stems.items()})
    print(f"Leakage check -> train&valid={len(tv)} train&test={len(tt)} valid&test={len(vt)}")

    for split in SPLITS:
        n_img = len(list((OUT / split / "images").glob("*.jpg")))
        n_lbl = len(list((OUT / split / "labels").glob("*.txt")))
        print(f"{split}: images={n_img} labels={n_lbl}")


if __name__ == "__main__":
    main()
