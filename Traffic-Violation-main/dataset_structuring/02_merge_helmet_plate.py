"""
Step 2 of the dataset pipeline.

Builds dataset/helmet_and_plate from two raw sources:
  - backup/helmet and number plate dataset   (re-split by source image to remove
                                               the train/valid/test leakage found
                                               in the original Roboflow export)
  - backup/Number Plate.v3i.yolo26           (Nepali plates, folded in as extra
                                               number-plate-class examples, class
                                               remapped 0 -> 3)

Run from anywhere; paths are hardcoded below. Re-running this deletes and
rebuilds dataset/helmet_and_plate from scratch (random.seed(42) makes the
re-split deterministic, so the result is reproducible). This is the raw merge
only -- it still contains the polygon-format labels fixed in step 5.
"""
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\Parin\Navrachna\7th Sem\Artificial Intelligence of Things [CSE712]")
HNP_SRC = ROOT / "backup" / "helmet and number plate dataset"
NP_SRC = ROOT / "backup" / "Number Plate.v3i.yolo26"
OUT = ROOT / "dataset" / "helmet_and_plate"

SPLITS = ["train", "valid", "test"]
STEM_RE = re.compile(r"_jpg\.rf\.[0-9a-f]+$", re.IGNORECASE)
SPLIT_RATIOS = {"train": 0.85, "valid": 0.075, "test": 0.075}
NAMES = ["bike", "helmet", "no-helmet", "number-plate"]  # index 3 = number-plate

random.seed(42)


def stem_of(path: Path) -> str:
    return STEM_RE.sub("", path.stem)


def reset_output():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)


def resplit_hnp():
    """Re-split the helmet+plate dataset by SOURCE image, fixing the
    train/valid/test leakage found in the earlier audit (same source photo's
    augmented copies were previously scattered across splits)."""
    groups = defaultdict(list)  # stem -> list of (split, img_path)
    for split in SPLITS:
        img_dir = HNP_SRC / split / "images"
        for img in img_dir.glob("*.jpg"):
            groups[stem_of(img)].append((split, img))

    stems = list(groups.keys())
    random.shuffle(stems)
    n = len(stems)
    cut1 = round(n * SPLIT_RATIOS["train"])
    cut2 = cut1 + round(n * SPLIT_RATIOS["valid"])

    new_split_of = {}
    for i, stem in enumerate(stems):
        new_split_of[stem] = "train" if i < cut1 else ("valid" if i < cut2 else "test")

    counts = {s: 0 for s in SPLITS}
    for stem, items in groups.items():
        target = new_split_of[stem]
        for old_split, img in items:
            lbl = HNP_SRC / old_split / "labels" / f"{img.stem}.txt"
            dst_name = f"hnp_{img.name}"
            shutil.copy2(img, OUT / target / "images" / dst_name)
            dst_lbl = OUT / target / "labels" / f"hnp_{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl)
            else:
                dst_lbl.write_text("")
            counts[target] += 1
    return counts


def fold_in_nepali_plates():
    """Add every Nepali plate image in, remapping its single class (0) to
    class 3 (number-plate) to match this dataset's schema. Its own
    train/valid/test split is trustworthy (verified zero leakage, no
    augmentation applied), so splits are carried over as-is."""
    counts = {s: 0 for s in SPLITS}
    for split in SPLITS:
        img_dir = NP_SRC / split / "images"
        lbl_dir = NP_SRC / split / "labels"
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / f"{img.stem}.txt"
            dst_name = f"np_{img.name}"
            shutil.copy2(img, OUT / split / "images" / dst_name)
            dst_lbl = OUT / split / "labels" / f"np_{img.stem}.txt"
            if lbl.exists():
                remapped = []
                for line in lbl.read_text().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    parts[0] = "3"  # license_plate (0) -> number-plate (3)
                    remapped.append(" ".join(parts))
                dst_lbl.write_text("\n".join(remapped) + ("\n" if remapped else ""))
            else:
                dst_lbl.write_text("")
            counts[split] += 1
    return counts


def write_data_yaml():
    (OUT / "data.yaml").write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )


def write_readme(hnp_counts, np_counts):
    lines = [
        "Merged helmet + number-plate dataset",
        "=====================================",
        "",
        "Base: helmet and number plate dataset",
        "  (helmet-and-number-plate-detection-project/...-iityz, v3)",
        "  Re-split by SOURCE image (not by augmented copy) to fix the train/valid/test",
        "  leakage found in the earlier audit -- ~11-13% of the original val/test images",
        "  were near-duplicates of training images before this fix.",
        "",
        "Folded in: Number Plate.v3i.yolo26 (sujalkishwor/number-plate-wevp8, v3)",
        "  Nepali plate photos, class remapped 0 (license_plate) -> 3 (number-plate).",
        "  These images contribute plate boxes only -- they have no bike/helmet/no-helmet",
        "  annotations, which is expected (that source was never labeled for those classes)",
        "  and is normal practice when combining single-purpose and multi-class detection sets.",
        "  Their own split was carried over as-is (verified zero leakage, no augmentation).",
        "",
        "Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(NAMES)),
        "",
        "Split counts:",
    ]
    for split in SPLITS:
        hnp = hnp_counts[split]
        npp = np_counts[split]
        lines.append(f"  {split}: {hnp} (helmet dataset) + {npp} (Nepali plates) = {hnp+npp} total")
    lines.append("")
    lines.append("Filename prefixes: hnp_ = original helmet dataset, np_ = folded-in Nepali plates.")
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


def class_distribution():
    counts = {i: 0 for i in range(len(NAMES))}
    for split in SPLITS:
        for lbl in (OUT / split / "labels").glob("*.txt"):
            for line in lbl.read_text().splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
    return counts


def main():
    reset_output()
    hnp_counts = resplit_hnp()
    print("helmet+plate dataset re-split copied:", hnp_counts)

    np_counts = fold_in_nepali_plates()
    print("Nepali plate images folded in:", np_counts)

    write_data_yaml()
    write_readme(hnp_counts, np_counts)

    stems, tv, tt, vt = verify_no_leakage()
    print("\nFinal split sizes (unique source stems):", {k: len(v) for k, v in stems.items()})
    print(f"Leakage check -> train&valid={len(tv)} train&test={len(tt)} valid&test={len(vt)}")

    for split in SPLITS:
        n_img = len(list((OUT / split / "images").glob("*.jpg")))
        n_lbl = len(list((OUT / split / "labels").glob("*.txt")))
        print(f"{split}: images={n_img} labels={n_lbl}")

    dist = class_distribution()
    print("\nClass distribution (final merged dataset):")
    for i, name in enumerate(NAMES):
        print(f"  {name} (id {i}): {dist[i]} boxes")


if __name__ == "__main__":
    main()
