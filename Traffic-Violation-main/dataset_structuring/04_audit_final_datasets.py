"""
Step 4 of the dataset pipeline.

Full byte-level integrity audit of both final datasets (dataset/helmet_and_plate
and dataset/triple_riding) -- not a sample, every image is decoded and every
label line is parsed. Checks: corrupt/unreadable images, orphan image/label
pairs, malformed label lines (field count, class range, bbox bounds), exact
byte-identical duplicates (including cross-split leakage), resolution stats,
empty-label (background) ratio, and class balance.

This is what originally caught the 328 polygon-format label lines fixed in
step 5 -- re-run it any time after touching either dataset folder.

Usage:
    python 04_audit_final_datasets.py [output_report.txt]

Takes a few minutes (it fully decodes ~35k images). Requires Pillow:
    pip install Pillow
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image

ROOT = Path(r"C:\Parin\Navrachna\7th Sem\Artificial Intelligence of Things [CSE712]")
SPLITS = ["train", "valid", "test"]
STEM_RE = re.compile(r"_jpg\.rf\.[0-9a-f]+$", re.IGNORECASE)

DATASETS = [
    {
        "name": "dataset/helmet_and_plate",
        "nc": 4,
        "names": ["bike", "helmet", "no-helmet", "number-plate"],
    },
    {
        "name": "dataset/triple_riding",
        "nc": 1,
        "names": ["tripleriding"],
    },
]


def stem_of(path: Path) -> str:
    return STEM_RE.sub("", path.stem)


def audit_dataset(cfg, report):
    root = ROOT / cfg["name"]
    nc = cfg["nc"]
    names = cfg["names"]
    report.write(f"\n{'=' * 70}\nDATASET: {cfg['name']}\n{'=' * 70}\n")
    report.flush()

    all_hashes = defaultdict(list)  # md5 -> list of (split, filename)
    all_stems = {s: set() for s in SPLITS}
    widths, heights = [], []
    tiny_images = []
    corrupt_images = []
    orphan_images = []
    orphan_labels = []
    invalid_label_lines = []
    class_counts = Counter()
    empty_labels_per_split = {s: 0 for s in SPLITS}
    total_images_per_split = {s: 0 for s in SPLITS}
    aspect_ratios = Counter()

    t0 = time.time()
    for split in SPLITS:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        images = sorted(img_dir.glob("*.jpg"))
        total_images_per_split[split] = len(images)
        report.write(f"\n-- {split}: {len(images)} images --\n")
        report.flush()

        img_stems_seen = set()
        for i, img_path in enumerate(images):
            img_stems_seen.add(img_path.stem)
            all_stems[split].add(stem_of(img_path))

            raw = img_path.read_bytes()
            h = hashlib.md5(raw).hexdigest()
            all_hashes[h].append((split, img_path.name))

            try:
                im = Image.open(img_path)
                im.load()
                w, ht = im.size
                widths.append(w)
                heights.append(ht)
                aspect_ratios[round(w / ht, 2)] += 1
                if w < 64 or ht < 64:
                    tiny_images.append((split, img_path.name, w, ht))
            except Exception as e:
                corrupt_images.append((split, img_path.name, str(e)))

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                orphan_images.append((split, img_path.name))
                continue

            text = lbl_path.read_text().strip()
            if not text:
                empty_labels_per_split[split] += 1
                continue

            for line_no, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid_label_lines.append(
                        (split, lbl_path.name, line_no, f"expected 5 fields, got {len(parts)}")
                    )
                    continue
                try:
                    cid = int(parts[0])
                    x, y, w2, h2 = (float(p) for p in parts[1:])
                except ValueError:
                    invalid_label_lines.append((split, lbl_path.name, line_no, "non-numeric field"))
                    continue

                if not (0 <= cid < nc):
                    invalid_label_lines.append(
                        (split, lbl_path.name, line_no, f"class id {cid} out of range [0,{nc - 1}]")
                    )
                    continue
                class_counts[cid] += 1

                if w2 <= 0 or h2 <= 0:
                    invalid_label_lines.append((split, lbl_path.name, line_no, f"non-positive w/h ({w2},{h2})"))
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    invalid_label_lines.append((split, lbl_path.name, line_no, f"center out of [0,1] ({x},{y})"))
                x1, x2b, y1, y2b = x - w2 / 2, x + w2 / 2, y - h2 / 2, y + h2 / 2
                if x1 < -0.01 or x2b > 1.01 or y1 < -0.01 or y2b > 1.01:
                    invalid_label_lines.append(
                        (split, lbl_path.name, line_no, f"box extends outside image bounds ({x1:.3f},{y1:.3f},{x2b:.3f},{y2b:.3f})")
                    )

            if (i + 1) % 5000 == 0:
                report.write(f"   ...{i + 1}/{len(images)} checked ({time.time() - t0:.0f}s elapsed)\n")
                report.flush()

        # orphan labels: label files with no matching image
        for lbl_path in lbl_dir.glob("*.txt"):
            if lbl_path.stem not in img_stems_seen:
                orphan_labels.append((split, lbl_path.name))

    # cross-split leakage (belt-and-suspenders re-check)
    tv = all_stems["train"] & all_stems["valid"]
    tt = all_stems["train"] & all_stems["test"]
    vt = all_stems["valid"] & all_stems["test"]

    # exact duplicate files (byte-identical), anywhere in the dataset
    dupe_groups = {h: v for h, v in all_hashes.items() if len(v) > 1}
    dupes_within_split = 0
    dupes_across_split = 0
    for h, occurrences in dupe_groups.items():
        splits_involved = set(s for s, _ in occurrences)
        if len(splits_involved) > 1:
            dupes_across_split += 1
        else:
            dupes_within_split += 1

    report.write(f"\n--- RESULTS: {cfg['name']} ---\n")
    report.write(f"Total images checked: {sum(total_images_per_split.values())} in {time.time() - t0:.0f}s\n")
    report.write(f"Per-split totals: {total_images_per_split}\n")

    report.write(f"\nCorrupt/unreadable images: {len(corrupt_images)}\n")
    for s, n, e in corrupt_images[:20]:
        report.write(f"  [{s}] {n}: {e}\n")

    report.write(f"\nOrphan images (no label file): {len(orphan_images)}\n")
    for s, n in orphan_images[:20]:
        report.write(f"  [{s}] {n}\n")

    report.write(f"\nOrphan labels (no matching image): {len(orphan_labels)}\n")
    for s, n in orphan_labels[:20]:
        report.write(f"  [{s}] {n}\n")

    report.write(f"\nInvalid label lines: {len(invalid_label_lines)}\n")
    for s, n, ln, msg in invalid_label_lines[:30]:
        report.write(f"  [{s}] {n}:{ln} -- {msg}\n")

    report.write(f"\nExact byte-identical duplicate groups: {len(dupe_groups)}\n")
    report.write(f"  within same split (harmless, just redundant): {dupes_within_split}\n")
    report.write(f"  ACROSS splits (real leakage if >0): {dupes_across_split}\n")
    if dupes_across_split:
        shown = 0
        for h, occurrences in dupe_groups.items():
            splits_involved = set(s for s, _ in occurrences)
            if len(splits_involved) > 1:
                report.write(f"    {occurrences}\n")
                shown += 1
                if shown >= 15:
                    break

    report.write(f"\nSource-stem leakage (regex-based, cross-check): train&valid={len(tv)} train&test={len(tt)} valid&test={len(vt)}\n")

    if widths:
        report.write(f"\nImage resolution: min=({min(widths)}x{min(heights)}) max=({max(widths)}x{max(heights)}) "
                      f"mean=({sum(widths)/len(widths):.0f}x{sum(heights)/len(heights):.0f})\n")
        report.write(f"Tiny images (<64px either dim): {len(tiny_images)}\n")
        for s, n, w, h in tiny_images[:10]:
            report.write(f"  [{s}] {n}: {w}x{h}\n")
        top_ar = aspect_ratios.most_common(5)
        report.write(f"Top aspect ratios: {top_ar}\n")

    report.write(f"\nEmpty label files (background images) per split:\n")
    for s in SPLITS:
        tot = total_images_per_split[s]
        emp = empty_labels_per_split[s]
        pct = (emp / tot * 100) if tot else 0
        report.write(f"  {s}: {emp}/{tot} ({pct:.1f}%)\n")

    report.write(f"\nClass distribution (valid boxes only):\n")
    total_boxes = sum(class_counts.values())
    for i, name in enumerate(names):
        c = class_counts[i]
        pct = (c / total_boxes * 100) if total_boxes else 0
        report.write(f"  {name} (id {i}): {c} boxes ({pct:.1f}%)\n")
    if len(names) > 1:
        mx, mn = max(class_counts.values()), min(class_counts[i] for i in range(nc))
        report.write(f"  imbalance ratio (max class / min class): {mx / mn:.2f}x\n" if mn else "  min class has 0 boxes!\n")

    report.flush()
    return {
        "name": cfg["name"],
        "corrupt": len(corrupt_images),
        "orphan_images": len(orphan_images),
        "orphan_labels": len(orphan_labels),
        "invalid_lines": len(invalid_label_lines),
        "dupes_across_split": dupes_across_split,
        "leakage": len(tv) + len(tt) + len(vt),
    }


def main():
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dataset_structuring" / "audit_report.txt"
    summary = []
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("FINAL DATASET AUDIT\n")
        for cfg in DATASETS:
            summary.append(audit_dataset(cfg, report))

        report.write(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}\n")
        report.write(json.dumps(summary, indent=2))
        report.write("\n")
    print(f"Report written to {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
