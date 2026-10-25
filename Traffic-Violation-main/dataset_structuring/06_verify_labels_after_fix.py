"""
Step 6 of the dataset pipeline.

Fast label-only re-check after step 5's polygon fix (skips image decoding,
so it runs in seconds rather than minutes) -- confirms zero malformed label
lines remain and reports the final class distribution.
"""
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\Parin\Navrachna\7th Sem\Artificial Intelligence of Things [CSE712]")
DATASET = ROOT / "dataset" / "helmet_and_plate"
SPLITS = ["train", "valid", "test"]
NAMES = ["bike", "helmet", "no-helmet", "number-plate"]
NC = 4

invalid = []
class_counts = Counter()
total_lines = 0

for split in SPLITS:
    lbl_dir = DATASET / split / "labels"
    for lbl_path in lbl_dir.glob("*.txt"):
        for ln, line in enumerate(lbl_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            total_lines += 1
            if len(parts) != 5:
                invalid.append((split, lbl_path.name, ln, len(parts)))
                continue
            cid = int(parts[0])
            x, y, w, h = (float(p) for p in parts[1:])
            if not (0 <= cid < NC):
                invalid.append((split, lbl_path.name, ln, f"bad class {cid}"))
                continue
            class_counts[cid] += 1
            if w <= 0 or h <= 0:
                invalid.append((split, lbl_path.name, ln, f"bad w/h {w},{h}"))
            x1, x2, y1, y2 = x - w / 2, x + w / 2, y - h / 2, y + h / 2
            if x1 < -0.01 or x2 > 1.01 or y1 < -0.01 or y2 > 1.01:
                invalid.append((split, lbl_path.name, ln, f"box out of bounds {x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}"))

print(f"Total label lines: {total_lines}")
print(f"Invalid lines remaining: {len(invalid)}")
for row in invalid[:20]:
    print(" ", row)
print("\nClass distribution:")
for i, n in enumerate(NAMES):
    print(f"  {n} (id {i}): {class_counts[i]}")
