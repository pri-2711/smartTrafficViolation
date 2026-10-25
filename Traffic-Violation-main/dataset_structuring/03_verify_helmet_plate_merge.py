"""
Step 3 of the dataset pipeline.

Independent re-check of step 2's output, re-derived straight from the files on
disk rather than trusting that script's own printed summary -- run this any
time you want to re-confirm dataset/helmet_and_plate is still leakage-free and
its class counts still add up.
"""
import re
from pathlib import Path

ROOT = Path(r"C:\Parin\Navrachna\7th Sem\Artificial Intelligence of Things [CSE712]")
OUT = ROOT / "dataset" / "helmet_and_plate"
SPLITS = ["train", "valid", "test"]
STEM_RE = re.compile(r"_jpg\.rf\.[0-9a-f]+$", re.IGNORECASE)
NAMES = ["bike", "helmet", "no-helmet", "number-plate"]


def stem_of(path: Path) -> str:
    return STEM_RE.sub("", path.stem)


stems = {s: set() for s in SPLITS}
for split in SPLITS:
    for img in (OUT / split / "images").glob("*.jpg"):
        stems[split].add(stem_of(img))

tv = stems["train"] & stems["valid"]
tt = stems["train"] & stems["test"]
vt = stems["valid"] & stems["test"]
print("Unique source stems:", {k: len(v) for k, v in stems.items()})
print(f"Leakage check -> train&valid={len(tv)} train&test={len(tt)} valid&test={len(vt)}")

counts = {i: 0 for i in range(len(NAMES))}
empty_files = 0
total_files = 0
for split in SPLITS:
    for lbl in (OUT / split / "labels").glob("*.txt"):
        total_files += 1
        text = lbl.read_text().strip()
        if not text:
            empty_files += 1
            continue
        for line in text.splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1

print(f"\nLabel files: {total_files} total, {empty_files} empty")
print("Class distribution:")
for i, name in enumerate(NAMES):
    print(f"  {name} (id {i}): {counts[i]} boxes")
