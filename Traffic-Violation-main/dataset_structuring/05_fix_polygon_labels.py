"""
Step 5 of the dataset pipeline.

Fixes a defect found by step 4's audit: 328 label lines (199 files) in the
original 'helmet and number plate dataset' Roboflow source were in
polygon/segmentation format (class_id x1 y1 x2 y2 ... xn yn) instead of plain
YOLO boxes -- a plain detect-task loader expects exactly 5 fields per line.

Converts each polygon to a standard box via its min/max extent:
    class_id x_center y_center width height

Idempotent: it only touches lines that don't already have 5 fields, so
running this again after it's already been applied is a safe no-op. This has
already been run once against dataset/helmet_and_plate; keep it here as the
record of what was changed and why, and as the fix to re-apply if the dataset
is ever rebuilt from scratch via 02_merge_helmet_plate.py.
"""
from pathlib import Path

ROOT = Path(r"C:\Parin\Navrachna\7th Sem\Artificial Intelligence of Things [CSE712]")
DATASET = ROOT / "dataset" / "helmet_and_plate"
SPLITS = ["train", "valid", "test"]


def polygon_to_bbox(cid, coords):
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main():
    files_touched = {s: 0 for s in SPLITS}
    lines_converted = {s: 0 for s in SPLITS}

    for split in SPLITS:
        lbl_dir = DATASET / split / "labels"
        for lbl_path in lbl_dir.glob("*.txt"):
            text = lbl_path.read_text()
            lines = [l for l in text.splitlines() if l.strip()]
            if not lines:
                continue

            changed = False
            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    new_lines.append(line)
                    continue
                # polygon line: class_id followed by an even number of coord values
                cid = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                if len(coords) % 2 != 0 or len(coords) < 6:
                    # not a recognizable polygon either -- keep original, will be flagged by audit
                    new_lines.append(line)
                    continue
                new_lines.append(polygon_to_bbox(cid, coords))
                changed = True
                lines_converted[split] += 1

            if changed:
                lbl_path.write_text("\n".join(new_lines) + "\n")
                files_touched[split] += 1

    print("Label files rewritten (polygon -> bbox) per split:", files_touched)
    print("Individual polygon lines converted per split:", lines_converted)


if __name__ == "__main__":
    main()
