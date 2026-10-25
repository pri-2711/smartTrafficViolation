"""YOLO-format detection dataset with letterbox resizing and light augmentation.

Shared by train_option_b.py and evaluate.py, so Option B and YOLO26 are always
scored on identical pixels and identical ground truth.

Label files are plain YOLO txt: `class x_center y_center width height`, normalized.
An empty .txt means a negative image (no objects).
"""
import random
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PAD_VALUE = 114  # grey letterbox padding, same convention as YOLO


def load_data_yaml(data_yaml):
    """Return (class_names, {split: images_dir}) from a data.yaml, paths resolved relative to the yaml."""
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    splits = {}
    for key, split in (("train", "train"), ("val", "val"), ("test", "test")):
        if key in cfg:
            splits[split] = (root / cfg[key]).resolve()
    return list(names), splits


class YoloDetectionDataset(Dataset):
    def __init__(self, images_dir, imgsz=640, augment=False, limit=None):
        self.images = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        if limit:
            self.images = self.images[:limit]
        self.imgsz = imgsz
        self.augment = augment
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _read_labels(img_path):
        """Normalized (cls, x1, y1, x2, y2) rows. Polygon labels are reduced to their bounding extent."""
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        rows = []
        if not label_path.exists():
            return rows
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
            if len(vals) == 4:
                xc, yc, w, h = vals
                rows.append((cls, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                xs, ys = vals[0::2], vals[1::2]
                rows.append((cls, min(xs), min(ys), max(xs), max(ys)))
        return rows

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        S = self.imgsz

        # Random scale 0.7-1.3 during training; exactly fit-to-canvas at eval.
        scale = random.uniform(0.7, 1.3) if self.augment else 1.0
        r = S * scale / max(w0, h0)
        nw, nh = max(1, round(w0 * r)), max(1, round(h0 * r))
        img = img.resize((nw, nh), Image.BILINEAR)

        flip = False
        if self.augment:
            img = self.color_jitter(img)
            flip = random.random() < 0.5
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.augment:
            ox = random.randint(min(0, S - nw), max(0, S - nw))
            oy = random.randint(min(0, S - nh), max(0, S - nh))
        else:
            ox, oy = (S - nw) // 2, (S - nh) // 2
        canvas = Image.new("RGB", (S, S), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
        canvas.paste(img, (ox, oy))

        boxes = []
        for cls, x1, y1, x2, y2 in self._read_labels(path):
            x1, x2 = x1 * w0 * r, x2 * w0 * r
            y1, y2 = y1 * h0 * r, y2 * h0 * r
            if flip:
                x1, x2 = nw - x2, nw - x1
            x1, x2 = min(max(x1 + ox, 0), S), min(max(x2 + ox, 0), S)
            y1, y2 = min(max(y1 + oy, 0), S), min(max(y2 + oy, 0), S)
            if x2 - x1 > 2 and y2 - y1 > 2:
                boxes.append((cls, x1, y1, x2, y2))

        image = TF.to_tensor(canvas)
        target = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        return image, target


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
