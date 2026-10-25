"""Inference: turn raw Option B outputs into per-image detections (score threshold + NMS)."""
import torch
from torchvision.ops import batched_nms

from .grids import decode_boxes, grids_for
from .model import flatten_outputs


@torch.no_grad()
def postprocess(outputs, imgsz, conf=0.001, iou=0.6, max_det=300, topk=1000):
    cls, reg, obj = flatten_outputs(outputs)
    boxes = decode_boxes(reg, grids_for(outputs)).clamp(0, imgsz)
    scores = cls.sigmoid() * obj.sigmoid().unsqueeze(-1)  # (B, N, C) final confidence
    num_classes = scores.shape[-1]

    results = []
    for b in range(scores.shape[0]):
        flat = scores[b].flatten()
        top_s, top_i = flat.topk(min(topk, flat.numel()))
        keep = top_s > conf
        top_s, top_i = top_s[keep], top_i[keep]
        point_idx = top_i // num_classes
        labels = top_i % num_classes
        bx = boxes[b][point_idx]
        k = batched_nms(bx, top_s, labels, iou)[:max_det]
        results.append({"boxes": bx[k], "scores": top_s[k], "labels": labels[k]})
    return results
