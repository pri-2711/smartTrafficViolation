"""Grid geometry: grid-point centers, strides, object-size ranges per level, and box decoding."""
import torch
import torch.nn.functional as F

from .model import STRIDES

CENTER_RADIUS = 2.5  # positives sit within 2.5 strides of the object center
BASE_SCALE_RANGES = ((0, 64), (64, 128), (128, float("inf")))  # object size in px at 640 input


def scale_ranges_for(imgsz):
    k = imgsz / 640
    return tuple((lo * k, hi * k) for lo, hi in BASE_SCALE_RANGES)


def make_grids(shapes, device):
    """shapes: [(H, W), ...] one per level. Returns per-level dicts with centers (HW, 2), stride, offset."""
    grids, offset = [], 0
    for (h, w), s in zip(shapes, STRIDES):
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        centers = torch.stack([(xs.flatten() + 0.5) * s, (ys.flatten() + 0.5) * s], dim=1).float()
        grids.append({"centers": centers, "stride": s, "offset": offset, "n": h * w})
        offset += h * w
    return grids


def grids_for(outputs):
    shapes = [(c.shape[2], c.shape[3]) for c, _, _ in outputs]
    return make_grids(shapes, outputs[0][0].device)


def decode_boxes(reg, grids):
    """reg (B, N, 4) raw distances -> xyxy boxes (B, N, 4) in input pixels."""
    centers = torch.cat([g["centers"] for g in grids], 0)
    strides = torch.cat([torch.full((g["n"],), float(g["stride"]), device=reg.device) for g in grids], 0)
    dist = F.softplus(reg) * strides[None, :, None]
    cx, cy = centers[None, :, 0], centers[None, :, 1]
    return torch.stack(
        [cx - dist[..., 0], cy - dist[..., 1], cx + dist[..., 2], cy + dist[..., 3]],
        dim=-1,
    )
