"""Static center-sampling label assignment: which grid points are positives for each ground-truth box."""
import torch

from .grids import CENTER_RADIUS


def _grid_for_size(size, grids, scale_ranges):
    for g, (lo, hi) in zip(grids, scale_ranges):
        if lo <= size < hi:
            return g
    return grids[-1]


@torch.no_grad()
def assign_image(target, grids, num_classes, scale_ranges):
    """target: (K, 5) rows of [cls, x1, y1, x2, y2] in pixels.

    Returns over all N grid points: pos (N,) bool, cls_target (N, C), box_target (N, 4).
    """
    n_total = sum(g["n"] for g in grids)
    device = grids[0]["centers"].device
    pos = torch.zeros(n_total, dtype=torch.bool, device=device)
    cls_t = torch.zeros(n_total, num_classes, device=device)
    box_t = torch.zeros(n_total, 4, device=device)
    if target.numel() == 0:
        return pos, cls_t, box_t

    target = target.to(device)
    widths = target[:, 3] - target[:, 1]
    heights = target[:, 4] - target[:, 2]
    sizes = torch.maximum(widths, heights)
    areas = widths * heights

    # Largest first, so smaller objects overwrite shared points (smaller object wins a conflict).
    for k in torch.argsort(areas, descending=True).tolist():
        cls = int(target[k, 0])
        x1, y1, x2, y2 = target[k, 1:].tolist()
        g = _grid_for_size(float(sizes[k]), grids, scale_ranges)
        px, py = g["centers"][:, 0], g["centers"][:, 1]
        inside = (px > x1) & (px < x2) & (py > y1) & (py < y2)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = CENTER_RADIUS * g["stride"]
        near = ((px - cx).abs() < r) & ((py - cy).abs() < r)
        mask = inside & near
        if not bool(mask.any()):  # tiny box: fall back to the grid point nearest its center
            d = (px - cx) ** 2 + (py - cy) ** 2
            mask = torch.zeros_like(mask)
            mask[d.argmin()] = True
        idx = mask.nonzero().squeeze(1) + g["offset"]
        pos[idx] = True
        cls_t[idx] = 0
        cls_t[idx, cls] = 1
        box_t[idx] = target[k, 1:]
    return pos, cls_t, box_t
