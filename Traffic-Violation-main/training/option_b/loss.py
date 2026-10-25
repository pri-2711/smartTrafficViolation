"""Training loss: BCE classification + BCE objectness + CIoU box regression, normalized by positives."""
import torch
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss

from .assign import assign_image
from .grids import decode_boxes
from .model import flatten_outputs

LOSS_WEIGHTS = {"cls": 1.0, "obj": 1.0, "box": 5.0}


def detection_loss(outputs, targets, grids, num_classes, scale_ranges, weights=LOSS_WEIGHTS):
    cls, reg, obj = flatten_outputs(outputs)
    boxes = decode_boxes(reg, grids)

    pos_l, cls_l, box_l = [], [], []
    for t in targets:
        p, c, b = assign_image(t, grids, num_classes, scale_ranges)
        pos_l.append(p)
        cls_l.append(c)
        box_l.append(b)
    pos = torch.stack(pos_l)
    cls_t = torch.stack(cls_l)
    box_t = torch.stack(box_l)

    npos = pos.sum().clamp(min=1).float()
    l_obj = F.binary_cross_entropy_with_logits(obj, pos.float(), reduction="sum") / npos
    l_cls = F.binary_cross_entropy_with_logits(cls[pos], cls_t[pos], reduction="sum") / npos
    if pos.any():
        l_box = complete_box_iou_loss(boxes[pos], box_t[pos], reduction="sum") / npos
    else:
        l_box = boxes.sum() * 0.0

    total = weights["cls"] * l_cls + weights["obj"] * l_obj + weights["box"] * l_box
    parts = {"cls": l_cls.item(), "obj": l_obj.item(), "box": l_box.item()}
    return total, parts
