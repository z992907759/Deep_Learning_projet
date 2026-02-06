from __future__ import annotations
import torch

def soft_iou(att_map: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    att = att_map.clamp(0, 1)
    m = mask.clamp(0, 1)
    inter = (att * m).sum(dim=(1,2,3))
    union = (att + m - att*m).sum(dim=(1,2,3))
    return (inter + eps) / (union + eps)