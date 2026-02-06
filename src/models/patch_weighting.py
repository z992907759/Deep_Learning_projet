from __future__ import annotations
import torch
import torch.nn.functional as F

def mask_to_patch_ratio(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, C, H, W = mask.shape
    assert C == 1
    assert H % patch_size == 0 and W % patch_size == 0, "H/W must be divisible by patch_size"

    rp = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)  # (B,1,Hp,Wp)
    rp = rp.flatten(1)  # (B, P)
    return rp

def f_weight(rp: torch.Tensor, eps: float = 1e-3, gamma: float = 1.0, normalize: bool = True) -> torch.Tensor:
    wp = (eps + rp).pow(gamma)
    if normalize:
        wp = wp / (wp.mean(dim=1, keepdim=True) + 1e-8)
    return wp