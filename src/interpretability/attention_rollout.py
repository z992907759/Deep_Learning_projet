from __future__ import annotations

from typing import List, Tuple

import math
import torch
from torch import nn


def _get_timm_vit_blocks(enc: nn.Module) -> List[nn.Module]:
    blocks = getattr(enc, "blocks", None)
    if blocks is None:
        raise AttributeError("找不到 enc.blocks")
    return list(blocks)


def _wrap_attention_forward(attn_mod: nn.Module):
    if hasattr(attn_mod, "_o4_wrapped"):
        return

    required = ["qkv", "num_heads", "scale", "attn_drop", "proj", "proj_drop"]
    for r in required:
        if not hasattr(attn_mod, r):
            raise AttributeError(f"Attention 模块缺少属性 {r}，可能不是 timm 的 Attention")

    attn_mod._o4_wrapped = True

    def forward_with_save(x: torch.Tensor, attn_mask=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        qkv = attn_mod.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, attn_mod.num_heads, C // attn_mod.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * attn_mod.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn_mod._last_attn = attn.detach()
        attn = attn_mod.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = attn_mod.proj(x_out)
        x_out = attn_mod.proj_drop(x_out)
        return x_out

    attn_mod.forward = forward_with_save


def extract_attn_mats_from_encoder(enc: nn.Module, x: torch.Tensor, detach: bool = True):
    blocks = _get_timm_vit_blocks(enc)

    for blk in blocks:
        attn = getattr(blk, "attn", None)
        if attn is None:
            raise AttributeError("ViT block 找不到 blk.attn")
        _wrap_attention_forward(attn)

    if hasattr(enc, "forward_features"):
        _ = enc.forward_features(x)
    else:
        _ = enc(x)

    mats: List[torch.Tensor] = []
    for blk in blocks:
        attn = blk.attn
        if not hasattr(attn, "_last_attn"):
            raise RuntimeError("没有捕获到 attention matrix，请确认 encoder 是 timm ViT")
        mats.append(attn._last_attn)
    return mats


def attention_rollout(attn_mats: List[torch.Tensor]) -> torch.Tensor:
    assert len(attn_mats) > 0
    B = attn_mats[0].shape[0]
    device = attn_mats[0].device

    result = None
    for A in attn_mats:
        # A: (B, heads, T, T)
        A = A.mean(dim=1)  # (B, T, T)
        I = torch.eye(A.size(-1), device=device).unsqueeze(0).expand(B, -1, -1)
        A = A + I
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        result = A if result is None else torch.bmm(A, result)
    return result


def cls_to_patch_map(rollout: torch.Tensor, num_patches: int) -> torch.Tensor:
    cls_attn = rollout[:, 0, 1 : 1 + num_patches]
    return cls_attn


def rollout_patch_map_from_model(model: nn.Module, raw: torch.Tensor, seg: torch.Tensor, branch: str = "seg") -> torch.Tensor:
    if branch not in {"raw", "seg"}:
        raise ValueError("branch 必须是 'raw' 或 'seg'")

    x = raw if branch == "raw" else seg

    # 选择 encoder（按模型结构适配）
    enc = None

    # O2/O3：enc1/enc2
    if hasattr(model, "enc1") and hasattr(model, "enc2"):
        enc = model.enc1 if branch == "raw" else model.enc2


    elif hasattr(model, "raw_enc") and hasattr(model, "seg_enc"):
        enc = model.raw_enc if branch == "raw" else model.seg_enc

    elif hasattr(model, "raw_encoder") and hasattr(model, "seg_encoder"):
        enc = model.raw_encoder if branch == "raw" else model.seg_encoder

    if enc is None:
        raise AttributeError(
            "无法识别模型的双分支 encoder。需要 enc1/enc2 或 raw_enc/seg_enc（或 raw_encoder/seg_encoder）。"
        )

    if hasattr(enc, "encoder"):
        enc = enc.encoder

    mats = extract_attn_mats_from_encoder(enc, x)
    roll = attention_rollout(mats)  # (B,T,T)

    T = roll.shape[-1]
    P = T - 1
    cls_imp = cls_to_patch_map(roll, num_patches=P)[0]  # (P,)

    g = int(math.sqrt(P))
    if g * g != P:
        w = g
        h = P // g
        if h * w != P:
            raise RuntimeError(f"无法推断 patch 网格形状：P={P}")
    else:
        h = w = g

    return cls_imp.reshape(h, w).detach().cpu()