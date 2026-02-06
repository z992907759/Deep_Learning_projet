from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.utils import load_yaml, seed_everything, get_device, split_indices
from src.data.transforms import PairedTransform
from src.data.paired_dataset import PairedHerbierDataset
from src.interpretability.attention_rollout import rollout_patch_map_from_model

# 模型类：不同 tag 用不同结构
from src.models.crossvit_like import CrossViTLike, O2SameResTwinViT, O3WeightedTwinViT


def find_project_root() -> Path:
    """从当前文件位置向上找到项目根目录（包含 configs/base.yaml）。"""
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "configs" / "base.yaml").exists():
            return p
    raise FileNotFoundError("找不到 configs/base.yaml：请确认脚本在项目目录内")


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def _compute_iou(att_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    inter = np.logical_and(att_bin, mask_bin).sum()
    union = np.logical_or(att_bin, mask_bin).sum()
    return float(inter / union) if union > 0 else 0.0


def build_model_for_tag(cfg: Dict, tag: str) -> torch.nn.Module:
    """根据 tag 构建与 best.pt 匹配的模型结构。"""
    backbone = str(cfg["model"]["backbone"])
    num_classes = int(cfg["model"].get("num_classes", 2))
    pooling = str(cfg.get("model", {}).get("pooling", "cls"))
    share_weights = bool(cfg.get("model", {}).get("share_weights", False))

    if tag.startswith("O1_"):
        # O1: CrossViTLike（raw_enc/seg_enc）
        return CrossViTLike(backbone=backbone, num_classes=num_classes, pooling=pooling)

    if tag == "O2":
        return O2SameResTwinViT(
            backbone=backbone,
            num_classes=num_classes,
            pooling=pooling,
            share_weights=share_weights,
            pretrained=False,
        )

    if tag == "O3":
        return O3WeightedTwinViT(
            backbone=backbone,
            num_classes=num_classes,
            pooling=pooling,
            share_weights=share_weights,
            pretrained=False,
        )

    if tag == "O5":
        return O2SameResTwinViT(
            backbone=backbone,
            num_classes=num_classes,
            pooling=pooling,
            share_weights=share_weights,
            pretrained=False,
        )

    return O2SameResTwinViT(
        backbone=backbone,
        num_classes=num_classes,
        pooling=pooling,
        share_weights=share_weights,
        pretrained=False,
    )


@torch.no_grad()
def eval_iou_for_tag(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    quantile: float = 0.8,
    branch: str = "seg",
) -> List[float]:
    # 计算 tag 的 iou 统计
    ious: List[float] = []
    model.eval()

    for batch in loader:
        raw = batch["raw"].to(device)
        seg = batch["seg"].to(device)

        if "mask" not in batch:
            raise KeyError(f"batch 没有 mask 字段，keys={list(batch.keys())}")

        mask = batch["mask"]
        # mask 可能是 (B,1,H,W) 或 (B,H,W)
        if mask.dim() == 4:
            mask = mask[:, 0]
        mask_bin = (mask.detach().cpu().numpy() > 0.5)  # (B,H,W)

        # rollout_patch_map_from_model 返回 (h_patch, w_patch)（单张）
        B = raw.shape[0]
        for i in range(B):
            r = raw[i : i + 1]
            s = seg[i : i + 1]

            patch_map = rollout_patch_map_from_model(model, r, s, branch=branch)
            patch_map = patch_map.float().unsqueeze(0).unsqueeze(0)

            # upsample 到原图大小
            H, W = r.shape[-2], r.shape[-1]
            heat = F.interpolate(patch_map, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            heat = heat.detach().cpu().numpy()
            heat = _normalize01(heat)

            thr = float(np.quantile(heat.reshape(-1), quantile))
            att_bin = heat >= thr

            iou = _compute_iou(att_bin, mask_bin[i])
            ious.append(iou)

    return ious


def main():
    project_root = find_project_root()
    cfg = load_yaml(str(project_root / "configs" / "base.yaml"))
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    print("device =", device)

    tags = ["O1_A", "O1_B", "O1_C1", "O1_C2", "O2", "O3", "O5"]
    quantile = 0.8
    branch = "seg"

    # 数据集（与训练一致）
    transform = PairedTransform(image_size=int(cfg.get("data", {}).get("image_size", 224)))
    data_root = (project_root / cfg["data"]["root"]).resolve()
    manifest_csv = (project_root / cfg["data"]["manifest_csv"]).resolve()
    ds = PairedHerbierDataset(
        data_root=str(data_root),
        manifest_csv=str(manifest_csv),
        transform=transform,
    )

    tr_idx, va_idx = split_indices(
        len(ds),
        train_ratio=float(cfg.get("train", {}).get("train_ratio", 0.8)),
        seed=int(cfg.get("seed", 42)),
    )
    val_set = Subset(ds, va_idx)
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("train", {}).get("num_workers", 0)),
    )

    out_root = project_root / "outputs" / "runs" / "IOU_STATS"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Tuple[str, int, float, float]] = []

    for tag in tags:
        ckpt = project_root / "outputs" / "runs" / tag / "best.pt"
        if not ckpt.exists():
            print(f"[SKIP] {tag}: 找不到 {ckpt}")
            continue

        model = build_model_for_tag(cfg, tag)
        state = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model = model.to(device)

        print(f"\n[{tag}] evaluating IoU ... ({ckpt.name})")
        ious = eval_iou_for_tag(model, val_loader, device=device, quantile=quantile, branch=branch)
        ious_np = np.array(ious, dtype=np.float32)

        stats = {
            "tag": tag,
            "n_val": int(len(ious_np)),
            "quantile": float(quantile),
            "branch": branch,
            "iou_mean": float(ious_np.mean()) if len(ious_np) else 0.0,
            "iou_std": float(ious_np.std()) if len(ious_np) else 0.0,
            "iou_min": float(ious_np.min()) if len(ious_np) else 0.0,
            "iou_max": float(ious_np.max()) if len(ious_np) else 0.0,
        }

        # 保存 json
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "iou_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        summary_rows.append((tag, stats["n_val"], stats["iou_mean"], stats["iou_std"]))
        print(f"[OK] {tag} IoU mean/std = {stats['iou_mean']:.4f} / {stats['iou_std']:.4f}")
        print(f"[OK] saved {out_dir / 'iou_stats.json'}")

    # 汇总 CSV
    csv_path = out_root / "iou_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "n_val", "iou_mean", "iou_std"])
        for r in summary_rows:
            w.writerow(list(r))

    print("\n[OK] saved summary:", csv_path)


if __name__ == "__main__":
    main()