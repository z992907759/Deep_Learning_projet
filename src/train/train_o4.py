from __future__ import annotations

from pathlib import Path
import json
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import load_yaml, seed_everything, get_device, split_indices
from src.data.transforms import PairedTransform
from src.data.paired_dataset import PairedHerbierDataset
from src.models.crossvit_like import CrossViTLike, O2SameResTwinViT, O3WeightedTwinViT  # 按你实际用的模型改
from src.interpretability.attention_rollout import rollout_patch_map_from_model


def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _compute_iou(att_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    inter = np.logical_and(att_bin, mask_bin).sum()
    union = np.logical_or(att_bin, mask_bin).sum()
    return float(inter / union) if union > 0 else 0.0


def _select_o4_inputs_and_branch(tag: str, raw: torch.Tensor, seg: torch.Tensor):
    if tag == "O1_A":
        return raw, raw, "raw"
    if tag == "O1_B":
        return seg, seg, "seg"
    if tag == "O1_C1":
        return raw, seg, "seg"
    if tag == "O1_C2":
        return seg, raw, "raw"
    if tag in {"O2", "O3", "O5"}:
        return raw, seg, "seg"
    return raw, seg, "seg"


@torch.no_grad()
def main():
    here = Path(__file__).resolve()
    project_root = None
    for p in [here.parent] + list(here.parents):
        if (p / "configs" / "base.yaml").exists():
            project_root = p
            break
    if project_root is None:
        raise FileNotFoundError("找不到 configs/base.yaml，请确认从项目目录运行")

    cfg = load_yaml(str(project_root / "configs" / "base.yaml"))
    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    print("device =", device)

    tag = "O3"  # 改成 "O1_A" / "O1_B" / "O1_C1" / "O1_C2" / "O2" / "O3"
    ckpt_path = project_root / "outputs" / "runs" / tag / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}。请确认先训练并生成 best.pt，或检查 tag 名称是否和 outputs/runs 下的文件夹一致。")

    out_dir = project_root / "outputs" / "runs" / "O4" / tag
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    data_root = (project_root / cfg["data"]["root"]).resolve()
    manifest_csv = (project_root / cfg["data"]["manifest_csv"]).resolve()
    tfm = PairedTransform(image_size=int(cfg["data"]["image_size"]))

    ds = PairedHerbierDataset(
        data_root=str(data_root),
        manifest_csv=str(manifest_csv),
        transform=tfm,
    )
    tr_idx, va_idx = split_indices(len(ds), train_ratio=0.8, seed=int(cfg["seed"]))

    if tag.startswith("O1_"):
        sig = inspect.signature(CrossViTLike.__init__)
        params = sig.parameters

        small = cfg.get("model", {}).get("backbone_small", cfg["model"]["backbone"])
        large = cfg.get("model", {}).get("backbone_large", cfg["model"]["backbone"])
        num_classes = int(cfg["model"]["num_classes"])
        pooling = cfg.get("model", {}).get("pooling", "cls")
        cross_attn_heads = int(cfg.get("model", {}).get("cross_attn_heads", 8))
        fusion_dim = cfg.get("model", {}).get("fusion_dim", None)

        kwargs = {}
        if "backbone_small" in params:
            kwargs["backbone_small"] = small
        if "backbone_large" in params:
            kwargs["backbone_large"] = large


        if "backbone" in params and "backbone_small" not in params and "backbone_large" not in params:
            kwargs["backbone"] = cfg["model"]["backbone"]


        if "small_backbone" in params:
            kwargs["small_backbone"] = small
        if "large_backbone" in params:
            kwargs["large_backbone"] = large

        # 类别数
        if "num_classes" in params:
            kwargs["num_classes"] = num_classes
        elif "n_classes" in params:
            kwargs["n_classes"] = num_classes

        if "pooling" in params:
            kwargs["pooling"] = pooling
        if "cross_attn_heads" in params:
            kwargs["cross_attn_heads"] = cross_attn_heads
        if "fusion_dim" in params:
            kwargs["fusion_dim"] = (int(fusion_dim) if fusion_dim is not None else None)
        if "pretrained" in params:
            kwargs["pretrained"] = False

        try:
            model = CrossViTLike(**kwargs)
        except TypeError as e:
            raise TypeError(
                "构建 CrossViTLike 失败：CrossViTLike.__init__ 的参数名与当前适配逻辑不匹配。\n"
                f"CrossViTLike.__init__ 签名: {sig}\n"
                f"我们尝试传入的 kwargs: {kwargs}\n"
                "请把 src/models/crossvit_like.py 里 CrossViTLike 的 __init__ 定义截图发我，我会按你的真实签名改。\n"
                "原始错误：\n" + str(e)
            )
    elif tag == "O2":
        model = O2SameResTwinViT(
            backbone=cfg["model"]["backbone"],
            num_classes=int(cfg["model"]["num_classes"]),
            pretrained=False,
        )
    elif tag == "O3":
        model = O3WeightedTwinViT(
            backbone=cfg["model"]["backbone"],
            num_classes=int(cfg["model"]["num_classes"]),
            pooling="wmean",
            share_weights=False,
            pretrained=False,
        )
    else:
        model = O2SameResTwinViT(
            backbone=cfg["model"]["backbone"],
            num_classes=int(cfg["model"]["num_classes"]),
            pretrained=False,
        )

    try:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "加载权重失败：这通常是因为 tag 对应的模型结构和这里实例化的不一致。\n"
            f"当前 tag={tag}，ckpt={ckpt_path}\n"
            "请确认 train_o1/train_o2/train_o3 保存 best.pt 时使用的模型类，\n"
            "并在 train_o4.py 里对应该 tag 选择相同的模型类。\n"
            "原始错误：\n" + str(e)
        )
    model = model.to(device)
    model.eval()
    print("[OK] loaded:", ckpt_path)

    # IoU 统计
    ious = []

    N_VIZ = 30

    for j, idx in enumerate(va_idx):
        batch = ds[idx]
        raw = batch["raw"].unsqueeze(0).to(device)   # (1,3,H,W)
        seg = batch["seg"].unsqueeze(0).to(device)
        x1, x2, rollout_branch = _select_o4_inputs_and_branch(tag, raw, seg)
        mask = batch["mask"]                         # (H,W) 或 (1,H,W)
        if mask.dim() == 3:
            mask = mask[0]
        mask_bin = (mask.detach().cpu().numpy() > 0.5)

        rollout_patch = rollout_patch_map_from_model(model, x1, x2, branch=rollout_branch)  # (h_patch, w_patch)
        rollout_map = torch.as_tensor(rollout_patch).float().unsqueeze(0).unsqueeze(0)  # (1,1,h,w)

        rollout_map = torch.nn.functional.interpolate(
            rollout_map, size=raw.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0].detach().cpu().numpy()

        # 归一化
        rollout_map = (rollout_map - rollout_map.min()) / (rollout_map.max() - rollout_map.min() + 1e-8)

        # 二值化
        thr = np.quantile(rollout_map.reshape(-1), 0.8)
        att_bin = rollout_map >= thr

        # IoU
        iou = _compute_iou(att_bin, mask_bin)
        ious.append(iou)

        # 保存可视化叠加图
        if j < N_VIZ:
            raw_img = _to_numpy_img(batch["raw"])
            plt.figure()
            plt.imshow(raw_img)
            plt.imshow(rollout_map, alpha=0.45)  # 热力图叠加
            plt.title(f"{tag} | idx={idx} | IoU={iou:.3f}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(overlays_dir / f"overlay_{j:03d}_iou_{iou:.3f}.png", dpi=200)
            plt.close()

    # 统计输出
    ious = np.array(ious, dtype=np.float32)
    stats = {
        "tag": tag,
        "n_val": int(len(va_idx)),
        "iou_mean": float(ious.mean()),
        "iou_std": float(ious.std()),
        "quantile": 0.8,
        "n_viz_saved": int(min(N_VIZ, len(va_idx))),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "iou_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[O4] IoU mean/std =", stats["iou_mean"], stats["iou_std"])
    print("[OK] saved to:", out_dir)


if __name__ == "__main__":
    main()
