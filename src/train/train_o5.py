"""train_o5.py

O5：在 O2 的训练基础上，把 O4 的“注意力与植物区域的重合度(IoU)”加入训练目标（作为正则项）。

老师 PDF 的核心诉求（用人话复述）：
- O4：能把 attention rollout 叠加到图像上，并用 mask 计算 IoU（你已经做完）。
- O5：把这个“重合度”(IoU)纳入 loss，让模型在训练时更倾向于把注意力放在植物上。

实现策略（工程上可落地）：
- 分类/识别的主损失：CrossEntropy（或 BCE，取决于 num_classes）
- IoU 正则：
    loss = loss_cls + lambda_iou * (1 - soft_iou(att_map, mask))
  其中 att_map 来自 encoder 的 attention rollout（默认用 seg 分支）。

注意：
- 这里使用的是“soft IoU”，是可微的（att_map/mask 都是 float，做 soft intersection/union）。
- attention rollout 的计算需要从 timm 的 Attention 模块里抓 attention matrix。
  本项目已经在 src/interpretability/attention_rollout.py 里实现了 wrap/save 的逻辑；
  我们这里复用它，并在训练中把 detach 关掉，让 IoU 项能反向传播。

输出：
- outputs/runs/O5/best.pt
- outputs/runs/O5/curve_loss.png
- outputs/runs/O5/curve_val_acc.png
- outputs/runs/O5/curve_val_f1.png

如果你只想快速验证流程，可以在 config 里把 epochs 设小一点。
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from src.utils import load_yaml, seed_everything, get_device, split_indices
from src.data.transforms import PairedTransform
from src.data.paired_dataset import PairedHerbierDataset

# O2 的模型
from src.models.crossvit_like import O2SameResTwinViT

# 复用 O4 rollout 工具
from src.interpretability.attention_rollout import (
    extract_attn_mats_from_encoder,
    attention_rollout,
    cls_to_patch_map,
)



# 工具函数
def _ensure_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "configs" / "base.yaml").exists():
            return p
    raise FileNotFoundError("找不到 configs/base.yaml，请确认从项目目录运行")


def _plot_curve(values, title: str, ylabel: str, save_path: Path):
    plt.figure()
    plt.plot(range(1, len(values) + 1), values)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def _get_label_tensor(batch: Dict) -> torch.Tensor:
    for k in ["label", "y", "target", "class_id", "cls", "class"]:
        if k in batch:
            return batch[k]
    raise KeyError(f"batch 里找不到标签字段。可用 keys={list(batch.keys())}")


def _compute_acc_f1_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        if logits.shape[1] == 1:
            # 二分类（logit 形式）
            probs = torch.sigmoid(logits.squeeze(1))
            pred = (probs >= 0.5).long()
        else:
            pred = torch.argmax(logits, dim=1)

        y = y.long()
        acc = (pred == y).float().mean().item()

        num_classes = int(logits.shape[1]) if logits.shape[1] > 1 else 2
        f1s = []
        for c in range(num_classes):
            tp = ((pred == c) & (y == c)).sum().item()
            fp = ((pred == c) & (y != c)).sum().item()
            fn = ((pred != c) & (y == c)).sum().item()
            if tp == 0 and fp == 0 and fn == 0:
                continue
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            f1s.append(f1)
        f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
        return float(acc), float(f1)


def _soft_iou(att_map: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    att = att_map.clamp(0, 1)
    m = mask.clamp(0, 1)
    inter = (att * m).sum(dim=(2, 3))
    union = (att + m - att * m).sum(dim=(2, 3))
    return (inter + eps) / (union + eps)


def _rollout_heatmap_from_encoder(enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # 关键：detach=False，让 IoU 项能反向传播
    mats = extract_attn_mats_from_encoder(enc, x, detach=False)
    roll = attention_rollout(mats)  # (B,T,T)

    T = roll.shape[-1]
    P = T - 1
    cls_imp = cls_to_patch_map(roll, num_patches=P)  # (B,P)

    g = int(np.sqrt(P))
    if g * g != P:
        w = g
        h = P // g
        if h * w != P:
            raise RuntimeError(f"无法推断 patch 网格形状：P={P}")
    else:
        h = w = g

    heat = cls_imp.reshape(cls_imp.shape[0], 1, h, w)
    # 归一化到 [0,1]
    heat = heat - heat.amin(dim=(2, 3), keepdim=True)
    heat = heat / (heat.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return heat



# 训练主函数
def main():
    project_root = _ensure_project_root()
    cfg = load_yaml(str(project_root / "configs" / "base.yaml"))

    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    print("device =", device)

    o5_cfg: Dict = cfg.get("o5", {})
    tag = str(o5_cfg.get("tag", "O5"))
    epochs = int(o5_cfg.get("epochs", cfg.get("train", {}).get("epochs", 10)))
    batch_size = int(o5_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 8)))
    lr = float(o5_cfg.get("lr", cfg.get("train", {}).get("lr", 1e-4)))
    weight_decay = float(o5_cfg.get("weight_decay", cfg.get("train", {}).get("weight_decay", 1e-4)))
    lambda_iou = float(o5_cfg.get("lambda_iou", 0.5))
    branch = str(o5_cfg.get("branch", "seg"))  # 默认用 seg 分支做正则

    max_train_steps = o5_cfg.get("max_train_steps", None)
    max_val_steps = o5_cfg.get("max_val_steps", None)

    out_dir = project_root / "outputs" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    data_root = (project_root / cfg["data"]["root"]).resolve()
    manifest_csv = (project_root / cfg["data"]["manifest_csv"]).resolve()

    image_size = int(cfg["data"].get("image_size", 224))
    tfm = PairedTransform(image_size=image_size)

    ds = PairedHerbierDataset(
        data_root=str(data_root),
        manifest_csv=str(manifest_csv),
        transform=tfm,
    )

    tr_idx, va_idx = split_indices(
        len(ds),
        train_ratio=float(cfg.get("train", {}).get("train_ratio", 0.8)),
        seed=int(cfg.get("seed", 42)),
    )

    tr_set = Subset(ds, tr_idx)
    va_set = Subset(ds, va_idx)

    tr_loader = DataLoader(
        tr_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg.get("train", {}).get("num_workers", 0)),
    )
    va_loader = DataLoader(
        va_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("train", {}).get("num_workers", 0)),
    )

    model = O2SameResTwinViT(
        backbone=str(cfg["model"]["backbone"]),
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)

    # 分类损失
    num_classes = int(cfg["model"]["num_classes"])
    if num_classes <= 2:
        # 二分类时 O2SameResTwinViT 可能输出 (B,1) 或 (B,2)
        loss_cls_fn = nn.CrossEntropyLoss()
    else:
        loss_cls_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_epoch = None
    best_val_acc = None
    history_loss = []
    history_val_acc = []
    history_val_f1 = []

    print(f"[O5] epochs={epochs} bs={batch_size} lr={lr} lambda_iou={lambda_iou} branch={branch}")

    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = []
        print(f"\n[epoch {ep}/{epochs}] train ...")

        step = 0
        for batch in tr_loader:
            raw = batch["raw"].to(device)      # (B,3,H,W)
            seg = batch["seg"].to(device)      # (B,3,H,W)
            mask = batch["mask"].to(device)    # (B,1,H,W) 或 (B,H,W)
            # 标签字段名不确定，做兼容
            y = _get_label_tensor(batch).to(device)
            y = y.long()

            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

            opt.zero_grad(set_to_none=True)

            logits = model(raw, seg)

            # 分类 loss
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)

            if logits.shape[1] == 1 and num_classes <= 2:
                logits_ce = torch.cat([-logits, logits], dim=1)
            else:
                logits_ce = logits

            loss_cls = loss_cls_fn(logits_ce, y.long())

            # 选择 encoder 分支
            if not hasattr(model, "enc1") or not hasattr(model, "enc2"):
                raise AttributeError("O2SameResTwinViT 预期应有 enc1/enc2")

            enc = model.enc2 if branch == "seg" else model.enc1


            heat_patch = _rollout_heatmap_from_encoder(enc, seg if branch == "seg" else raw)


            heat = F.interpolate(heat_patch, size=mask.shape[-2:], mode="bilinear", align_corners=False)

            # soft IoU
            iou = _soft_iou(heat, mask)
            loss_iou = (1.0 - iou).mean()

            loss = loss_cls + lambda_iou * loss_iou

            loss.backward()
            opt.step()

            ep_losses.append(float(loss.item()))

            step += 1
            if max_train_steps is not None and step >= int(max_train_steps):
                break

        mean_loss = float(np.mean(ep_losses)) if len(ep_losses) > 0 else 0.0
        history_loss.append(mean_loss)

        # 验证
        model.eval()
        accs, f1s = [], []
        print(f"[epoch {ep}/{epochs}] val ...")

        vstep = 0
        for batch in va_loader:
            raw = batch["raw"].to(device)
            seg = batch["seg"].to(device)
            y = _get_label_tensor(batch).to(device)
            y = y.long()

            logits = model(raw, seg)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)

            if logits.shape[1] == 1 and num_classes <= 2:
                logits_eval = torch.cat([-logits, logits], dim=1)
            else:
                logits_eval = logits

            acc, f1 = _compute_acc_f1_from_logits(logits_eval, y)
            accs.append(acc)
            f1s.append(f1)

            vstep += 1
            if max_val_steps is not None and vstep >= int(max_val_steps):
                break

        val_acc = float(np.mean(accs)) if len(accs) > 0 else 0.0
        val_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

        history_val_acc.append(val_acc)
        history_val_f1.append(val_f1)

        print(f"[epoch {ep}] loss={mean_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        # 保存 best
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = ep
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

        # 每个 epoch 都保存曲线
        _plot_curve(history_loss, f"{tag} Train Loss", "train loss", out_dir / "curve_loss.png")
        _plot_curve(history_val_acc, f"{tag} Val Accuracy", "val acc", out_dir / "curve_val_acc.png")
        _plot_curve(history_val_f1, f"{tag} Val F1", "val f1", out_dir / "curve_val_f1.png")

        # 保存一份 json 日志
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_f1": best_f1,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "lambda_iou": lambda_iou,
                    "branch": branch,
                    "history_loss": history_loss,
                    "history_val_acc": history_val_acc,
                    "history_val_f1": history_val_f1,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 输出 metrics.csv
        metrics_csv = out_dir / "metrics.csv"
        with open(metrics_csv, "w", encoding="utf-8", newline="") as fcsv:
            fcsv.write("epoch,train_loss,val_acc,val_f1\n")
            for i in range(len(history_loss)):
                e = i + 1
                tl = history_loss[i]
                va = history_val_acc[i] if i < len(history_val_acc) else ""
                vf = history_val_f1[i] if i < len(history_val_f1) else ""
                fcsv.write(f"{e},{tl},{va},{vf}\n")

    print("\n[OK] O5 finished.")
    print("best_epoch =", best_epoch)
    print("best_val_acc =", best_val_acc)
    print("best_val_f1 =", best_f1)
    print("[OK] saved to:", out_dir)


if __name__ == "__main__":
    main()
