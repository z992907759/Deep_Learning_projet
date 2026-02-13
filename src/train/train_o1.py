from __future__ import annotations

from pathlib import Path
import csv
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.utils import load_yaml, seed_everything, get_device, ensure_dir, split_indices
from src.data.transforms import PairedTransform
from src.data.paired_dataset import PairedHerbierDataset
from src.models.crossvit_like import CrossViTLike
from src.eval import evaluate


def pick_inputs(batch, mode: str):
    raw = batch["raw"]
    seg = batch["seg"]
    if mode == "A":
        return raw, raw
    if mode == "B":
        return seg, seg
    if mode == "C1":
        return raw, seg
    if mode == "C2":
        return seg, raw
    raise ValueError(f"Unknown mode={mode}")


def main():
    # 项目根目录：.../DL_projet
    project_root = Path(__file__).resolve().parents[2]

    # 读取配置（
    cfg = load_yaml(str(project_root / "configs" / "base.yaml"))

    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    mode = cfg["experiment"]["mode"]

    # 输出目录（绝对路径）
    out_dir = ensure_dir(project_root / "outputs" / "runs" / f"O1_{mode}")

    # 数据/清单路径（绝对路径）
    data_root = (project_root / cfg["data"]["root"]).resolve()
    manifest_csv = (project_root / cfg["data"]["manifest_csv"]).resolve()

    tfm = PairedTransform(image_size=int(cfg["data"]["image_size"]))

    ds = PairedHerbierDataset(
        data_root=str(data_root),
        manifest_csv=str(manifest_csv),
        transform=tfm,
    )

    tr_idx, va_idx = split_indices(len(ds), train_ratio=0.8, seed=int(cfg["seed"]))

    pin_memory = (device == "cuda")

    tr_loader = DataLoader(
        Subset(ds, tr_idx),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=pin_memory,
    )
    va_loader = DataLoader(
        Subset(ds, va_idx),
        batch_size=int(cfg["train"]["batch_size"]) * 2,
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=pin_memory,
    )

    pooling = cfg["model"].get("pooling", "cls")
    backbone_small = cfg["model"].get("backbone_small", cfg["model"].get("backbone"))
    backbone_large = cfg["model"].get("backbone_large", cfg["model"].get("backbone"))
    cross_attn_heads = int(cfg["model"].get("cross_attn_heads", 8))
    fusion_dim = cfg["model"].get("fusion_dim", None)
    pretrained = bool(cfg["model"].get("pretrained", True))

    model = CrossViTLike(
        backbone_small=backbone_small,
        backbone_large=backbone_large,
        num_classes=int(cfg["model"]["num_classes"]),
        pooling=pooling,
        cross_attn_heads=cross_attn_heads,
        fusion_dim=(int(fusion_dim) if fusion_dim is not None else None),
        pretrained=pretrained,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    ce = nn.CrossEntropyLoss()

    best_f1 = -1.0

    # 记录曲线
    history_epochs: list[int] = []
    history_train_loss: list[float] = []
    history_val_acc: list[float] = []
    history_val_f1: list[float] = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0

        for batch in tr_loader:
            x1, x2 = pick_inputs(batch, mode)
            x1, x2 = x1.to(device), x2.to(device)
            y = batch["y"].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x1, x2)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.detach().item()

        metrics = evaluate(model, va_loader, device, mode=mode)
        avg_loss = total_loss / max(1, len(tr_loader))
        print(
            f"epoch={epoch:02d} loss={avg_loss:.4f} "
            f"acc={metrics['acc']:.4f} f1={metrics['f1']:.4f}"
        )

        # 记录历史
        history_epochs.append(epoch)
        history_train_loss.append(avg_loss)
        history_val_acc.append(float(metrics["acc"]))
        history_val_f1.append(float(metrics["f1"]))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    # 保存日志到 CSV（用于复现实验/画表）
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_acc", "val_f1"])
        for e, l, a, f1 in zip(history_epochs, history_train_loss, history_val_acc, history_val_f1):
            w.writerow([e, f"{l:.6f}", f"{a:.6f}", f"{f1:.6f}"])

    # 保存曲线图（用于报告）
    # 1 Loss
    plt.figure()
    plt.plot(history_epochs, history_train_loss)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title(f"O1-{mode} Train Loss")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_loss.png", dpi=200)
    plt.close()

    # 2 Val Acc
    plt.figure()
    plt.plot(history_epochs, history_val_acc)
    plt.xlabel("epoch")
    plt.ylabel("val acc")
    plt.title(f"O1-{mode} Val Accuracy")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_val_acc.png", dpi=200)
    plt.close()

    # 3 Val F1
    plt.figure()
    plt.plot(history_epochs, history_val_f1)
    plt.xlabel("epoch")
    plt.ylabel("val f1")
    plt.title(f"O1-{mode} Val F1")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_val_f1.png", dpi=200)
    plt.close()

    print(f"[OK] metrics saved to: {csv_path}")
    print(f"[OK] curves saved to: {out_dir}")

    print("Done. best_f1=", best_f1)


if __name__ == "__main__":
    main()
