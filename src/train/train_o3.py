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
from src.models.crossvit_like import O3WeightedTwinViT
from src.models.patch_weighting import mask_to_patch_ratio, f_weight


@torch.no_grad()
def evaluate_o3(model, loader, device, patch_size: int, eps: float, gamma: float, normalize: bool):
    model.eval()
    correct = 0
    total = 0

    tp = fp = fn = 0

    for batch in loader:
        x1 = batch["raw"].to(device)
        x2 = batch["seg"].to(device)
        y = batch["y"].to(device)

        mask = batch["mask"]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float().to(device)

        rp = mask_to_patch_ratio(mask, patch_size=patch_size)
        wp = f_weight(rp, eps=eps, gamma=gamma, normalize=normalize)

        logits = model(x1, x2, wp=wp)
        pred = torch.argmax(logits, dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()


        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / max(1, total)
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return {"acc": acc, "f1": f1}


def main():
    project_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(str(project_root / "configs" / "base.yaml"))

    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    print(device)

    out_dir = ensure_dir(project_root / "outputs" / "runs" / "O3")

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

    # 关键超参（
    patch_size = int(cfg.get("o3", {}).get("patch_size", 16))
    eps = float(cfg.get("o3", {}).get("eps", 1e-3))
    gamma = float(cfg.get("o3", {}).get("gamma", 1.0))
    normalize = bool(cfg.get("o3", {}).get("normalize", True))

    model = O3WeightedTwinViT(
        backbone=cfg["model"]["backbone"],
        num_classes=int(cfg["model"]["num_classes"]),
        pooling="wmean",
        share_weights=False,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    ce = nn.CrossEntropyLoss()

    best_f1 = -1.0
    hist_e, hist_loss, hist_acc, hist_f1 = [], [], [], []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0

        for batch in tr_loader:
            x1 = batch["raw"].to(device)
            x2 = batch["seg"].to(device)
            y = batch["y"].to(device)

            mask = batch["mask"]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = (mask > 0.5).float().to(device)

            rp = mask_to_patch_ratio(mask, patch_size=patch_size)
            wp = f_weight(rp, eps=eps, gamma=gamma, normalize=normalize)

            opt.zero_grad(set_to_none=True)
            logits = model(x1, x2, wp=wp)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.detach().item()

        avg_loss = total_loss / max(1, len(tr_loader))
        metrics = evaluate_o3(model, va_loader, device, patch_size, eps, gamma, normalize)

        hist_e.append(epoch)
        hist_loss.append(avg_loss)
        hist_acc.append(metrics["acc"])
        hist_f1.append(metrics["f1"])

        print(f"[O3] epoch={epoch:02d} loss={avg_loss:.4f} acc={metrics['acc']:.4f} f1={metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    # 保存 CSV
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_acc", "val_f1"])
        for e, l, a, f1 in zip(hist_e, hist_loss, hist_acc, hist_f1):
            w.writerow([e, f"{l:.6f}", f"{a:.6f}", f"{f1:.6f}"])

    # 曲线
    plt.figure()
    plt.plot(hist_e, hist_loss)
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.title("O3 Train Loss")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(hist_e, hist_acc)
    plt.xlabel("epoch"); plt.ylabel("val acc"); plt.title("O3 Val Accuracy")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_val_acc.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(hist_e, hist_f1)
    plt.xlabel("epoch"); plt.ylabel("val f1"); plt.title("O3 Val F1")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_val_f1.png", dpi=200)
    plt.close()

    print("Done. best_f1=", best_f1)
    print(f"[OK] metrics saved to: {csv_path}")
    print(f"[OK] curves saved to: {out_dir}")


if __name__ == "__main__":
    main()