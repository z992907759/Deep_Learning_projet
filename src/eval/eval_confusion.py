import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset

from src.data.paired_dataset import PairedHerbierDataset
from src.data.transforms import PairedTransform
from src.models.crossvit_like import O2SameResTwinViT
from src.utils import split_indices



def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "configs" / "base.yaml").exists():
            return p
    raise FileNotFoundError("找不到 configs/base.yaml：请确认脚本在项目目录内")


def get_label(batch: dict) -> torch.Tensor:
    for k in ["label", "y", "target", "class_id", "cls", "class"]:
        if k in batch:
            return batch[k]
    raise KeyError(f"batch 中找不到标签字段, keys={list(batch.keys())}")


def main():
    #配置设备
    project_root = find_project_root()
    cfg_path = project_root / "configs" / "base.yaml"
    from src.utils import load_yaml, seed_everything, get_device
    cfg = load_yaml(str(cfg_path))
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    print("device =", device)

    # 构建与训练时一致的 Dataset
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
        batch_size=cfg.get("train", {}).get("batch_size", 8),
        shuffle=False,
        num_workers=0,
    )


    # O5 的 best.pt 对应 O2SameResTwinViT（enc1/enc2 + head）结构
    model = O2SameResTwinViT(
        backbone=str(cfg["model"]["backbone"]),
        num_classes=int(cfg["model"].get("num_classes", 2)),
        pooling=str(cfg.get("model", {}).get("pooling", "cls")),
        share_weights=bool(cfg.get("model", {}).get("share_weights", False)),
        pretrained=False,  # 评估时只加载 best.pt，不再下载预训练
    ).to(device)
    ckpt = project_root / "outputs" / "runs" / "O5" / "best.pt"
    assert ckpt.exists(), f"找不到模型权重: {ckpt}"

    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"[OK] loaded {ckpt}")

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
            raw = batch["raw"].to(device)
            seg = batch["seg"].to(device)
            y = get_label(batch).long().to(device)

            logits = model(raw, seg)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
            else:
                pred = logits.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())


    if int(cfg.get("model", {}).get("num_classes", 2)) == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    else:
        cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)


    # Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("O5 Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im)

    fig_dir = project_root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "confusion_matrix_O5.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()

    print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()