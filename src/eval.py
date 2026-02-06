from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

@torch.no_grad()
def evaluate(model, loader, device: str, mode: str = "A"):
    model.eval()
    ys, ps = [], []

    for batch in loader:
        raw = batch["raw"].to(device)
        seg = batch["seg"].to(device)
        y = batch["y"].cpu().numpy()

        if mode == "A":
            x1, x2 = raw, raw
        elif mode == "B":
            x1, x2 = seg, seg
        elif mode == "C1":
            x1, x2 = raw, seg
        elif mode == "C2":
            x1, x2 = seg, raw
        else:
            raise ValueError(mode)

        logits = model(x1, x2)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

        ys.append(y)
        ps.append(pred)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    return {
        "acc": float(accuracy_score(ys, ps)),
        "f1": float(f1_score(ys, ps)),
        "cm": confusion_matrix(ys, ps),
    }