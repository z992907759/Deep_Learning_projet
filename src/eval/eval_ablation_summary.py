from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import json


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "configs" / "base.yaml").exists():
            return p
    raise FileNotFoundError("找不到 configs/base.yaml：请确认脚本在项目目录内")


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


def load_best_from_metrics_csv(csv_path: Path) -> Dict:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    col_epoch = _pick_col(df, ["epoch", "step"])
    col_val_f1 = _pick_col(df, ["val_f1", "f1_val", "f1"])
    col_val_acc = _pick_col(df, ["val_acc", "acc_val", "accuracy_val", "val_accuracy", "acc"])

    if col_val_f1 is None and col_val_acc is None:
        raise ValueError(f"{csv_path} 里找不到 val_f1/val_acc 列，现有列={list(df.columns)}")

    for c in [col_epoch, col_val_f1, col_val_acc]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    key_cols = [c for c in [col_val_f1, col_val_acc] if c is not None]
    df2 = df.dropna(subset=key_cols)
    if len(df2) == 0:
        raise ValueError(f"{csv_path} 里 val 指标全是 NaN，无法选 best")

    if col_val_f1 is not None:
        df2 = df2.sort_values(by=[col_val_f1, col_val_acc] if col_val_acc else [col_val_f1],
                              ascending=[False, False] if col_val_acc else [False])
    else:
        df2 = df2.sort_values(by=[col_val_acc], ascending=[False])

    best = df2.iloc[0]

    out = {
        "best_epoch": int(best[col_epoch]) if col_epoch is not None and pd.notna(best[col_epoch]) else None,
        "best_val_f1": float(best[col_val_f1]) if col_val_f1 is not None and pd.notna(best[col_val_f1]) else None,
        "best_val_acc": float(best[col_val_acc]) if col_val_acc is not None and pd.notna(best[col_val_acc]) else None,
    }
    return out


def load_best_from_metrics_json(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容常见字段名
    def pick(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    out = {
        "best_epoch": pick(data, ["best_epoch", "epoch"]),
        "best_val_f1": pick(data, ["best_val_f1", "val_f1", "f1"]),
        "best_val_acc": pick(data, ["best_val_acc", "val_acc", "accuracy"]),
    }

    return out


def main():
    project_root = find_project_root()
    runs_dir = project_root / "outputs" / "runs"

    tags = ["O1_A", "O1_B", "O1_C1", "O1_C2", "O2", "O3", "O5"]

    iou_csv = runs_dir / "IOU_STATS" / "iou_summary.csv"
    if not iou_csv.exists():
        raise FileNotFoundError(f"找不到 {iou_csv}，请先跑 eval_iou_stats.py")

    iou_df = pd.read_csv(iou_csv)
    iou_df["tag"] = iou_df["tag"].astype(str)

    rows = []
    for tag in tags:
        metrics_csv = runs_dir / tag / "metrics.csv"
        metrics_json = runs_dir / tag / "metrics.json"

        if metrics_csv.exists():
            best = load_best_from_metrics_csv(metrics_csv)
        elif metrics_json.exists():
            best = load_best_from_metrics_json(metrics_json)
        else:
            print(f"[SKIP] {tag}: 找不到 metrics.csv 或 metrics.json")
            continue

        row = {
            "tag": tag,
            "best_epoch": best["best_epoch"],
            "best_val_acc": best["best_val_acc"],
            "best_val_f1": best["best_val_f1"],
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("没有任何 tag 被成功读取到 metrics.csv")

    df = pd.DataFrame(rows)

    # fusionner IoU（mean/std）
    df = df.merge(iou_df, on="tag", how="left")  # iou_df 含 iou_mean / iou_std

    # Sortie dans un ordre expérimental fixe (à partir de O1)
    order = ["O1_A", "O1_B", "O1_C1", "O1_C2", "O2", "O3", "O5"]
    df["tag"] = pd.Categorical(df["tag"].astype(str), categories=order, ordered=True)
    df = df.sort_values(by=["tag"], ascending=True).reset_index(drop=True)

    # Conservez toutes les colonnes numériques à 3 décimales pour éviter que les nombres ne soient trop rapprochés dans les images du tableau.
    for c in ["best_val_acc", "best_val_f1", "iou_mean", "iou_std"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

    out_csv = runs_dir / "ablation_summary.csv"
    df.to_csv(out_csv, index=False, float_format="%.3f")
    print("[OK] saved:", out_csv)

    fig_path = runs_dir / "ablation_summary.png"

    df_display = df.copy()
    if "tag" in df_display.columns:
        df_display["tag"] = df_display["tag"].astype(str)
    df_display = df_display.fillna("")

    for c in ["best_val_acc", "best_val_f1", "iou_mean", "iou_std"]:
        if c in df_display.columns:
            df_display[c] = df_display[c].map(lambda x: f"{x:.3f}" if x != "" else "")

    fig, ax = plt.subplots(figsize=(12, 0.8 + 0.45 * len(df_display)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("[OK] saved:", fig_path)


if __name__ == "__main__":
    main()