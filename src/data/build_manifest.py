from __future__ import annotations
from pathlib import Path
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def index_by_stem(folder: Path) -> dict[str, Path]:
    m = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            m[p.stem] = p
    return m

def main():
    root = Path(__file__).resolve().parents[2]
    data_root = root / "data"

    raw_dir = data_root / "mission_herbonaute_2000"
    seg_dir = data_root / "mission_herbonaute_2000_seg_black"

    assert raw_dir.exists(), f"找不到目录: {raw_dir}"
    assert seg_dir.exists(), f"找不到目录: {seg_dir}"

    raw_map = index_by_stem(raw_dir)
    seg_map = index_by_stem(seg_dir)


    common = sorted(set(raw_map) & set(seg_map))
    if not common:
        raise RuntimeError("raw 与 seg 没有任何同名样本！请检查文件命名是否一致。")


    labels_path = data_root / "labels_clean.csv"
    if not labels_path.exists():
        labels_path = data_root / "labels.csv"

    assert labels_path.exists(), f"找不到 labels 文件: {labels_path}"

    labels_df = pd.read_csv(labels_path)

    # Noms de colonnes compatibles : code/Code, epines/épines, etc.
    col_map = {}
    for c in labels_df.columns:
        cl = str(c).strip().lower().replace("é", "e")
        if "code" in cl:
            col_map[c] = "code"
        if "epine" in cl:
            col_map[c] = "epines"
    labels_df = labels_df.rename(columns=col_map)

    if not {"code", "epines"} <= set(labels_df.columns):
        raise RuntimeError(f"labels 列名不符合预期：{list(labels_df.columns)}，需要至少包含 code 和 epines")


    def normalize_code(x: object) -> str:
        s = str(x).strip().upper()
        if s.endswith(".0"):
            s = s[:-2]
        return s.replace(" ", "").replace("_", "").replace("-", "")

    labels_df["code"] = labels_df["code"].map(normalize_code)
    labels_df["epines"] = labels_df["epines"].astype(float).astype(int)

    #Seules les étiquettes binaires 0/1 sont acceptées ; les autres valeurs (par exemple, -1) sont ignorées.
    labels_df = labels_df[labels_df["epines"].isin([0, 1])].copy()

    label_map = dict(zip(labels_df["code"], labels_df["epines"]))

    rows = []
    missing_label = 0

    for sid in common:
        code = normalize_code(sid)
        label = label_map.get(code, None)
        if label is None:
            missing_label += 1
            continue

        rows.append({
            "id": code,
            "raw_path": str(raw_map[sid].relative_to(data_root)),
            "seg_path": str(seg_map[sid].relative_to(data_root)),
            "mask_path": "",
            "label": label,
        })

    out = pd.DataFrame(rows)

    print(f"[统计] raw/seg 同名样本总数: {len(common)}")
    print(f"[统计] labels 行数: {len(labels_df)}")
    print(f"[统计] 没有 label 被过滤掉的样本: {missing_label}")
    print(f"[统计] 最终可用于训练的样本: {len(out)}")

    if len(out) == 0:
        example_labels = list(labels_df["code"].dropna().astype(str).head(10))
        example_imgs = [normalize_code(x) for x in common[:10]]
        print("[调试] labels code 示例:", example_labels)
        print("[调试] 图片 stem 示例:", example_imgs)
        raise RuntimeError("最终可用样本为 0：labels 的 code 与图片文件名不匹配，请把上面两行示例发我。")

    out_path = data_root / "manifest.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(out)} rows.")

if __name__ == "__main__":
    main()