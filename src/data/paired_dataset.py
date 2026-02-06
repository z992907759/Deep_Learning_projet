from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

class PairedHerbierDataset(Dataset):
    """
    Returns dict:
      raw:  (3,H,W)
      seg:  (3,H,W)
      mask: (1,H,W) 0/1
      y:    (,) long
      id:   str
    """
    def __init__(self, data_root: str | Path, manifest_csv: str | Path, transform):
        self.data_root = Path(data_root)
        self.manifest_csv = Path(manifest_csv)
        self.transform = transform

        df = pd.read_csv(self.manifest_csv, keep_default_na=False)
        required = {"id", "raw_path", "seg_path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {missing}")
        self.df = df
        if "mask_path" not in self.df.columns:
            self.df["mask_path"] = ""
        else:
            # Clean mask_path column: replace nan/None/'nan' with empty string
            def clean_mask_path(x):
                if isinstance(x, str):
                    if x.strip().lower() == "nan" or x.strip() == "":
                        return ""
                    else:
                        return x
                else:
                    return ""
            self.df["mask_path"] = self.df["mask_path"].map(clean_mask_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        raw = Image.open(self.data_root / r["raw_path"]).convert("RGB")
        seg = Image.open(self.data_root / r["seg_path"]).convert("RGB")

        mask_path = str(r.get("mask_path", "")).strip()
        if mask_path and mask_path.lower() not in {"nan", "none"}:
            mask = Image.open(self.data_root / mask_path).convert("L")
        else:
            # Générer un masque à partir du segment: traiter les zones non noires comme premier plan
            seg_gray = ImageOps.grayscale(seg)
            # Binarisation : Les valeurs de pixels > 5 sont considérées comme faisant partie du premier plan (pour éviter la compression du bruit).
            mask = seg_gray.point(lambda p: 255 if p > 5 else 0).convert("L")

        raw_t, seg_t, mask_t = self.transform(raw, seg, mask)
        y = torch.tensor(int(r["label"]), dtype=torch.long)
        return {"raw": raw_t, "seg": seg_t, "mask": mask_t, "y": y, "id": str(r["id"])}