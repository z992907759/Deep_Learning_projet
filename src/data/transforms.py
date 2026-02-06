from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image

@dataclass
class PairedTransform:
    image_size: int = 224
    hflip_p: float = 0.5

    def _resize_img(self, img: Image.Image) -> Image.Image:
        return img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

    def _resize_mask(self, mask: Image.Image) -> Image.Image:
        return mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

    @staticmethod
    def _to_tensor_rgb(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        arr = np.transpose(arr, (2, 0, 1))               # CHW
        return torch.from_numpy(arr)

    @staticmethod
    def _to_tensor_mask(mask: Image.Image) -> torch.Tensor:
        arr = np.asarray(mask, dtype=np.float32) / 255.0  # HW
        t = torch.from_numpy(arr)[None, ...]              # 1HW
        return (t > 0.5).float()

    def __call__(self, raw: Image.Image, seg: Image.Image, mask: Image.Image):
        raw = self._resize_img(raw)
        seg = self._resize_img(seg)
        mask = self._resize_mask(mask)

        if np.random.rand() < self.hflip_p:
            raw = raw.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        raw_t = self._to_tensor_rgb(raw)
        seg_t = self._to_tensor_rgb(seg)
        mask_t = self._to_tensor_mask(mask)
        return raw_t, seg_t, mask_t