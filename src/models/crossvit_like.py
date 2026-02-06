from __future__ import annotations
import timm
import torch
from torch import nn

class DualBranchEncoder(nn.Module):
    def __init__(self, backbone: str = "vit_base_patch16_224"):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)

    @property
    def feat_dim(self) -> int:
        return self.encoder.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # (B, D)

class CrossViTLike(nn.Module):
    def __init__(self, backbone: str, num_classes: int = 2, pooling: str = "cls"):
        super().__init__()
        self.raw_enc = DualBranchEncoder(backbone)
        self.seg_enc = DualBranchEncoder(backbone)
        d = self.raw_enc.feat_dim
        self.head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, num_classes),
        )
        self.pooling = pooling

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.raw_enc(x1)
        f2 = self.seg_enc(x2)
        f = torch.cat([f1, f2], dim=1)
        return self.head(f)

class O2SameResTwinViT(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int = 2,
        pooling: str = "cls",
        share_weights: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()

        try:
            import timm
        except Exception as e:
            raise ImportError("缺少依赖 timm：请先 pip install timm") from e

        self.pooling = pooling

        # 两分支同 backbone => 同分辨率
        self.enc1 = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        if share_weights:
            self.enc2 = self.enc1
        else:
            self.enc2 = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        dim = getattr(self.enc1, "num_features", None)
        if dim is None:
            with torch.no_grad():
                x = torch.zeros(1, 3, 224, 224)
                f = self._forward_features(self.enc1, x)
                dim = int(f.shape[-1])

        self.head = nn.Linear(dim * 2, int(num_classes))

    def _forward_features(self, enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(enc, "forward_features"):
            feat = enc.forward_features(x)
        else:
            feat = enc(x)

        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if feat.dim() == 3:
            if self.pooling == "cls":
                return feat[:, 0, :]
            if self.pooling == "mean":
                return feat.mean(dim=1)
            raise ValueError(f"Unknown pooling={self.pooling}")

        if feat.dim() == 2:
            return feat

        raise RuntimeError(f"Unexpected feature shape: {tuple(feat.shape)}")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self._forward_features(self.enc1, x1)
        f2 = self._forward_features(self.enc2, x2)
        fused = torch.cat([f1, f2], dim=1)
        return self.head(fused)

class O3WeightedTwinViT(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int = 2,
        pooling: str = "wmean",   # wmean / cls / mean
        share_weights: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError("缺少依赖 timm：请先 pip install timm") from e

        self.pooling = pooling

        self.enc1 = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        if share_weights:
            self.enc2 = self.enc1
        else:
            self.enc2 = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        dim = getattr(self.enc1, "num_features", None)
        if dim is None:
            dim = 768  # 兜底（一般 vit-base）

        self.head = nn.Linear(dim * 2, int(num_classes))

    def _forward_tokens(self, enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(enc, "forward_features"):
            feat = enc.forward_features(x)
        else:
            feat = enc(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return feat  # (B,N,C) 或 (B,C)

    def _pool(self, tokens: torch.Tensor, wp: torch.Tensor | None) -> torch.Tensor:
        # tokens: (B,N,C) 或 (B,C)
        if tokens.dim() == 2:
            return tokens

        # (B,N,C)
        if self.pooling == "cls":
            return tokens[:, 0, :]
        if self.pooling == "mean":
            return tokens[:, 1:, :].mean(dim=1)

        if self.pooling == "wmean":
            if wp is None:
                raise ValueError("pooling=wmean 时必须提供 wp")
            patch_tokens = tokens[:, 1:, :]  # (B,P,C)
            # wp: (B,P) -> 归一化使 sum=1
            w = wp / (wp.sum(dim=1, keepdim=True) + 1e-8)
            return (patch_tokens * w.unsqueeze(-1)).sum(dim=1)

        raise ValueError(f"Unknown pooling={self.pooling}")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, wp: torch.Tensor | None = None) -> torch.Tensor:
        t1 = self._forward_tokens(self.enc1, x1)
        t2 = self._forward_tokens(self.enc2, x2)
        f1 = self._pool(t1, wp)
        f2 = self._pool(t2, wp)
        fused = torch.cat([f1, f2], dim=1)
        return self.head(fused)