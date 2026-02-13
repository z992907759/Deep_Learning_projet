from __future__ import annotations
import timm
import torch
from torch import nn

class DualBranchEncoder(nn.Module):
    def __init__(self, backbone: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

    @property
    def feat_dim(self) -> int:
        return self.encoder.num_features

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            feat = self.encoder.forward_features(x)
        else:
            feat = self.encoder(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.dim() == 3:
            return feat
        if feat.dim() == 2:
            return feat.unsqueeze(1)
        raise RuntimeError(f"Unexpected token shape: {tuple(feat.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        if tokens.size(1) == 1:
            return tokens[:, 0, :]
        return tokens[:, 0, :]

class CrossViTLike(nn.Module):
    def __init__(
        self,
        backbone: str | None = None,
        backbone_small: str | None = None,
        backbone_large: str | None = None,
        num_classes: int = 2,
        pooling: str = "cls",
        cross_attn_heads: int = 8,
        fusion_dim: int | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        small_name = backbone_small or backbone or "vit_small_patch16_224"
        large_name = backbone_large or backbone or "vit_base_patch16_224"

        self.raw_enc = DualBranchEncoder(small_name, pretrained=pretrained)
        self.seg_enc = DualBranchEncoder(large_name, pretrained=pretrained)

        self.small_dim = int(self.raw_enc.feat_dim)
        self.large_dim = int(self.seg_enc.feat_dim)
        self.fusion_dim = int(fusion_dim or min(self.small_dim, self.large_dim))
        heads = int(cross_attn_heads)
        if self.fusion_dim % heads != 0:
            for h in range(heads, 0, -1):
                if self.fusion_dim % h == 0:
                    heads = h
                    break
        self.pooling = pooling

        self.small_q = nn.Linear(self.small_dim, self.fusion_dim)
        self.large_q = nn.Linear(self.large_dim, self.fusion_dim)
        self.small_kv = nn.Linear(self.small_dim, self.fusion_dim)
        self.large_kv = nn.Linear(self.large_dim, self.fusion_dim)

        self.small_to_large = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.large_to_small = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm_small = nn.LayerNorm(self.fusion_dim)
        self.norm_large = nn.LayerNorm(self.fusion_dim)
        self.cross_small_out = nn.Linear(self.fusion_dim, self.small_dim)
        self.cross_large_out = nn.Linear(self.fusion_dim, self.large_dim)

        d = self.small_dim * 2 + self.large_dim * 2
        self.head = nn.Sequential(
            nn.Linear(d, max(256, d // 2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(max(256, d // 2), num_classes),
        )

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected (B,N,C), got {tuple(tokens.shape)}")
        if tokens.size(1) == 1:
            return tokens[:, 0, :]
        if self.pooling == "cls":
            return tokens[:, 0, :]
        if self.pooling == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        raise ValueError(f"Unknown pooling={self.pooling}")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        t_small = self.raw_enc.forward_tokens(x1)
        t_large = self.seg_enc.forward_tokens(x2)

        f_small = self._pool_tokens(t_small)
        f_large = self._pool_tokens(t_large)

        q_small = self.small_q(f_small).unsqueeze(1)
        q_large = self.large_q(f_large).unsqueeze(1)
        kv_small = self.small_kv(t_small)
        kv_large = self.large_kv(t_large)

        c_small, _ = self.small_to_large(q_small, kv_large, kv_large, need_weights=False)
        c_large, _ = self.large_to_small(q_large, kv_small, kv_small, need_weights=False)
        c_small = self.cross_small_out(self.norm_small(c_small.squeeze(1)))
        c_large = self.cross_large_out(self.norm_large(c_large.squeeze(1)))

        f = torch.cat([f_small, f_large, c_small, c_large], dim=1)
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

        self.feat_dim = int(dim)

        heads = 8
        if self.feat_dim % heads != 0:
            for h in range(heads, 0, -1):
                if self.feat_dim % h == 0:
                    heads = h
                    break

        self.cross_q1 = nn.Linear(self.feat_dim, self.feat_dim)
        self.cross_q2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.cross_kv1 = nn.Linear(self.feat_dim, self.feat_dim)
        self.cross_kv2 = nn.Linear(self.feat_dim, self.feat_dim)

        self.attn_1_to_2 = nn.MultiheadAttention(
            embed_dim=self.feat_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.attn_2_to_1 = nn.MultiheadAttention(
            embed_dim=self.feat_dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm_cross1 = nn.LayerNorm(self.feat_dim)
        self.norm_cross2 = nn.LayerNorm(self.feat_dim)

        self.head = nn.Linear(self.feat_dim * 4, int(num_classes))

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

    def _forward_tokens(self, enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(enc, "forward_features"):
            feat = enc.forward_features(x)
        else:
            feat = enc(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.dim() == 3:
            return feat
        if feat.dim() == 2:
            return feat.unsqueeze(1)
        raise RuntimeError(f"Unexpected feature shape: {tuple(feat.shape)}")

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected (B,N,C), got {tuple(tokens.shape)}")
        if tokens.size(1) == 1:
            return tokens[:, 0, :]
        if self.pooling == "cls":
            return tokens[:, 0, :]
        if self.pooling == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        raise ValueError(f"Unknown pooling={self.pooling}")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        t1 = self._forward_tokens(self.enc1, x1)
        t2 = self._forward_tokens(self.enc2, x2)

        f1 = self._pool_tokens(t1)
        f2 = self._pool_tokens(t2)

        q1 = self.cross_q1(f1).unsqueeze(1)
        q2 = self.cross_q2(f2).unsqueeze(1)
        kv1 = self.cross_kv1(t1)
        kv2 = self.cross_kv2(t2)

        c1, _ = self.attn_1_to_2(q1, kv2, kv2, need_weights=False)
        c2, _ = self.attn_2_to_1(q2, kv1, kv1, need_weights=False)
        c1 = self.norm_cross1(c1.squeeze(1))
        c2 = self.norm_cross2(c2.squeeze(1))

        fused = torch.cat([f1, f2, c1, c2], dim=1)
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
