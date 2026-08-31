# -*- coding: utf-8 -*-
"""Two-branch AIGC detector model, extracted from aigc_detector_3.py.

- RGB branch: frozen CLIP ViT-H/14 (~632M params, `vit_huge_patch14_clip_224.laion2b`
  — the plain OpenCLIP/LAION-2B image encoder, NOT an ImageNet-fine-tuned variant,
  which would use different normalization and defeat the point of a general-purpose
  frozen feature extractor). Per Ojha et al., CVPR 2023 ("Towards Universal Fake
  Image Detectors that Generalize Across Generative Models"), a frozen large
  pretrained backbone + linear probe generalizes to UNSEEN generators far better
  than fine-tuning a smaller CNN end-to-end — directly relevant since WildFake is
  the out-of-distribution benchmark. Frozen also means zero gradient/optimizer-
  state memory cost for this branch, despite its size.
- Frequency branch: trainable ConvNeXt-Base (~88M params) on the FFT
  spectrum — this one DOES need to adapt to the FFT-magnitude input domain,
  so it stays trainable, unlike the RGB branch.
- Fusion: a small cross-modal attention block instead of naive
  concatenation, so the model can learn interactions between pixel content
  and frequency artifacts rather than treating them as independent evidence.

Total ~721M params — still comfortably under the hackathon's 2B cap.
Trainable ~91M of those (frequency branch + fusion + head).
"""

import timm
import torch
import torch.nn as nn

from config import RGB_BACKBONE_NAME, FREQ_BACKBONE_NAME, FUSION_DIM


class CrossModalFusion(nn.Module):
    """
    Treats the RGB and frequency branch's pooled features as a 2-token
    sequence and runs one self-attention + feed-forward block over them
    (a lightweight transformer layer). This lets each modality's features
    attend to and modulate the other before classification, rather than
    just being concatenated side by side.
    """

    def __init__(self, rgb_dim: int, freq_dim: int, fusion_dim: int = FUSION_DIM, num_heads: int = 4):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, fusion_dim)
        self.freq_proj = nn.Linear(freq_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

    def forward(self, rgb_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        rgb_tok = self.rgb_proj(rgb_feat).unsqueeze(1)    # [B, 1, D]
        freq_tok = self.freq_proj(freq_feat).unsqueeze(1)  # [B, 1, D]
        tokens = torch.cat([rgb_tok, freq_tok], dim=1)     # [B, 2, D]

        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        return tokens.flatten(1)  # [B, 2*D]


class AIGCDetector(nn.Module):
    def __init__(self,
                 rgb_backbone_name: str = RGB_BACKBONE_NAME,
                 freq_backbone_name: str = FREQ_BACKBONE_NAME,
                 fusion_dim: int = FUSION_DIM,
                 pretrained: bool = True):
        super().__init__()

        self.rgb_backbone = timm.create_model(
            rgb_backbone_name, pretrained=pretrained, num_classes=0
        )
        for p in self.rgb_backbone.parameters():
            p.requires_grad = False
        self.rgb_backbone.eval()  # frozen — never switches to train-mode behavior

        self.freq_backbone = timm.create_model(
            freq_backbone_name, pretrained=pretrained, num_classes=0, in_chans=1
        )

        self.fusion = CrossModalFusion(
            rgb_dim=self.rgb_backbone.num_features,
            freq_dim=self.freq_backbone.num_features,
            fusion_dim=fusion_dim,
        )

        self.head = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def train(self, mode: bool = True):
        # Override so calling model.train() never flips the frozen CLIP
        # backbone's internal layers (e.g. dropout, if any) into training
        # behavior — its weights never update, so its behavior shouldn't
        # change between train/eval either.
        super().train(mode)
        self.rgb_backbone.eval()
        return self

    def forward_rgb_features(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.rgb_backbone(rgb)

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.forward_rgb_features(rgb)
        freq_feat = self.freq_backbone(freq)
        fused = self.fusion(rgb_feat, freq_feat)
        logit = self.head(fused).squeeze(1)
        return logit  # apply sigmoid at inference / use BCEWithLogitsLoss for training


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
