# -*- coding: utf-8 -*-
"""Shared constants for the model architecture and training-time augmentation.

Extracted from aigc_detector_3.py's SETTINGS block so image_transforms.py and
model.py can both import them without creating a circular import back into
the main script.
"""

# ── Model architecture ───────────────────────────────────────────────────
IMG_SIZE = 224
RGB_BACKBONE_NAME = "vit_huge_patch14_clip_224.laion2b"  # frozen — ~632M params
FREQ_BACKBONE_NAME = "convnext_base"                     # trainable — ~88M params
FUSION_DIM = 512

# ── Training augmentation ────────────────────────────────────────────────
# Fraction of training samples left completely untouched. The remainder get
# 1 or more DISTINCT transforms stacked in sequence (e.g. blur THEN JPEG
# compression) rather than exactly one.
P_CLEAN = 1 / 15  # roughly matches the old "1-of-15-keys is clean" odds
MAX_STACKED_TRANSFORMS = 3
