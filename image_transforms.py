# -*- coding: utf-8 -*-
"""Image transform pool implementing the hackathon's robustness spec table.

`TRANSFORM_POOL` is reused in three places: training-time augmentation
(`apply_random_transform_stack`), the robustness eval (`apply_named_transform`),
and the named-transform demo grid. Extracted from aigc_detector_3.py.
"""

import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T

from config import IMG_SIZE, P_CLEAN, MAX_STACKED_TRANSFORMS


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 1) * 255.0
    return Image.fromarray(arr.astype(np.uint8))


def color_jitter(img: Image.Image, delta: float = 0.2) -> Image.Image:
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-delta, delta)
        img = enhancer_cls(img).enhance(factor)
    return img


def center_crop_pct(img: Image.Image, pct: float = 0.8) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * pct), int(h * pct)
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BICUBIC)


TRANSFORM_POOL = {
    "clean": lambda img: img,
    "jpeg_90": lambda img: jpeg_compress(img, 90),
    "jpeg_70": lambda img: jpeg_compress(img, 70),
    "jpeg_50": lambda img: jpeg_compress(img, 50),
    "jpeg_30": lambda img: jpeg_compress(img, 30),
    "blur_0.5": lambda img: gaussian_blur(img, 0.5),
    "blur_1.0": lambda img: gaussian_blur(img, 1.0),
    "blur_2.0": lambda img: gaussian_blur(img, 2.0),
    "resize_0.5": lambda img: resize_roundtrip(img, 0.5),
    "resize_0.25": lambda img: resize_roundtrip(img, 0.25),
    "noise_0.02": lambda img: gaussian_noise(img, 0.02),
    "noise_0.05": lambda img: gaussian_noise(img, 0.05),
    "noise_0.10": lambda img: gaussian_noise(img, 0.10),
    "color_jitter": lambda img: color_jitter(img),
    "center_crop_80": lambda img: center_crop_pct(img, 0.8),
}

# P_CLEAN and MAX_STACKED_TRANSFORMS come from config.py. The remainder of
# samples (not left clean) get 1 or more DISTINCT transforms stacked in
# sequence (e.g. blur THEN JPEG compression) rather than exactly one —
# real-world images are often degraded by multiple processes in a row
# (resized for a thumbnail, then re-compressed on re-upload), and prior
# research on cross-generator generalization (Wang et al., CVPR 2020)
# specifically credits this kind of combined augmentation over
# single-transform augmentation.

_STACKABLE_TRANSFORM_NAMES = [k for k in TRANSFORM_POOL if k != "clean"]
_ROBUST_STACK_PRIORITY = [
    "blur_0.5",
    "blur_1.0",
    "blur_2.0",
    "resize_0.5",
    "resize_0.25",
]


def _weighted_stack_start() -> str:
    """Biases the first stacked transform toward the prompt's weak spots."""
    names = list(_STACKABLE_TRANSFORM_NAMES)
    weights = [2.5 if name in _ROBUST_STACK_PRIORITY else 1.0 for name in names]
    return random.choices(names, weights=weights, k=1)[0]


def apply_random_transform_stack(img: Image.Image) -> Image.Image:
    """
    With probability P_CLEAN, returns the image untouched. Otherwise applies
    a random number (1 to MAX_STACKED_TRANSFORMS) of DISTINCT transforms
    from TRANSFORM_POOL, in sequence — e.g. blur_1.0 then jpeg_50 then
    center_crop_80 all applied to the same image, one after another.
    random.sample (not random.choices) is used so the same transform is
    never picked twice in one stack — applying "blur_1.0" then "blur_1.0"
    again adds no realism, but "blur_1.0" then "jpeg_50" does.
    """
    if random.random() < P_CLEAN:
        return img
    k = random.randint(1, MAX_STACKED_TRANSFORMS)
    first = _weighted_stack_start()
    img = TRANSFORM_POOL[first](img)
    if k == 1:
        return img
    remaining = [name for name in _STACKABLE_TRANSFORM_NAMES if name != first]
    for name in random.sample(remaining, k=k - 1):
        img = TRANSFORM_POOL[name](img)
    return img


# CLIP's own normalization stats — NOT ImageNet's. The RGB branch is a frozen
# CLIP backbone; feeding it ImageNet-normalized input would silently mismatch
# the distribution it was pretrained on and degrade feature quality.
# These stats are shared by OpenAI's original CLIP and OpenCLIP/LAION's plain
# (non-ImageNet-fine-tuned) CLIP encoders alike — e.g. both
# "vit_large_patch14_clip_224.openai" and "vit_huge_patch14_clip_224.laion2b"
# use these exact values. Note this does NOT hold for ImageNet-fine-tuned
# variants (any name ending "_ft_in1k" / "_ft_in12k_in1k") — some of those
# switch to different normalization during fine-tuning, so verify before
# swapping to one of those specifically.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_normalize = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def to_freq_tensor(img: Image.Image) -> torch.Tensor:
    """Grayscale log-magnitude FFT spectrum, resized to IMG_SIZE, single channel."""
    gray = np.asarray(img.convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return torch.from_numpy(mag).unsqueeze(0).float()


def apply_named_transform(img: Image.Image, name: str) -> Image.Image:
    """Used by the robustness eval to apply one specific named transform."""
    return TRANSFORM_POOL[name](img)
