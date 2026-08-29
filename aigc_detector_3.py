# -*- coding: utf-8 -*-
"""aigc_detector.py — AIGC Detector, TikTok TechJam Hackathon

Robust detection of AI-generated images under real-world transformations
(JPEG compression, blur, resize, noise, color jitter, center crop).

**Sections:**
0. Environment, credentials, and local cache setup (Windows/local-run specific)
1. Plan & dataset split
2. Dataset & transform pool
3. Model (two-branch: RGB + frequency)
4. Train/val/test split (leak-safe, stratified)
5. Training loop (with early stopping, cosine LR schedule, mixed precision)
6. Inference script (required deliverable: `image_dir -> JSON {image_path, pred}`)
7. Robustness evaluation (clean vs. transformed, in-distribution vs. external benchmark)

## 1. Plan

### Datasets

| Split | Source | Purpose |
|---|---|---|
| Train | CIFAKE + SID_Set (toggle via USE_CIFAKE/USE_SID_SET below) | Core training data |
| Validation | Carved out of each enabled source's TRAIN pool (VAL_SPLIT_RATIO) | Checkpoint selection, early stopping — never the official test/validation split |
| Test (in-distribution) | Each source's OFFICIAL test/validation split, touched only in Section 7 | Final honest accuracy — same distribution as training, never touched during training |
| External benchmark (out-of-distribution) | WildFake subset (COCO val2017 real + DALL·E Advanced fake) | Generalization check + required robustness table. **Never trained on.** |

### Why three eval numbers, not one
- **In-distribution test accuracy** — is the model actually good, or did it overfit?
- **Robustness-under-transform accuracy** (on the same test set) — does it survive compression/blur/crop?
- **External (WildFake) accuracy** — does it generalize to a completely different generator/photo source?

### Model
Two-branch design (see Section 4 for full detail), well under the 2B parameter cap:
- **RGB branch** — frozen CLIP ViT-L/14 (~304M params, never updated during training).
- **Frequency branch** — trainable ConvNeXt-Tiny on the log-magnitude FFT spectrum.
- **Fusion** — a small cross-modal attention block combining both branches' features.

### Robustness strategy
Augmentation-in-the-loop: every training sample gets one randomly sampled transform from the
hackathon's own spec table (or stays clean) before being fed to the model.
"""

# =============================================================================
# 0. Environment, credentials, and local cache setup
#
# This MUST run before importing kagglehub, datasets, huggingface_hub, or timm —
# those libraries read cache-location environment variables at import/first-use
# time, so setting them any later has no effect.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory into environment variables

# ── Toggle which data sources to use — flip these off to quickly iterate ──
# without waiting on a source you don't currently need (e.g. testing the
# pipeline on CIFAKE alone while SID_Set's network streaming is being flaky).
USE_CIFAKE = True
USE_SID_SET = False
SKIP_TRAINING = False  # set True to skip straight to inference/eval using an
                       # existing checkpoints/best_model.pt, without building
                       # the full training DataLoaders or downloading more
                       # than Section 7's small eval pool needs.

# Fraction of each source's TRAIN pool held back as the early-stopping
# validation set. The source's OFFICIAL test/validation split is reserved
# entirely for Section 7 — it is never touched during training or checkpoint
# selection, so it's a genuine "touched once" test set, matching the plan
# above (previously, the official test/validation split was used for BOTH
# per-epoch checkpoint selection AND the final robustness report, which
# meant the reported numbers were measured on data the checkpoint-selection
# process was implicitly tuned against).
VAL_SPLIT_RATIO = 0.15

if not USE_CIFAKE and not USE_SID_SET:
    raise RuntimeError("At least one of USE_CIFAKE or USE_SID_SET must be True.")

if USE_CIFAKE and (not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY")):
    raise RuntimeError(
        "USE_CIFAKE is True but KAGGLE_USERNAME/KAGGLE_KEY are missing. Check that "
        ".env exists in the project root and contains both values — or set "
        "USE_CIFAKE = False if you don't want CIFAKE this run."
    )
if USE_SID_SET and not os.environ.get("HF_TOKEN"):
    raise RuntimeError(
        "USE_SID_SET is True but HF_TOKEN is missing. Check that .env exists in "
        "the project root and contains a HuggingFace token — or set "
        "USE_SID_SET = False if you don't want SID_Set this run."
    )

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"
KAGGLE_CACHE_DIR = CACHE_DIR / "kagglehub"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INFERENCE_INPUT_DIR = PROJECT_ROOT / "inference_images"

for _d in (KAGGLE_CACHE_DIR, HF_CACHE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, INFERENCE_INPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Redirect CIFAKE (kagglehub) and everything HuggingFace (SID_Set + timm's
# pretrained backbone weights) into the project folder instead of the OS
# default user-profile cache.
os.environ["KAGGLEHUB_CACHE"] = str(KAGGLE_CACHE_DIR)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

# Windows without Developer Mode / admin can't create symlinks, so
# huggingface_hub falls back to plain file copies automatically — this only
# silences the (harmless) warning about that fallback, it doesn't change
# behavior.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Without this, huggingface_hub's underlying HTTP session has NO timeout on
# requests used by SID_Set's streaming — a stalled/dropped connection can
# block forever rather than ever raising an exception. That silent hang is
# indistinguishable from the training bar finishing and "nothing happening."
# Setting this turns a stall into an actual TimeoutError, which the
# HFStreamDataset resilience logic below can then catch and act on.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

print(f"Project root:      {PROJECT_ROOT}")
print(f"Kaggle cache:       {KAGGLE_CACHE_DIR}")
print(f"HuggingFace cache:  {HF_CACHE_DIR}")
print(f"Checkpoints:        {CHECKPOINT_DIR}")
print(f"Inference images:   {INFERENCE_INPUT_DIR}  (put images to check here)")
print(f"Outputs:            {OUTPUT_DIR}")

# ── CIFAKE: download from Kaggle, then read images lazily (no full unzip to RAM) ──
import kagglehub

if USE_CIFAKE:
    cifake_root = Path(kagglehub.dataset_download(
        "birdy654/cifake-real-and-ai-generated-synthetic-images"
    ))
    print("CIFAKE cached at:", cifake_root)
    print("Contents:", os.listdir(cifake_root))
else:
    cifake_root = None
    print("USE_CIFAKE is False — skipping CIFAKE download.")

# Expected layout inside cifake_root:
#   train/REAL/  train/FAKE/
#   test/REAL/   test/FAKE/

# ── SID_Set: streamed from HuggingFace ────────────────────────────────────────
from huggingface_hub import login
from datasets import load_dataset

if USE_SID_SET:
    login(token=os.environ["HF_TOKEN"])

    sid_ds_train = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    sid_ds_val = load_dataset("saberzl/SID_Set", split="validation", streaming=True)

    sample = next(iter(sid_ds_train))
    print("SID_Set sample keys:", sample.keys())
    print("Label:", sample['label'], " | Image size:", sample['image'].size)
else:
    print("USE_SID_SET is False — skipping SID_Set setup.")

# ── Dataset wrappers: one for HuggingFace streams, one for Kaggle dirs ───────

import torch
from torch.utils.data import IterableDataset
import random
from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


import hashlib

SPLIT_MANIFEST_PATH = PROJECT_ROOT / "split_manifest.json"
# Deliberately NOT inside CHECKPOINT_DIR: that folder is gitignored, but this
# file defines which exact images are train vs validation — it needs to ship
# with the repo so anyone reproducing training gets the same split, not a
# fresh random one.


def load_split_manifest() -> dict:
    """Loads the persisted train/val file assignments, or {} if none exist yet."""
    if SPLIT_MANIFEST_PATH.exists():
        with open(SPLIT_MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_split_manifest(manifest: dict) -> None:
    with open(SPLIT_MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)


def _stable_hash_fraction(key: str) -> float:
    """
    Deterministic hash of a string mapped to [0, 1), stable across process
    restarts — unlike Python's builtin hash(), which is randomized per
    process for security and would silently reassign every file on every
    run if used here.
    """
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()[:8]
    return int(digest, 16) / 0x100000000


def assign_split(paths: list, manifest_bucket: dict, val_ratio: float) -> tuple:
    """
    Assigns each path to 'train' or 'val', persistently: a path already
    present in manifest_bucket keeps its existing assignment FOREVER,
    regardless of what val_ratio is passed on this or any future call —
    only a path seen for the first time gets a fresh hash-based assignment.
    manifest_bucket is mutated in place with any new assignments.

    This is what makes changing VAL_SPLIT_RATIO safe for local file-based
    sources: it can only affect files not yet assigned, never retroactively
    move an already-trained-on file into the validation set (or vice versa).
    Returns (train_paths, val_paths, count_newly_assigned).
    """
    train_paths, val_paths = [], []
    newly_assigned = 0
    for p in paths:
        key = p.name
        assignment = manifest_bucket.get(key)
        if assignment is None:
            assignment = 'val' if _stable_hash_fraction(key) < val_ratio else 'train'
            manifest_bucket[key] = assignment
            newly_assigned += 1
        (val_paths if assignment == 'val' else train_paths).append(p)
    return train_paths, val_paths, newly_assigned


class KaggleDirStreamDataset(IterableDataset):
    """
    Streams images lazily from a directory pair (real_dir / fake_dir).
    Images are opened one at a time — no bulk loading into RAM.
    Compatible with CIFAKE's layout: train/REAL, train/FAKE, test/REAL, test/FAKE.

    NOTE: this does NOT shard samples across DataLoader workers. If you ever
    raise NUM_WORKERS above 0, every worker will iterate the FULL sample list
    independently, silently duplicating every sample once per worker. Add
    torch.utils.data.get_worker_info()-based sharding before doing that.
    """
    def __init__(self, real_dir, fake_dir, train=True, aug_in_loop=True, max_samples=None,
                 manifest: dict = None, source_key: str = "default", role: str = "train",
                 val_ratio: float = 0.15):
        self.real_paths = sorted(p for p in Path(real_dir).rglob('*') if p.suffix.lower() in IMG_EXTS)
        self.fake_paths = sorted(p for p in Path(fake_dir).rglob('*') if p.suffix.lower() in IMG_EXTS)

        if manifest is not None:
            # Separate buckets per class, so a real image and a fake image
            # that happen to share a filename can never collide in the
            # manifest lookup.
            bucket = manifest.setdefault(source_key, {})
            real_bucket = bucket.setdefault('real', {})
            fake_bucket = bucket.setdefault('fake', {})

            real_train, real_val, n_new_r = assign_split(self.real_paths, real_bucket, val_ratio)
            fake_train, fake_val, n_new_f = assign_split(self.fake_paths, fake_bucket, val_ratio)
            newly_assigned = n_new_r + n_new_f
            if newly_assigned:
                print(f"[{source_key}] assigned {newly_assigned} file(s) to train/val for "
                      f"the first time (any existing assignments were left unchanged).")

            real_split = real_train if role == 'train' else real_val
            fake_split = fake_train if role == 'train' else fake_val
        else:
            # Fallback if no manifest is wired up: use everything, unsplit.
            # Only relevant if this class is ever constructed directly
            # without going through Section 3's manifest-based setup.
            real_split, fake_split = self.real_paths, self.fake_paths

        if max_samples:
            # Split the cap evenly across classes BEFORE truncating. Doing
            # this after concatenating real+fake into one list and slicing
            # the front off is a real bug: CIFAKE's train split lists all
            # 50,000 real paths before any of the 50,000 fake ones, so any
            # max_samples <= 50,000 silently produced a ZERO-fake, real-only
            # subset — same for a cap sitting exactly at 50,000.
            per_class_cap = max_samples // 2
            real_subset = real_split[:per_class_cap]
            fake_subset = fake_split[:per_class_cap]
        else:
            real_subset = real_split
            fake_subset = fake_split

        self.samples = [(p, 0) for p in real_subset] + [(p, 1) for p in fake_subset]
        self.train = train
        self.aug_in_loop = aug_in_loop

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        paths = self.samples.copy()
        if self.train:
            random.shuffle(paths)
        for path, label in paths:
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                print(f'[warn] skipping {path}: {e}')
                continue
            if self.train and self.aug_in_loop:
                transform_name = random.choice(list(TRANSFORM_POOL.keys()))
                img = TRANSFORM_POOL[transform_name](img)
            rgb = _normalize(img)
            freq = to_freq_tensor(img)
            yield {
                'rgb': rgb,
                'freq': freq,
                'label': torch.tensor(label, dtype=torch.float32),
                'path': str(path),
            }


class HFStreamDataset(IterableDataset):
    """
    Wraps a HuggingFace streaming dataset.
    Each item must have 'image' (PIL Image) and 'label' (int: 0=real, 1=fake).

    Same worker-sharding caveat as KaggleDirStreamDataset above applies here.
    """
    def __init__(self, hf_dataset, train=True, aug_in_loop=True, max_samples=None, skip_first: int = 0):
        self.hf_dataset = hf_dataset
        self.train = train
        self.aug_in_loop = aug_in_loop
        self.max_samples = max_samples
        self.skip_first = skip_first  # discard this many items before yielding —
                                       # used to carve a disjoint early-stopping
                                       # validation slice out of the same stream
                                       # a separate instance uses for training

    def __len__(self):
        if self.max_samples is None:
            # An uncapped streaming HF dataset has no reliable length —
            # deliberately NOT implementing a fallback here, so ChainDataset
            # (and tqdm) correctly fall back to counter-only display in the
            # uncapped case, same as before this change.
            raise TypeError(
                "length is unknown for an uncapped HFStreamDataset — set max_samples to enable it."
            )
        return self.max_samples

    def __iter__(self):
        count = 0
        skipped = 0
        consecutive_errors = 0
        max_consecutive_errors = 20  # tolerate transient network blips, but don't retry forever
        iterator = iter(self.hf_dataset)

        while True:
            if self.max_samples and count >= self.max_samples:
                break

            try:
                item = next(iterator)
                consecutive_errors = 0
            except StopIteration:
                break
            except Exception as e:
                # Network hiccups, timeouts, or a momentarily unavailable shard
                # land here. Unlike a bad local file, we can't just "skip and
                # move on" as cleanly — next(iterator) may or may not recover
                # on its own. Skip a bounded number of consecutive failures,
                # but stop for real if it looks like the stream is actually
                # broken rather than just having a bad moment.
                consecutive_errors += 1
                print(f'[warn] SID_Set stream error ({consecutive_errors}/{max_consecutive_errors}): {e}')
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f'SID_Set streaming failed {max_consecutive_errors} times in a row — '
                        f'this looks like a persistent problem (network connectivity, an expired '
                        f'HF_TOKEN, or a gated-access issue), not a transient blip. Stopping '
                        f'instead of retrying forever so this doesn\'t look like a silent hang.'
                    ) from e
                continue

            if skipped < self.skip_first:
                skipped += 1
                continue

            try:
                img = item['image'].convert('RGB')
                label = int(item['label'])
            except Exception as e:
                print(f'[warn] skipping a malformed SID_Set sample: {e}')
                continue

            if self.train and self.aug_in_loop:
                transform_name = random.choice(list(TRANSFORM_POOL.keys()))
                img = TRANSFORM_POOL[transform_name](img)
            rgb = _normalize(img)
            freq = to_freq_tensor(img)
            yield {
                'rgb': rgb,
                'freq': freq,
                'label': torch.tensor(label, dtype=torch.float32),
                'path': str(item.get('image_path', f'stream_{count}')),
            }
            count += 1


class InterleavedIterableDataset(IterableDataset):
    """
    Interleaves multiple IterableDatasets by randomly picking which still-
    active source to draw the next sample from, instead of ChainDataset's
    behavior of fully exhausting one source before starting the next.

    Without this, every single epoch sees a hard "regime switch" at the same
    relative point (e.g. all of CIFAKE, then all of SID_Set) rather than a
    well-mixed blend of sources — a real, if likely minor, training-dynamics
    quirk that a plain ChainDataset can't avoid.
    """
    def __init__(self, datasets: list):
        self.datasets = datasets

    def __len__(self):
        # Mirrors ChainDataset's own __len__: sums each source's length,
        # propagating TypeError if any source's length is unknown (e.g. an
        # uncapped HFStreamDataset) so callers fall back the same way.
        return sum(len(d) for d in self.datasets)

    def __iter__(self):
        iterators = [iter(d) for d in self.datasets]
        active = list(range(len(iterators)))
        while active:
            idx = random.choice(active)
            try:
                yield next(iterators[idx])
            except StopIteration:
                active.remove(idx)


print('Dataset wrappers defined. No images loaded yet.')

"""## 2. Setup"""

import io
import csv
import json
import sys

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm
from PIL import ImageEnhance, ImageFilter

import timm
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

"""## 2b. Dataset & transform pool

Implements the exact transform table from the problem statement. `TRANSFORM_POOL` is reused in
three places: training-time augmentation, the robustness eval, and the (optional) named-transform
demo in your video.
"""

IMG_SIZE = 224


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

# CLIP's own normalization stats — NOT ImageNet's. The RGB branch is now a
# frozen CLIP backbone; feeding it ImageNet-normalized input would silently
# mismatch the distribution it was pretrained on and degrade feature quality.
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


def get_hf_split_total(repo_id: str, split: str):
    """
    Looks up a HuggingFace dataset split's true sample count from its
    metadata (dataset_infos.json / README YAML), without downloading or
    streaming any actual data. Returns None — rather than raising — if this
    isn't available for this dataset/split, so the caller can report exactly
    which source couldn't be determined instead of crashing.
    """
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(repo_id)
        if not builder.info.splits or split not in builder.info.splits:
            return None
        return builder.info.splits[split].num_examples  # may itself be None
    except Exception as e:
        print(f"[warn] could not fetch size metadata for {repo_id} split={split}: {e}")
        return None


"""## 3. Train / validation split, DataLoaders

CIFAKE  -> KaggleDirStreamDataset  (reads from the project-local kagglehub cache)
SID_Set -> HFStreamDataset         (HuggingFace streaming, no disk write)

Neither source requires the full dataset to be in RAM or copied elsewhere.

IMPORTANT: the early-stopping validation set used here is carved out of each
source's TRAIN pool (via VAL_SPLIT_RATIO), NOT from the official test/
validation split. The official test/validation split is reserved entirely
for Section 7 — it is never touched during training or checkpoint selection,
so it's a genuine "touched once" test set. Previously, the official
test/validation split served double duty as both the checkpoint-selection
signal and the final robustness report, which meant those numbers were
measured on data the checkpoint-selection process was implicitly tuned
against.
"""

from torch.utils.data import ChainDataset

MAX_SAMPLES_PER_SOURCE = None  # e.g. 20_000 to cap; None = use all
BATCH_SIZE = 64
# Lowered from 128 (the EfficientNet-B0 baseline's setting): CLIP ViT-L/14's
# forward pass has real activation-memory cost even frozen and gradient-free.
# Watch nvidia-smi on your first epoch and raise this back up if there's
# headroom left on your 12GB card — this is a conservative starting point,
# not a measured limit.
#
# NUM_WORKERS is 0 deliberately, for two independent reasons:
#   1. Windows uses "spawn" (not "fork") for multiprocessing, which requires
#      all top-level script code to be re-import-safe under a `if __name__ ==
#      "__main__":` guard — this script doesn't have that structure, so any
#      value above 0 crashes with "An attempt has been made to start a new
#      process before the current process has finished its bootstrapping
#      phase."
#   2. Even with that guard added, KaggleDirStreamDataset and HFStreamDataset
#      above do not shard samples across workers, so num_workers>0 would
#      silently duplicate every sample once per worker rather than just
#      crashing. Both issues need fixing together before raising this.
NUM_WORKERS = 0

if SKIP_TRAINING:
    print("SKIP_TRAINING is True — skipping training DataLoader construction "
          "entirely (Section 7's small eval pool is built independently later).")
    train_loader = None
    val_loader = None
    cifake_train_ds = cifake_earlystop_val_ds = None
    sid_train_ds = sid_earlystop_val_ds = None
else:
    train_sources = []
    val_sources = []

    # Loaded once per run — mutated in place as new files get their one-time
    # assignment, then saved back to disk after all manifest-based sources
    # are constructed below.
    _split_manifest = load_split_manifest()

    if USE_CIFAKE:
        cifake_train_ds = KaggleDirStreamDataset(
            real_dir=cifake_root / 'train' / 'REAL',
            fake_dir=cifake_root / 'train' / 'FAKE',
            train=True, aug_in_loop=True,
            max_samples=MAX_SAMPLES_PER_SOURCE,
            manifest=_split_manifest, source_key="CIFAKE", role="train",
            val_ratio=VAL_SPLIT_RATIO,
        )
        cifake_earlystop_val_ds = KaggleDirStreamDataset(
            real_dir=cifake_root / 'train' / 'REAL',
            fake_dir=cifake_root / 'train' / 'FAKE',
            train=False, aug_in_loop=False,
            max_samples=MAX_SAMPLES_PER_SOURCE,
            manifest=_split_manifest, source_key="CIFAKE", role="val",
            val_ratio=VAL_SPLIT_RATIO,
        )
        print(f'CIFAKE train:       {len(cifake_train_ds.samples):,} images')
        print(f'CIFAKE early-stop val: {len(cifake_earlystop_val_ds.samples):,} images')
        train_sources.append(cifake_train_ds)
        val_sources.append(cifake_earlystop_val_ds)
    else:
        cifake_train_ds = None
        cifake_earlystop_val_ds = None

    save_split_manifest(_split_manifest)

    if USE_SID_SET:
        sid_train_stream = load_dataset('saberzl/SID_Set', split='train', streaming=True)
        sid_earlystop_val_stream = load_dataset('saberzl/SID_Set', split='train', streaming=True)

        _sid_train_total = get_hf_split_total('saberzl/SID_Set', 'train')
        _SID_FALLBACK_SPLIT_SIZE = 50_000  # only used if size is unknown AND uncapped

        if _sid_train_total is not None:
            _sid_val_size = max(1, int(_sid_train_total * VAL_SPLIT_RATIO))
            _sid_train_cap = _sid_train_total - _sid_val_size
        elif MAX_SAMPLES_PER_SOURCE is not None:
            _sid_val_size = max(1, int(MAX_SAMPLES_PER_SOURCE * VAL_SPLIT_RATIO))
            _sid_train_cap = MAX_SAMPLES_PER_SOURCE - _sid_val_size
        else:
            print(f"[warn] SID_Set train split size is unknown AND MAX_SAMPLES_PER_SOURCE "
                  f"is not set — falling back to a fixed {_SID_FALLBACK_SPLIT_SIZE:,}-sample "
                  f"boundary so train/early-stop-val stay disjoint. Set MAX_SAMPLES_PER_SOURCE "
                  f"for a split that's properly sized relative to the real split.")
            _sid_val_size = max(1, int(_SID_FALLBACK_SPLIT_SIZE * VAL_SPLIT_RATIO))
            _sid_train_cap = _SID_FALLBACK_SPLIT_SIZE - _sid_val_size

        if MAX_SAMPLES_PER_SOURCE is not None:
            _sid_train_cap = min(_sid_train_cap, MAX_SAMPLES_PER_SOURCE)

        # skip_first=_sid_train_cap guarantees the val slice starts exactly
        # where the train slice's cap ends — disjoint by construction,
        # regardless of which branch above computed the numbers.
        sid_train_ds = HFStreamDataset(sid_train_stream, train=True, aug_in_loop=True,
                                        max_samples=_sid_train_cap, skip_first=0)
        sid_earlystop_val_ds = HFStreamDataset(sid_earlystop_val_stream, train=False, aug_in_loop=False,
                                                max_samples=_sid_val_size, skip_first=_sid_train_cap)
        train_sources.append(sid_train_ds)
        val_sources.append(sid_earlystop_val_ds)
    else:
        sid_train_ds = None
        sid_earlystop_val_ds = None

    if not train_sources:
        raise RuntimeError("No data sources ended up enabled — check USE_CIFAKE/USE_SID_SET above.")

    # Training data is interleaved (mixed sample-by-sample across sources)
    # rather than chained (all of source A, then all of source B) — see
    # InterleavedIterableDataset's docstring for why this matters. Validation
    # order doesn't affect the aggregate accuracy metric, so it stays a
    # simple ChainDataset for simplicity.
    train_ds = InterleavedIterableDataset(train_sources) if len(train_sources) > 1 else train_sources[0]
    val_ds = ChainDataset(val_sources)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    print('\nDataLoaders ready. Images will be read lazily — no bulk copy elsewhere.')




def summarize_sample_usage(sources: list):
    """
    sources: list of dicts with keys:
      name            - display name, e.g. "CIFAKE (train)"
      total_available - true sample count, or None if unknown
      used            - samples this run will actually draw from this source
                         (already capped/balanced — the real number, not the
                         raw requested cap)
      used_is_upper_bound - True if `used` is only an upper bound because
                         total_available was unknown (i.e. the real number
                         could turn out smaller once the stream is consumed)
    Prints a table and a combined total, explicitly flagging any source
    whose true size couldn't be determined rather than silently guessing.
    """
    print(f"\n{'Source':<22} {'Used':>12} {'Total_Sample':>14}")
    print("-" * 50)

    unknown_sources = []
    combined_used = 0
    combined_total_known = True
    combined_total = 0

    for src in sources:
        used_str = f"{src['used']:,}" + ("*" if src['used_is_upper_bound'] else "")
        total_str = f"{src['total_available']:,}" if src['total_available'] is not None else "unknown"
        print(f"{src['name']:<22} {used_str:>12} {total_str:>14}")

        combined_used += src['used']
        if src['total_available'] is None:
            combined_total_known = False
            unknown_sources.append(src['name'])
        else:
            combined_total += src['total_available']

    print("-" * 50)
    combined_total_str = f"{combined_total:,}" if combined_total_known else f"{combined_total:,}+ (incomplete)"
    print(f"{'Combined_Total_Sample':<22} {combined_used:,} used out of {combined_total_str} total")

    if any(s['used_is_upper_bound'] for s in sources):
        print("(* = upper bound; true availability unknown, actual samples used may be fewer)")

    if unknown_sources:
        print(f"\nCannot obtain sample size from {len(unknown_sources)} source(s): "
              f"{', '.join(unknown_sources)}")
    else:
        print("\nSample sizes obtained for all sources.")


if SKIP_TRAINING:
    print("\n(Sample-usage summary skipped — no training DataLoaders were built this run.)")
else:
    _summary_sources = []

    if USE_CIFAKE:
        _cifake_train_total = len(cifake_train_ds.real_paths) + len(cifake_train_ds.fake_paths)
        _cifake_val_total = len(cifake_earlystop_val_ds.real_paths) + len(cifake_earlystop_val_ds.fake_paths)
        _summary_sources.append({"name": "CIFAKE (train)", "total_available": _cifake_train_total,
                                  "used": len(cifake_train_ds.samples), "used_is_upper_bound": False})
        _summary_sources.append({"name": "CIFAKE (early-stop val)", "total_available": _cifake_val_total,
                                  "used": len(cifake_earlystop_val_ds.samples), "used_is_upper_bound": False})

    if USE_SID_SET:
        # Reuse the split numbers already computed in Section 3 rather than
        # re-fetching metadata — _sid_train_total/_sid_train_cap/_sid_val_size
        # were computed there when constructing sid_train_ds/sid_earlystop_val_ds.
        _summary_sources.append({"name": "SID_Set (train)", "total_available": _sid_train_total,
                                  "used": _sid_train_cap, "used_is_upper_bound": _sid_train_total is None})
        _summary_sources.append({"name": "SID_Set (early-stop val)", "total_available": _sid_train_total,
                                  "used": _sid_val_size, "used_is_upper_bound": _sid_train_total is None})

    summarize_sample_usage(_summary_sources)





"""## 4. Model

Two-branch design, upgraded from the EfficientNet-B0 baseline:

- **RGB branch**: frozen CLIP ViT-L/14 (~304M params). Per Ojha et al., CVPR
  2023 ("Towards Universal Fake Image Detectors that Generalize Across
  Generative Models"), a frozen large pretrained backbone + linear probe
  generalizes to UNSEEN generators far better than fine-tuning a smaller CNN
  end-to-end — directly relevant since WildFake is your out-of-distribution
  benchmark. Frozen also means zero gradient/optimizer-state memory cost for
  this branch, despite its size.
- **Frequency branch**: trainable ConvNeXt-Tiny (~28M params) on the FFT
  spectrum — this one DOES need to adapt to the FFT-magnitude input domain,
  so it stays trainable, unlike the RGB branch.
- **Fusion**: a small cross-modal attention block instead of naive
  concatenation, so the model can learn interactions between pixel content
  and frequency artifacts rather than treating them as independent evidence.

Total ~332M params — still comfortably under the hackathon's 2B cap.
"""


class CrossModalFusion(nn.Module):
    """
    Treats the RGB and frequency branch's pooled features as a 2-token
    sequence and runs one self-attention + feed-forward block over them
    (a lightweight transformer layer). This lets each modality's features
    attend to and modulate the other before classification, rather than
    just being concatenated side by side.
    """

    def __init__(self, rgb_dim: int, freq_dim: int, fusion_dim: int = 512, num_heads: int = 4):
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
                 rgb_backbone_name: str = "vit_large_patch14_clip_224.openai",
                 freq_backbone_name: str = "convnext_tiny",
                 fusion_dim: int = 512,
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


if SKIP_TRAINING:
    print("(Parameter-count sanity check skipped — SKIP_TRAINING is True, "
          "no need to load a throwaway model just to print this.)")
else:
    _sanity_model = AIGCDetector()
    print(f"Total params: {count_params(_sanity_model):,} (Limit: 2B)")
    print(f"Trainable params: {count_trainable_params(_sanity_model):,} "
          f"(the rest is the frozen CLIP backbone)")
    del _sanity_model

"""## 5. Training loop

Optimizations over the baseline version:
  - BATCH_SIZE raised from 32 -> 128 (the model is tiny — 8-9M params — so a
    12GB GPU has plenty of headroom; watch nvidia-smi and raise further if
    there's room).
  - Cosine LR schedule instead of a flat learning rate.
  - Automatic mixed precision (torch.cuda.amp) for speed/memory on the 4070 Ti.
  - Early stopping: EPOCHS is a high ceiling, PATIENCE actually decides when
    training stops.
  - resume_from: continue training from a saved checkpoint instead of always
    starting over from pretrained ImageNet weights.
  - Checkpoints save to the project-local CHECKPOINT_DIR, not a Colab path.
"""

import shutil
import datetime
import time

EPOCHS = 30      # high ceiling — early stopping will cut this short in practice
PATIENCE = 5     # stop if val accuracy hasn't improved in this many epochs
LR = 1e-4


def _format_duration(seconds: float) -> str:
    """Formats a seconds count as HH:MM:SS for readable ETA display."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _active_sources_snapshot() -> dict:
    """
    Snapshot of which data sources are enabled right now. Stored in every
    checkpoint so a later resume can detect a change in what data is being
    trained on — since that silently changes what best_val_acc and the
    early-stopping patience counter even mean (they were calibrated against
    whatever validation mix existed at save time, not necessarily the one
    the resumed run will use). Add new source toggles here if more get added.
    """
    return {'USE_CIFAKE': USE_CIFAKE, 'USE_SID_SET': USE_SID_SET}


def _data_volume_snapshot() -> dict:
    """
    Snapshot of data-volume-related settings and observed dataset sizes for
    the CURRENTLY enabled sources. Stored in every checkpoint so a resume can
    detect a change in how much data existing sources are contributing —
    either from a config change (MAX_SAMPLES_PER_SOURCE) or from the
    underlying dataset itself growing/shrinking between runs.

    CIFAKE tracks its ACTUAL resulting split sizes (cifake_train_used /
    cifake_val_used), not the raw VAL_SPLIT_RATIO — the persisted split
    manifest makes CIFAKE's real outcome independent of that constant for
    any file already assigned, so comparing the raw ratio would flag a
    "change" even when literally nothing about the actual data differs
    (verified: editing the ratio alone, with no new files, produces byte-
    identical train/val file sets once the manifest is populated). Tracking
    the real outcome instead avoids an unnecessary re-baseline in that case.

    SID_Set still has no manifest (its streaming rows have no stable
    identity to hash against), so VAL_SPLIT_RATIO directly determines which
    files land where for it — it stays a necessary, direct signal there.
    """
    snapshot = {'MAX_SAMPLES_PER_SOURCE': MAX_SAMPLES_PER_SOURCE}
    if USE_CIFAKE:
        snapshot['cifake_total_available'] = (
            len(cifake_train_ds.real_paths) + len(cifake_train_ds.fake_paths)
        )
        snapshot['cifake_train_used'] = len(cifake_train_ds.samples)
        snapshot['cifake_val_used'] = len(cifake_earlystop_val_ds.samples)
    if USE_SID_SET:
        snapshot['sid_total_available'] = _sid_train_total
        snapshot['VAL_SPLIT_RATIO'] = VAL_SPLIT_RATIO
    return snapshot


def persist_checkpoint(local_path: str, dest_dir: str = str(CHECKPOINT_DIR)) -> Path:
    """Copies a checkpoint to a timestamped file so repeated runs don't overwrite each other."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(dest_dir) / f"best_model_{timestamp}.pt"
    shutil.copy(local_path, dest)
    print(f"Checkpoint snapshot saved to {dest}")
    return dest


LATEST_CHECKPOINT_PATH = str(CHECKPOINT_DIR / "latest_checkpoint.pt")


def train_model(train_loader, val_loader, epochs=None, patience=PATIENCE,
                 checkpoint_path=str(CHECKPOINT_DIR / "best_model.pt"),
                 latest_checkpoint_path=LATEST_CHECKPOINT_PATH,
                 resume_from=None):
    model = AIGCDetector().to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    start_epoch = 0
    best_val_acc = 0.0
    epochs_without_improvement = 0
    ckpt = None

    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=DEVICE)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        # Full checkpoint (this script's format): restores weights AND the
        # optimizer/scheduler/scaler state, so this is a true continuation,
        # not just a warm weight restart.
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        epochs_without_improvement = ckpt['epochs_without_improvement']

        if epochs is None:
            epochs = ckpt['epochs']  # keep the original run's ceiling so the
                                      # cosine LR schedule's shape isn't distorted
        elif epochs < ckpt['epochs']:
            raise RuntimeError(
                f"epochs={epochs} is LESS than this checkpoint's original ceiling "
                f"({ckpt['epochs']}). Shrinking the ceiling on a resume is almost "
                f"always a stale EPOCHS value rather than an intentional choice, and "
                f"proceeding silently could badly distort the LR schedule or produce "
                f"an empty training loop — refusing to continue. Set EPOCHS >= "
                f"{ckpt['epochs']}, or delete/rename the checkpoint if you genuinely "
                f"want to restart with a smaller ceiling."
            )
        elif epochs != ckpt['epochs']:
            print(f"Extending LR schedule ceiling from {ckpt['epochs']} to {epochs}.")

        print(f"Resumed from {resume_from}: continuing at epoch {start_epoch + 1}, "
              f"best_val_acc so far = {best_val_acc:.4f}, "
              f"epochs_without_improvement = {epochs_without_improvement}")

        saved_sources = ckpt.get('active_sources')  # None for checkpoints saved before this existed
        current_sources = _active_sources_snapshot()
        sources_changed = saved_sources is not None and saved_sources != current_sources
        if sources_changed:
            print(f"NOTICE: the enabled data sources have changed since this checkpoint "
                  f"was saved.\n"
                  f"  Then: {saved_sources}\n"
                  f"  Now:  {current_sources}")
        elif saved_sources is None:
            print("Note: this checkpoint predates source-tracking, so a data-source-change "
                  "check couldn't be performed on resume.")

        saved_volume = ckpt.get('data_volume')
        current_volume = _data_volume_snapshot()
        val_split_changed = False
        volume_changed = False
        if saved_volume is not None and saved_volume != current_volume:
            changed_keys = {k for k in current_volume if saved_volume.get(k) != current_volume.get(k)}
            # VAL_SPLIT_RATIO only ever appears in current_volume when
            # USE_SID_SET is True (see _data_volume_snapshot) — CIFAKE tracks
            # its actual resulting split sizes instead, since the persisted
            # manifest makes the raw ratio alone a non-signal for it.
            val_split_changed = 'VAL_SPLIT_RATIO' in changed_keys
            volume_changed = bool(changed_keys - {'VAL_SPLIT_RATIO'})

            if val_split_changed:
                print(f"WARNING: VAL_SPLIT_RATIO changed since this checkpoint was saved "
                      f"({saved_volume.get('VAL_SPLIT_RATIO')} -> {current_volume.get('VAL_SPLIT_RATIO')}). "
                      f"This is a MORE SERIOUS issue than a plain size change: SID_Set's "
                      f"streaming source still uses proportional splitting (not the "
                      f"persisted manifest CIFAKE uses), so the train/validation boundary "
                      f"within it has shifted — some images now labeled 'validation' may "
                      f"actually have already been trained on in earlier epochs, real "
                      f"leakage rather than just a different validation size. Strongly "
                      f"consider deleting this checkpoint and starting fresh rather than "
                      f"resuming across a VAL_SPLIT_RATIO change while SID_Set is enabled "
                      f"— this is NOT auto-recalibrated below, unlike a plain "
                      f"source/volume increase.")

            if volume_changed:
                details = {k: (saved_volume.get(k), current_volume.get(k))
                           for k in changed_keys - {'VAL_SPLIT_RATIO'}}
                print(f"NOTICE: the amount of available training data has changed since "
                      f"this checkpoint was saved (shown as (then, now) per setting): "
                      f"{details}.")
        elif saved_volume is None:
            print("Note: this checkpoint predates data-volume tracking, so a data-volume-"
                  "change check couldn't be performed on resume.")

        # More data or a new source is a genuine opportunity to improve, not just a
        # bookkeeping problem — so when the change is safe (no VAL_SPLIT_RATIO shift),
        # actually USE it: re-measure the already-trained model against the new
        # validation mix right now, and treat that fresh number as the new bar to
        # beat, with patience reset to give it a fair shot. Skipped entirely for a
        # VAL_SPLIT_RATIO change, since re-baselining across a leakage risk would
        # paper over the real problem instead of fixing it.
        if (sources_changed or volume_changed) and not val_split_changed:
            print(f"Re-evaluating the resumed model against the updated validation set "
                  f"to get a fair new baseline (the old best_val_acc of {best_val_acc:.4f} "
                  f"was measured on a different data mix)...")
            best_val_acc = evaluate_accuracy(model, val_loader)
            epochs_without_improvement = 0
            print(f"New baseline best_val_acc = {best_val_acc:.4f}. Training will treat "
                  f"this as the mark to beat going forward, and patience has been reset "
                  f"so the model gets a fair chance to improve against the expanded data "
                  f"rather than being judged against a stale comparison. If a lot more "
                  f"data is now available, consider also raising EPOCHS to make full use "
                  f"of it.")
    elif ckpt is not None:
        # Legacy checkpoint: a bare state_dict with no training state alongside
        # it (this is what earlier versions of this script saved). Weights load
        # fine; optimizer/scheduler/epoch counters just start fresh.
        model.load_state_dict(ckpt)
        print(f"Resumed model weights only (legacy checkpoint format) from {resume_from} "
              f"— optimizer/scheduler/epoch counters were not saved in this format, "
              f"starting those fresh.")

    if epochs is None:
        epochs = EPOCHS

    if start_epoch >= epochs:
        raise RuntimeError(
            f"This checkpoint already completed through epoch {start_epoch} (next epoch "
            f"to run would be 0-indexed epoch {start_epoch}), which leaves nothing to "
            f"train toward a ceiling of {epochs} — this would otherwise silently run zero "
            f"epochs. Raise EPOCHS above {start_epoch}, or set SKIP_TRAINING = True if you "
            f"just want to use the existing model as-is."
        )

    print(f'Model params: {count_params(model):,} (Limit: 2B)')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    if isinstance(ckpt, dict) and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # load_state_dict restores T_max from the CHECKPOINT, silently
        # discarding whatever `epochs` was just passed in. Cosine annealing
        # is periodic, so if that's left unfixed, continuing to step past the
        # OLD T_max makes the LR climb back up toward its original starting
        # value instead of staying low — exactly the wrong behavior when
        # resuming to keep fine-tuning an already-partially-trained model.
        # Re-apply the current ceiling explicitly so it actually takes effect.
        if scheduler.T_max != epochs:
            print(f"Adjusting LR schedule: extending T_max from {scheduler.T_max} to {epochs} "
                  f"so the cosine decay continues smoothly to the new ceiling instead of "
                  f"restarting/climbing back up past the original one.")
            scheduler.T_max = epochs
    if isinstance(ckpt, dict) and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    criterion = nn.BCEWithLogitsLoss()
    epoch_durations = []  # tracks wall-clock seconds per epoch, for the ETA estimate below

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch in pbar:
            rgb = batch['rgb'].to(DEVICE)
            freq = batch['freq'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                loss = criterion(model(rgb, freq), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader)

        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
        # Estimate against the epochs ceiling, not against unknown future
        # early-stopping — there's no way to know in advance which epoch
        # will trigger it, so this is "time left if training runs to the
        # ceiling," not a promise of exactly when it'll finish.
        epochs_remaining = epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * epochs_remaining

        print(f'Epoch {epoch+1}: Loss={total_loss/max(n_batches,1):.4f}, '
              f'Val Acc={val_acc:.4f}, LR={scheduler.get_last_lr()[0]:.2e}, '
              f'Time={_format_duration(epoch_duration)}, '
              f'Avg/epoch={_format_duration(avg_epoch_time)}, '
              f'Est. remaining (to ceiling)={_format_duration(eta_seconds)}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'epochs': epochs,
                'best_val_acc': best_val_acc,
                'epochs_without_improvement': epochs_without_improvement,
                'active_sources': _active_sources_snapshot(),
                'data_volume': _data_volume_snapshot(),
            }, checkpoint_path)
            print(f'  --> Saved new best model (val_acc={val_acc:.4f})')
        else:
            epochs_without_improvement += 1

        # Saved EVERY epoch, regardless of improvement — unlike checkpoint_path
        # (best_model.pt) above, which only saves on improvement and therefore
        # can only ever persist epochs_without_improvement=0 (it's always reset
        # to 0 right before that save). This is what makes epochs_without_
        # improvement genuinely restorable on resume, and lets a resume
        # continue from the actual last-trained weights instead of always
        # rewinding to the last improving epoch (re-doing any non-improving
        # epochs that happened after it).
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch,
            'epochs': epochs,
            'best_val_acc': best_val_acc,
            'epochs_without_improvement': epochs_without_improvement,
            'active_sources': _active_sources_snapshot(),
            'data_volume': _data_volume_snapshot(),
        }, latest_checkpoint_path)

        if epochs_without_improvement >= patience:
            print(f'No improvement for {patience} epochs, stopping early at epoch {epoch+1}.')
            break

    return model


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    # Previously this loop had zero visual feedback — after the training bar
    # hit 100%, the whole validation pass (which also streams SID_Set data)
    # ran silently, looking indistinguishable from a hang. This bar makes
    # that time visible instead of invisible.
    pbar = tqdm(loader, desc='Validating', leave=False)
    for batch in pbar:
        rgb = batch['rgb'].to(DEVICE)
        freq = batch['freq'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        preds = (torch.sigmoid(model(rgb, freq)) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if total > 0:
            pbar.set_postfix(acc=f'{correct/total:.4f}')
    return correct / total if total > 0 else 0.0


_default_checkpoint = str(CHECKPOINT_DIR / "best_model.pt")
_latest_checkpoint = str(CHECKPOINT_DIR / "latest_checkpoint.pt")

if SKIP_TRAINING:
    if not Path(_default_checkpoint).exists():
        raise RuntimeError(
            f"SKIP_TRAINING is True, but no checkpoint exists at {_default_checkpoint}. "
            f"There's nothing for inference/robustness-eval to load — either run with "
            f"SKIP_TRAINING = False first, or point this at a checkpoint you already have."
        )
    print(f"SKIP_TRAINING is True — skipping straight to inference/eval using "
          f"the existing checkpoint at {_default_checkpoint}.")
else:
    if Path(_latest_checkpoint).exists():
        print(f"Found {_latest_checkpoint} — resuming training from it (this reflects "
              f"the exact epoch and patience state training last left off at).")
        _resume_from = _latest_checkpoint
    elif Path(_default_checkpoint).exists():
        print(f"No 'latest' checkpoint found, but {_default_checkpoint} exists — "
              f"resuming from it. Note: only the 'latest' checkpoint format preserves "
              f"patience state, so epochs_without_improvement will restart fresh here.")
        _resume_from = _default_checkpoint
    else:
        print("No existing checkpoint found — starting training from scratch.")
        _resume_from = None

    model = train_model(train_loader, val_loader, resume_from=_resume_from, epochs=EPOCHS)
    persist_checkpoint(str(CHECKPOINT_DIR / "best_model.pt"))

"""## 6. Inference (required deliverable)

`image_dir -> JSON [{"image_path": ..., "pred": confidence}]` per the submission spec.
"""

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(checkpoint_path: str) -> AIGCDetector:
    """
    Loads a model for pure inference (no optimizer/scheduler needed here).
    Handles both this script's full-checkpoint format (a dict with
    'model_state_dict') and older bare-state-dict checkpoints transparently.
    """
    model = AIGCDetector(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def predict_one(model: AIGCDetector, image_path: Path) -> float:
    img = Image.open(image_path).convert("RGB")
    rgb = _normalize(img).unsqueeze(0).to(DEVICE)
    freq = to_freq_tensor(img).unsqueeze(0).to(DEVICE)
    logit = model(rgb, freq)
    return torch.sigmoid(logit).item()


def run_inference(image_dir: str,
                   checkpoint_path: str = str(CHECKPOINT_DIR / "best_model.pt"),
                   out_path: str = str(OUTPUT_DIR / "preds.json")):
    model = load_model(checkpoint_path)
    image_paths = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in VALID_EXT)

    results = []
    for p in image_paths:
        try:
            score = predict_one(model, p)
        except Exception as e:
            print(f"[warn] failed on {p}: {e}", file=sys.stderr)
            continue
        results.append({"image_path": str(p), "pred": score})

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {out_path}")
    return results


# Example — drop images into the INFERENCE_INPUT_DIR folder printed at
# startup, then uncomment this line to run the checker against them:
# run_inference(str(INFERENCE_INPUT_DIR))

"""## 7. Robustness evaluation (deliverable #4 + #5)

Because we're streaming, the robustness eval collects a **capped sample** into memory
(default 2,000 images) for repeatable per-transform scoring. This is fine for eval —
we never need the full dataset in RAM at once during training.

Produces the clean-vs-transformed table. Run **twice**:
1. On the held-out CIFAKE test + SID_Set validation stream (in-distribution)
2. On the external WildFake benchmark CSV (out-of-distribution, never trained on)
"""

from itertools import islice

EVAL_CAP = 2000  # total images; raise if you have RAM to spare
_active_source_count = sum([USE_CIFAKE, USE_SID_SET])
per_source_cap = EVAL_CAP // _active_source_count

eval_samples_raw = []

if USE_CIFAKE:
    cifake_eval_paths = (
        [(p, 0) for p in sorted((cifake_root / 'test' / 'REAL').rglob('*')) if p.suffix.lower() in IMG_EXTS] +
        [(p, 1) for p in sorted((cifake_root / 'test' / 'FAKE').rglob('*')) if p.suffix.lower() in IMG_EXTS]
    )[:per_source_cap]

    for path, label in cifake_eval_paths:
        try:
            eval_samples_raw.append((Image.open(path).convert('RGB'), label))
        except Exception as e:
            print(f'[warn] {path}: {e}')

if USE_SID_SET:
    sid_eval_stream = load_dataset('saberzl/SID_Set', split='validation', streaming=True)
    for item in islice(sid_eval_stream, per_source_cap):
        eval_samples_raw.append((item['image'].convert('RGB'), int(item['label'])))

print(f'Eval pool: {len(eval_samples_raw)} images ready.')


@torch.no_grad()
def run_transform_eval_pil(model, samples_raw: list, transform_name: str):
    """samples_raw: list of (PIL Image, label)"""
    preds, labels = [], []
    for img, label in samples_raw:
        img_t = apply_named_transform(img, transform_name)
        rgb = _normalize(img_t).unsqueeze(0).to(DEVICE)
        freq = to_freq_tensor(img_t).unsqueeze(0).to(DEVICE)
        score = torch.sigmoid(model(rgb, freq)).item()
        preds.append(score)
        labels.append(label)

    pred_labels = [1 if p > 0.5 else 0 for p in preds]
    acc = sum(int(p == l) for p, l in zip(pred_labels, labels)) / len(labels)
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else float('nan')
    return acc, auc, preds, labels


def build_robustness_table_streaming(model, samples_raw,
                                      out_csv=str(OUTPUT_DIR / 'robustness_table.csv')):
    rows = []
    for name in TRANSFORM_POOL:
        acc, auc, _, _ = run_transform_eval_pil(model, samples_raw, name)
        rows.append({'transform': name, 'accuracy': round(acc, 4), 'auc': round(auc, 4)})
        print(f'{name:20s}  acc={acc:.4f}  auc={auc:.4f}')

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['transform', 'accuracy', 'auc'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote robustness table -> {out_csv}')
    return rows


def error_analysis_pil(model, samples_raw, transform_name='clean', top_k=10):
    _, _, preds, labels = run_transform_eval_pil(model, samples_raw, transform_name)
    records = [(f'sample_{i}', p, l) for i, (p, l) in enumerate(zip(preds, labels))]

    false_positives = sorted(
        [r for r in records if r[2] == 0 and r[1] > 0.5], key=lambda r: -r[1]
    )[:top_k]
    false_negatives = sorted(
        [r for r in records if r[2] == 1 and r[1] < 0.5], key=lambda r: r[1]
    )[:top_k]

    print(f'Top {top_k} false positives (real flagged as fake):')
    for name, score, _ in false_positives:
        print(f'  {score:.3f}  {name}')

    print(f'\nTop {top_k} false negatives (fake missed as real):')
    for name, score, _ in false_negatives:
        print(f'  {score:.3f}  {name}')

    return false_positives, false_negatives


model_eval = load_model(str(CHECKPOINT_DIR / 'best_model.pt'))

print('=== In-distribution robustness ===')
build_robustness_table_streaming(model_eval, eval_samples_raw,
                                  str(OUTPUT_DIR / 'robustness_indist.csv'))

print('\n=== Error analysis (clean) ===')
error_analysis_pil(model_eval, eval_samples_raw, transform_name='clean', top_k=10)

# WildFake uses a local CSV of image paths — build this yourself from
# COCO val2017 (real) + DALL-E Advanced (fake), per the hackathon's
# validation-dataset instructions.
# def load_labels_csv(csv_path: str) -> list:
#     with open(csv_path) as f:
#         return [(row["image_path"], int(row["label"])) for row in csv.DictReader(f)]
#
# wildfake_samples_raw = [(Image.open(p).convert('RGB'), l)
#                          for p, l in load_labels_csv('wildfake_labels.csv')]
# print('=== WildFake (out-of-distribution) robustness ===')
# build_robustness_table_streaming(model_eval, wildfake_samples_raw,
#                                   str(OUTPUT_DIR / 'robustness_wildfake.csv'))

import pandas as pd

df = pd.read_csv(str(OUTPUT_DIR / "robustness_indist.csv"))
print(df.sort_values("accuracy"))  # worst transforms first