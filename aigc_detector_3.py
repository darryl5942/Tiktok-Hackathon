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
| Train | CIFAKE + AIGC_DETECTION + AIGC_DETECTION_TRANSFORMED + SID_Set (toggle via USE_* flags below) | Core training data |
| Validation | Carved out of each enabled source's TRAIN pool (VAL_SPLIT_RATIO) | Checkpoint selection, early stopping — never the official test/validation split |
| Test (in-distribution) | Each source's OFFICIAL test/validation split, touched only in Section 7 (AIGC_DETECTION_TRANSFORMED has none — train/val only) | Final honest accuracy — same distribution as training, never touched during training |
| External benchmark (out-of-distribution) | WildFake subset (COCO val2017 real + DALL·E Advanced fake) | Generalization check + required robustness table. **Never trained on.** |

### Why three eval numbers, not one
- **In-distribution test accuracy** — is the model actually good, or did it overfit?
- **Robustness-under-transform accuracy** (on the same test set) — does it survive compression/blur/crop?
- **External (WildFake) accuracy** — does it generalize to a completely different generator/photo source?

### Model
Two-branch design (see Section 4 for full detail), well under the 2B parameter cap:
- **RGB branch** — frozen CLIP ViT-H/14 (~632M params, never updated during training).
- **Frequency branch** — trainable ConvNeXt-Base on the log-magnitude FFT spectrum.
- **Fusion** — a small cross-modal attention block combining both branches' features.

### Robustness strategy
Augmentation-in-the-loop: each training sample either stays clean (probability P_CLEAN) or gets
1 to MAX_STACKED_TRANSFORMS DISTINCT transforms from the hackathon's spec table applied in
sequence (e.g. blur THEN JPEG compression) — simulating an image that's been through multiple
rounds of real-world processing, not just one.
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
from techjam_utils import (
    benchmark_rows_to_samples,
    env_flag,
    find_best_threshold,
    load_labeled_csv_rows,
    portable_identifier,
)

load_dotenv()  # reads .env in the current working directory into environment variables

# ── Toggle which data sources to use — flip these off to quickly iterate ──
# without waiting on a source you don't currently need (e.g. testing the
# pipeline on CIFAKE alone while SID_Set's network streaming is being flaky).
USE_CIFAKE = True
USE_SID_SET = False
USE_AIGC_DETECTION = True  # a third Kaggle source: 5k real (COCO train2017) +
                            # 6k fake (2k each ADM / SD1.5 / Midjourney) — adds
                            # Midjourney specifically, which neither CIFAKE nor
                            # SID_Set represents. Uses the SAME Kaggle account
                            # as CIFAKE, so no extra credentials are needed.
USE_AIGC_DETECTION_TRANSFORMED = True  # also load the dataset's own
                            # "transformed_data" folder (pre-transformed
                            # images) as a SEPARATE training source, with
                            # our own transform stack turned OFF for it —
                            # requires USE_AIGC_DETECTION = True, since it's
                            # a subfolder of the same download.
SKIP_TRAINING = env_flag("AIGC_SKIP_TRAINING", True)
# Set AIGC_SKIP_TRAINING=0 to train/resume instead of going straight to
# inference/eval. The old hard-coded default is preserved if the variable is
# not set, but the active mode is now explicit and overrideable.

# Fraction of each source's TRAIN pool held back as the early-stopping
# validation set. The source's OFFICIAL test/validation split is reserved
# entirely for Section 7 — it is never touched during training or checkpoint
# selection, so it's a genuine "touched once" test set, matching the plan
# above (previously, the official test/validation split was used for BOTH
# per-epoch checkpoint selection AND the final robustness report, which
# meant the reported numbers were measured on data the checkpoint-selection
# process was implicitly tuned against).
VAL_SPLIT_RATIO = 0.15

# Best-guess subfolder names inside the AIGC detection Kaggle dataset —
# verify against the directory listing printed on first download (Section 0),
# and adjust these two strings if the real folder names differ. Nothing else
# needs to change if they do.
AIGC_DETECTION_REAL_SUBDIR = "real"
AIGC_DETECTION_FAKE_SUBDIR = "fake"
# Assumed to contain its own real/fake subfolders following the same
# convention as above — unverified, check the printed subfolder breakdown
# on first run.
AIGC_DETECTION_TRANSFORMED_SUBDIR = "transformed_data"
# Confirmed from an actual run: the dataset has train/{real,fake} AND
# test/{real,fake} — it DOES have an official test split, unlike what was
# assumed before. transformed_data/ contains only a "train" subfolder (no
# separate test), which its own train/val split already correctly assumes.
AIGC_DETECTION_TRAIN_SUBDIR = "train"
AIGC_DETECTION_TEST_SUBDIR = "test"

# =============================================================================
# SETTINGS — every commonly-tuned value lives here, in one place. Everything
# below this block reads these as already-defined constants; you shouldn't
# need to hunt through the rest of the file to change any of the following
# day-to-day. (The data-source toggles above stay separate since they're
# tightly coupled to the credential checks immediately following them.)
# =============================================================================

# ── Data volume ──────────────────────────────────────────────────────────
MAX_SAMPLES_PER_SOURCE = None  # e.g. 20_000 to cap; None = use all

# ── Model architecture (Section 4) ───────────────────────────────────────
IMG_SIZE = 224
RGB_BACKBONE_NAME = "vit_huge_patch14_clip_224.laion2b"  # frozen — ~632M params
FREQ_BACKBONE_NAME = "convnext_base"                     # trainable — ~88M params
FUSION_DIM = 512

# ── Training augmentation (Section 2b) ───────────────────────────────────
# Fraction of training samples left completely untouched. The remainder get
# 1 or more DISTINCT transforms stacked in sequence (e.g. blur THEN JPEG
# compression) rather than exactly one.
P_CLEAN = 1 / 15  # roughly matches the old "1-of-15-keys is clean" odds
MAX_STACKED_TRANSFORMS = 3

# ── Training loop (Section 5) ────────────────────────────────────────────
BATCH_SIZE = 64
# NUM_WORKERS must stay 0 — see the detailed comment at its old call site in
# Section 3 for the two independent reasons (Windows spawn + no worker-
# sharding in the dataset classes) if you're ever tempted to raise it.
NUM_WORKERS = 0
EPOCHS = 10000      # high ceiling — early stopping will cut this short in practice
PATIENCE = 5        # stop if val accuracy hasn't improved in this many epochs
LR = 1e-4

# ── Evaluation (Section 7) ────────────────────────────────────────────────
EVAL_CAP = 2000  # total images in the robustness/error-analysis eval pool

if not USE_CIFAKE and not USE_SID_SET and not USE_AIGC_DETECTION:
    raise RuntimeError("At least one of USE_CIFAKE, USE_SID_SET, or USE_AIGC_DETECTION must be True.")

if (USE_CIFAKE or USE_AIGC_DETECTION) and (not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY")):
    raise RuntimeError(
        "USE_CIFAKE or USE_AIGC_DETECTION is True but KAGGLE_USERNAME/KAGGLE_KEY are "
        "missing. Check that .env exists in the project root and contains both values "
        "— or set both of those flags to False if you don't want any Kaggle source this run."
    )
if USE_SID_SET and not os.environ.get("HF_TOKEN"):
    raise RuntimeError(
        "USE_SID_SET is True but HF_TOKEN is missing. Check that .env exists in "
        "the project root and contains a HuggingFace token — or set "
        "USE_SID_SET = False if you don't want SID_Set this run."
    )
if USE_AIGC_DETECTION_TRANSFORMED and not USE_AIGC_DETECTION:
    raise RuntimeError(
        "USE_AIGC_DETECTION_TRANSFORMED is True but USE_AIGC_DETECTION is False. "
        "transformed_data is a subfolder of the AIGC detection dataset download, "
        "so USE_AIGC_DETECTION must also be True."
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

# ── AIGC detection dataset (shxrlenee/aigc-detection-dataset): 5k real ────
# (COCO train2017) + 6k fake (2k each ADM / SD1.5 / Midjourney), from Kaggle.
# CONFIRMED structure (from an actual run): {root}/train/{real,fake} and
# {root}/test/{real,fake} — both with generator subfolders nested under
# .../fake/ — plus {root}/transformed_data/train/(presumed real,fake).
# The print below shows the actual layout two levels deep on every run, so
# any future dataset-version change to this structure is caught immediately.
if USE_AIGC_DETECTION:
    aigc_detection_root = Path(kagglehub.dataset_download(
        "shxrlenee/aigc-detection-dataset"
    ))
    print("AIGC detection dataset cached at:", aigc_detection_root)
    print("Contents (top level):", os.listdir(aigc_detection_root))
    for _sub in sorted(p for p in aigc_detection_root.iterdir() if p.is_dir()):
        print(f"  {_sub.name}/ contains:", os.listdir(_sub)[:10],
              "..." if len(os.listdir(_sub)) > 10 else "")
        # One level deeper too — otherwise transformed_data/train/'s own
        # contents (does it have real/fake beneath it?) would only ever
        # surface later in Section 3, which SKIP_TRAINING=True skips
        # entirely. This makes the structure fully visible every run,
        # regardless of that flag.
        for _subsub in sorted(p for p in _sub.iterdir() if p.is_dir()):
            print(f"    {_sub.name}/{_subsub.name}/ contains:", os.listdir(_subsub)[:10],
                  "..." if len(os.listdir(_subsub)) > 10 else "")
else:
    aigc_detection_root = None
    print("USE_AIGC_DETECTION is False — skipping AIGC detection dataset download.")

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


_PRINTED_SUBFOLDER_BREAKDOWN = set()  # avoids printing the same directory's
                                       # breakdown twice (train-role and val-
                                       # role instances both scan it once each)


def _scan_images_excluding_test(root_dir: Path) -> list:
    """
    Recursively finds image files under root_dir — INCLUDING any nested
    subfolders (e.g. a generator-origin breakdown like fake/adm/, fake/sd15/,
    fake/midjourney/), since rglob is recursive by design.

    EXCLUDES any file sitting inside a subfolder literally named "test"
    (case-insensitive), at any depth — a defensive guard in case a dataset
    nests a held-out test split inside its real/fake folders (e.g.
    fake/test/) rather than as a separate sibling directory. Without this,
    such a split would be silently swept into training with no warning,
    since a plain recursive glob doesn't distinguish "generator subfolder"
    from "held-out test subfolder."

    Prints a one-time per-subfolder image count breakdown for each unique
    root_dir, so what's actually being loaded is visible rather than assumed.
    """
    root_dir = Path(root_dir)
    included = []
    excluded_count = 0
    per_subfolder = {}

    for p in root_dir.rglob('*'):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel_parts = p.relative_to(root_dir).parts[:-1]  # directory components only
        if any(part.lower() == 'test' for part in rel_parts):
            excluded_count += 1
            continue
        included.append(p)
        subfolder = rel_parts[0] if rel_parts else '(directly in ' + root_dir.name + ')'
        per_subfolder[subfolder] = per_subfolder.get(subfolder, 0) + 1

    if str(root_dir) not in _PRINTED_SUBFOLDER_BREAKDOWN:
        _PRINTED_SUBFOLDER_BREAKDOWN.add(str(root_dir))
        print(f"[{root_dir}] image breakdown by subfolder:")
        for subfolder, count in sorted(per_subfolder.items()):
            print(f"    {subfolder}: {count:,} images")
        if excluded_count:
            print(f"    (excluded {excluded_count:,} image(s) found inside a subfolder "
                  f"named 'test' — not used in training)")

    return sorted(included)


class KaggleDirStreamDataset(IterableDataset):
    """
    Streams images lazily from a directory pair (real_dir / fake_dir).
    Images are opened one at a time — no bulk loading into RAM.
    Compatible with CIFAKE's layout: train/REAL, train/FAKE, test/REAL, test/FAKE.
    Also handles sources with generator-origin subfolders nested under
    real_dir/fake_dir (e.g. fake/adm/, fake/sd15/, fake/midjourney/) —
    scanned recursively, with any nested "test"-named subfolder excluded.

    NOTE: this does NOT shard samples across DataLoader workers. If you ever
    raise NUM_WORKERS above 0, every worker will iterate the FULL sample list
    independently, silently duplicating every sample once per worker. Add
    torch.utils.data.get_worker_info()-based sharding before doing that.
    """
    def __init__(self, real_dir, fake_dir, train=True, aug_in_loop=True, max_samples=None,
                 manifest: dict = None, source_key: str = "default", role: str = "train",
                 val_ratio: float = 0.15):
        self.real_paths = _scan_images_excluding_test(real_dir)
        self.fake_paths = _scan_images_excluding_test(fake_dir)

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
                img = apply_random_transform_stack(img)
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
                img = apply_random_transform_stack(img)
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

if DEVICE == "cuda":
    print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
else:
    _has_cuda_build = "+cu" in torch.__version__
    print("Using device: cpu")
    print("WARNING: CUDA is not available. Training/inference will be dramatically "
          "slower on CPU for this model. Most likely causes:")
    if not _has_cuda_build:
        print(f"  -> Your installed torch ({torch.__version__}) is a CPU-ONLY build — its "
              f"version string has no '+cuXXX' suffix (a GPU build looks like "
              f"'2.13.0+cu130'). This happens automatically if torch/torchvision are ever "
              f"installed via a plain 'pip install torch' or 'pip install -r "
              f"requirements.txt' — PyPI's default wheel has no CUDA support at all.")
        print("     Fix:")
        print("       pip uninstall torch torchvision -y")
        print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
        print("     (check https://pytorch.org/get-started/locally/ for the exact CUDA tag "
              "your driver supports if cu130 doesn't install)")
    else:
        print(f"  -> torch ({torch.__version__}) IS a CUDA build, but no GPU/driver was "
              f"detected. Run 'nvidia-smi' in a terminal to check your driver is installed "
              f"and recognizes your GPU.")

"""## 2b. Dataset & transform pool

Implements the exact transform table from the problem statement. `TRANSFORM_POOL` is reused in
three places: training-time augmentation, the robustness eval, and the (optional) named-transform
demo in your video.
"""

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

# P_CLEAN and MAX_STACKED_TRANSFORMS are set in the SETTINGS block at the top
# of the file. The remainder of samples (not left clean) get 1 or more
# DISTINCT transforms stacked in sequence (e.g. blur THEN JPEG compression)
# rather than exactly one — real-world images are often degraded by multiple
# processes in a row (resized for a thumbnail, then re-compressed on
# re-upload), and prior research on cross-generator generalization (Wang et
# al., CVPR 2020) specifically credits this kind of combined augmentation
# over single-transform augmentation.

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

CIFAKE, AIGC_DETECTION, AIGC_DETECTION_TRANSFORMED -> KaggleDirStreamDataset
    (reads from the project-local kagglehub cache)
SID_Set -> HFStreamDataset (HuggingFace streaming, no disk write)

Neither source requires the full dataset to be in RAM or copied elsewhere.

IMPORTANT: the early-stopping validation set used here is carved out of each
source's TRAIN pool (via VAL_SPLIT_RATIO), NOT from the official test/
validation split. The official test/validation split is reserved entirely
for Section 7 — it is never touched during training or checkpoint selection,
so it's a genuine "touched once" test set. Previously, the official
test/validation split served double duty as both the checkpoint-selection
signal and the final robustness report, which meant those numbers were
measured on data the checkpoint-selection process was implicitly tuned
against. (AIGC_DETECTION_TRANSFORMED is the one exception — it has no
official test split at all, so it only ever participates in train/val here.)
"""

from torch.utils.data import ChainDataset

# MAX_SAMPLES_PER_SOURCE, BATCH_SIZE, and NUM_WORKERS are set in the SETTINGS
# block at the top of the file.
#
# On BATCH_SIZE: 64 is a conservative starting point given ViT-H/14's
# (~632M) forward-pass activation cost even frozen/gradient-free — watch
# nvidia-smi on your first epoch and raise it if there's headroom on your
# 12GB card.
#
# On NUM_WORKERS: it must stay 0, for two independent reasons:
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

    if USE_AIGC_DETECTION:
        # CONFIRMED structure (from an actual run's printed listing):
        # {root}/train/{real,fake} for training data, {root}/test/{real,fake}
        # reserved for Section 7 (see below) — this dataset DOES have an
        # official test split, contrary to what was assumed before this fix.
        # KaggleDirStreamDataset scans recursively, so this correctly picks
        # up per-generator subfolders (e.g. fake/adm/, fake/sd15/,
        # fake/midjourney/) nested under train/fake/.
        AIGC_DETECTION_REAL_DIR = aigc_detection_root / AIGC_DETECTION_TRAIN_SUBDIR / AIGC_DETECTION_REAL_SUBDIR
        AIGC_DETECTION_FAKE_DIR = aigc_detection_root / AIGC_DETECTION_TRAIN_SUBDIR / AIGC_DETECTION_FAKE_SUBDIR

        aigc_detection_train_ds = KaggleDirStreamDataset(
            real_dir=AIGC_DETECTION_REAL_DIR,
            fake_dir=AIGC_DETECTION_FAKE_DIR,
            train=True, aug_in_loop=True,
            max_samples=MAX_SAMPLES_PER_SOURCE,
            manifest=_split_manifest, source_key="AIGC_DETECTION", role="train",
            val_ratio=VAL_SPLIT_RATIO,
        )
        aigc_detection_earlystop_val_ds = KaggleDirStreamDataset(
            real_dir=AIGC_DETECTION_REAL_DIR,
            fake_dir=AIGC_DETECTION_FAKE_DIR,
            train=False, aug_in_loop=False,
            max_samples=MAX_SAMPLES_PER_SOURCE,
            manifest=_split_manifest, source_key="AIGC_DETECTION", role="val",
            val_ratio=VAL_SPLIT_RATIO,
        )
        print(f'AIGC detection train:       {len(aigc_detection_train_ds.samples):,} images')
        print(f'AIGC detection early-stop val: {len(aigc_detection_earlystop_val_ds.samples):,} images')
        train_sources.append(aigc_detection_train_ds)
        val_sources.append(aigc_detection_earlystop_val_ds)
        # This dataset DOES have an official test/ split (confirmed) — it's
        # reserved for Section 7's evaluation pool, never touched here,
        # matching CIFAKE's own train/val/test separation.
    else:
        aigc_detection_train_ds = None
        aigc_detection_earlystop_val_ds = None

    if USE_AIGC_DETECTION_TRANSFORMED:
        # CONFIRMED: transformed_data/ contains exactly one subfolder, "train"
        # (no separate test split for this source). real/fake beneath that
        # is assumed to follow the same convention as the base dataset —
        # check the printed subfolder breakdown below on first run.
        #
        # aug_in_loop=False on BOTH instances is the whole point of this
        # source: these images are already transformed, so our own random
        # transform stack must NOT be applied on top of them. train=True
        # still shuffles sample order — it just skips the transform-
        # selection step (train AND aug_in_loop both need to be True for a
        # transform to be applied — see KaggleDirStreamDataset's __iter__).
        AIGC_DETECTION_TRANSFORMED_REAL_DIR = (
            aigc_detection_root / AIGC_DETECTION_TRANSFORMED_SUBDIR / AIGC_DETECTION_TRAIN_SUBDIR
            / AIGC_DETECTION_REAL_SUBDIR
        )
        AIGC_DETECTION_TRANSFORMED_FAKE_DIR = (
            aigc_detection_root / AIGC_DETECTION_TRANSFORMED_SUBDIR / AIGC_DETECTION_TRAIN_SUBDIR
            / AIGC_DETECTION_FAKE_SUBDIR
        )

        aigc_detection_transformed_train_ds = KaggleDirStreamDataset(
            real_dir=AIGC_DETECTION_TRANSFORMED_REAL_DIR,
            fake_dir=AIGC_DETECTION_TRANSFORMED_FAKE_DIR,
            train=True, aug_in_loop=False,
            max_samples=MAX_SAMPLES_PER_SOURCE,
            manifest=_split_manifest, source_key="AIGC_DETECTION_TRANSFORMED", role="train",
            val_ratio=VAL_SPLIT_RATIO,
        )
        aigc_detection_transformed_earlystop_val_ds = KaggleDirStreamDataset(
            real_dir=AIGC_DETECTION_TRANSFORMED_REAL_DIR,
            fake_dir=AIGC_DETECTION_TRANSFORMED_FAKE_DIR,
            train=False, aug_in_loop=False,
            max_samples=MAX_SAMPLES_PER_SOURCE,
            manifest=_split_manifest, source_key="AIGC_DETECTION_TRANSFORMED", role="val",
            val_ratio=VAL_SPLIT_RATIO,
        )
        print(f'AIGC detection (pre-transformed) train:       '
              f'{len(aigc_detection_transformed_train_ds.samples):,} images')
        print(f'AIGC detection (pre-transformed) early-stop val: '
              f'{len(aigc_detection_transformed_earlystop_val_ds.samples):,} images')
        train_sources.append(aigc_detection_transformed_train_ds)
        val_sources.append(aigc_detection_transformed_earlystop_val_ds)
    else:
        aigc_detection_transformed_train_ds = None
        aigc_detection_transformed_earlystop_val_ds = None

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
        raise RuntimeError("No data sources ended up enabled — check USE_CIFAKE/USE_SID_SET/USE_AIGC_DETECTION above.")

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
    # Each source contributes a (train) row and an (early-stop val) row that
    # BOTH correctly report the same total_available — val is carved out of
    # the train pool via the manifest, not read from a separate directory.
    # Summing total_available across every row would therefore double-count
    # every source's pool once. Track which base source names have already
    # had their total counted, so each one only contributes once.
    _counted_totals_for = set()

    for src in sources:
        used_str = f"{src['used']:,}" + ("*" if src['used_is_upper_bound'] else "")
        total_str = f"{src['total_available']:,}" if src['total_available'] is not None else "unknown"
        print(f"{src['name']:<22} {used_str:>12} {total_str:>14}")

        combined_used += src['used']
        base_name = src['name'].rsplit(' (', 1)[0]  # "CIFAKE (train)" -> "CIFAKE"
        if src['total_available'] is None:
            combined_total_known = False
            unknown_sources.append(src['name'])
        elif base_name not in _counted_totals_for:
            _counted_totals_for.add(base_name)
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

    if USE_AIGC_DETECTION:
        _aigc_det_train_total = (len(aigc_detection_train_ds.real_paths) +
                                  len(aigc_detection_train_ds.fake_paths))
        _aigc_det_val_total = (len(aigc_detection_earlystop_val_ds.real_paths) +
                                len(aigc_detection_earlystop_val_ds.fake_paths))
        _summary_sources.append({"name": "AIGC_DETECTION (train)", "total_available": _aigc_det_train_total,
                                  "used": len(aigc_detection_train_ds.samples), "used_is_upper_bound": False})
        _summary_sources.append({"name": "AIGC_DETECTION (early-stop val)", "total_available": _aigc_det_val_total,
                                  "used": len(aigc_detection_earlystop_val_ds.samples), "used_is_upper_bound": False})

    if USE_AIGC_DETECTION_TRANSFORMED:
        _aigc_det_t_train_total = (len(aigc_detection_transformed_train_ds.real_paths) +
                                    len(aigc_detection_transformed_train_ds.fake_paths))
        _aigc_det_t_val_total = (len(aigc_detection_transformed_earlystop_val_ds.real_paths) +
                                  len(aigc_detection_transformed_earlystop_val_ds.fake_paths))
        _summary_sources.append({"name": "AIGC_DETECTION_TRANSFORMED (train)",
                                  "total_available": _aigc_det_t_train_total,
                                  "used": len(aigc_detection_transformed_train_ds.samples),
                                  "used_is_upper_bound": False})
        _summary_sources.append({"name": "AIGC_DETECTION_TRANSFORMED (early-stop val)",
                                  "total_available": _aigc_det_t_val_total,
                                  "used": len(aigc_detection_transformed_earlystop_val_ds.samples),
                                  "used_is_upper_bound": False})

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

- **RGB branch**: frozen CLIP ViT-H/14 (~632M params, `vit_huge_patch14_clip_224.laion2b`
  — the plain OpenCLIP/LAION-2B image encoder, NOT an ImageNet-fine-tuned variant,
  which would use different normalization and defeat the point of a general-purpose
  frozen feature extractor). Per Ojha et al., CVPR 2023 ("Towards Universal Fake
  Image Detectors that Generalize Across Generative Models"), a frozen large
  pretrained backbone + linear probe generalizes to UNSEEN generators far better
  than fine-tuning a smaller CNN end-to-end — directly relevant since WildFake is
  your out-of-distribution benchmark. Frozen also means zero gradient/optimizer-
  state memory cost for this branch, despite its size.
- **Frequency branch**: trainable ConvNeXt-Base (~88M params) on the FFT
  spectrum — this one DOES need to adapt to the FFT-magnitude input domain,
  so it stays trainable, unlike the RGB branch.
- **Fusion**: a small cross-modal attention block instead of naive
  concatenation, so the model can learn interactions between pixel content
  and frequency artifacts rather than treating them as independent evidence.

Total ~721M params — still comfortably under the hackathon's 2B cap.
Trainable ~91M of those (frequency branch + fusion + head).
"""


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

Key design points:
  - BATCH_SIZE=64: sized for the current ~721M-param model (frozen ViT-H/14
    + trainable ConvNeXt-Base) — watch nvidia-smi on your first epoch and
    adjust if needed; this is not the tiny-model baseline anymore.
  - Cosine LR schedule instead of a flat learning rate.
  - Automatic mixed precision (torch.cuda.amp) for speed/memory.
  - Early stopping: EPOCHS is a high ceiling, PATIENCE actually decides when
    training stops.
  - resume_from: continue training from a saved checkpoint instead of always
    starting over from pretrained ImageNet weights.
  - Checkpoints save to the project-local CHECKPOINT_DIR, not a Colab path.
"""

import shutil
import datetime
import time

# EPOCHS, PATIENCE, and LR are set in the SETTINGS block at the top of the file.


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
    return {'USE_CIFAKE': USE_CIFAKE, 'USE_SID_SET': USE_SID_SET,
            'USE_AIGC_DETECTION': USE_AIGC_DETECTION,
            'USE_AIGC_DETECTION_TRANSFORMED': USE_AIGC_DETECTION_TRANSFORMED}


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
    if USE_AIGC_DETECTION:
        # Same reasoning as CIFAKE above — manifest-based, so track actual
        # outcomes rather than the raw ratio.
        snapshot['aigc_detection_total_available'] = (
            len(aigc_detection_train_ds.real_paths) + len(aigc_detection_train_ds.fake_paths)
        )
        snapshot['aigc_detection_train_used'] = len(aigc_detection_train_ds.samples)
        snapshot['aigc_detection_val_used'] = len(aigc_detection_earlystop_val_ds.samples)
    if USE_AIGC_DETECTION_TRANSFORMED:
        snapshot['aigc_detection_transformed_total_available'] = (
            len(aigc_detection_transformed_train_ds.real_paths) +
            len(aigc_detection_transformed_train_ds.fake_paths)
        )
        snapshot['aigc_detection_transformed_train_used'] = len(aigc_detection_transformed_train_ds.samples)
        snapshot['aigc_detection_transformed_val_used'] = len(aigc_detection_transformed_earlystop_val_ds.samples)
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
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load {resume_from} into the current model architecture. "
                f"This almost always means the model definition (backbone names, "
                f"fusion_dim, etc.) has changed since this checkpoint was saved — e.g. "
                f"swapping to a different backbone size. A checkpoint's weights are tied "
                f"to the exact architecture that produced them and can't be loaded into "
                f"a differently-shaped model. Delete or rename this checkpoint to start "
                f"fresh with the new architecture, or revert the architecture change to "
                f"keep resuming from it.\n\nOriginal error: {e}"
            ) from e
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # load_state_dict restores EVERY param_group hyperparameter from the
        # checkpoint, including 'lr' — silently discarding whatever LR was
        # just set when the optimizer above was constructed. The scheduler
        # is built AFTER this point and reads its starting point from
        # whatever the optimizer says right now, so an unfixed LR here means
        # editing the LR constant in the script has NO EFFECT on a resumed
        # run — the checkpoint's old LR wins every time. Same failure shape
        # as the T_max bug fixed earlier, just on the optimizer's LR instead
        # of the scheduler's ceiling. Re-apply the current LR explicitly so
        # a deliberate change actually takes effect.
        old_lr = optimizer.param_groups[0]['lr']
        if old_lr != LR:
            print(f"Adjusting LR: checkpoint was saved at {old_lr:.2e}, applying the "
                  f"currently-configured LR of {LR:.2e} instead.")
            for group in optimizer.param_groups:
                group['lr'] = LR
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

    if DEVICE == "cuda":
        # Tracks the TRUE peak VRAM usage since this point, regardless of
        # brief spikes during backward/optimizer steps that Task
        # Manager/nvidia-smi's periodic sampling could easily miss —
        # this is PyTorch reporting its own actual peak, not a live snapshot.
        torch.cuda.reset_peak_memory_stats()

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

        if DEVICE == "cuda":
            peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            mem_str = (f'Peak VRAM={peak_allocated_gb:.2f}GB allocated / '
                       f'{peak_reserved_gb:.2f}GB reserved (of {total_gb:.1f}GB total), ')
        else:
            mem_str = ''

        print(f'Epoch {epoch+1}: Loss={total_loss/max(n_batches,1):.4f}, '
              f'Val Acc={val_acc:.4f}, LR={scheduler.get_last_lr()[0]:.2e}, '
              f'{mem_str}'
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
        preds = (torch.sigmoid(model(rgb, freq)) > TRAIN_EVAL_THRESHOLD).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if total > 0:
            pbar.set_postfix(acc=f'{correct/total:.4f}')
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def collect_scores(model, loader):
    """Collects labels and confidence scores for threshold calibration."""
    model.eval()
    labels, scores = [], []
    pbar = tqdm(loader, desc='Calibrating', leave=False)
    for batch in pbar:
        rgb = batch['rgb'].to(DEVICE)
        freq = batch['freq'].to(DEVICE)
        batch_labels = batch['label'].detach().cpu().tolist()
        batch_scores = torch.sigmoid(model(rgb, freq)).detach().cpu().tolist()
        labels.extend(int(v) for v in batch_labels)
        scores.extend(float(v) for v in batch_scores)
    return labels, scores


def save_decision_threshold(threshold: float, path: Path = OUTPUT_DIR / "decision_threshold.json") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"decision_threshold": threshold}, f, indent=2)
    return path


def load_decision_threshold(path: Path = OUTPUT_DIR / "decision_threshold.json", default: float = 0.5) -> float:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            payload = json.load(f)
        return float(payload.get("decision_threshold", default))
    except Exception as exc:
        print(f"[warn] could not load calibrated threshold from {path}: {exc}")
        return default


def calibrate_decision_threshold(model, loader) -> float:
    labels, scores = collect_scores(model, loader)
    threshold, acc = find_best_threshold(labels, scores)
    print(f"Calibrated decision threshold = {threshold:.4f} (val acc {acc:.4f})")
    save_decision_threshold(threshold)
    return threshold


_default_checkpoint = str(CHECKPOINT_DIR / "best_model.pt")
_latest_checkpoint = str(CHECKPOINT_DIR / "latest_checkpoint.pt")

TRAIN_EVAL_THRESHOLD = 0.5
DECISION_THRESHOLD = load_decision_threshold()

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
    DECISION_THRESHOLD = calibrate_decision_threshold(model, val_loader)
    print(f"Saved calibrated decision threshold to {OUTPUT_DIR / 'decision_threshold.json'}")

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


def summarize_inference_results(results: list, top_n: int = 5,
                                 out_csv=str(OUTPUT_DIR / "inference_summary.csv"),
                                 out_chart=str(OUTPUT_DIR / "inference_summary_chart.png")):
    """
    Summarizes run_inference()'s results: counts, confidence-score
    distribution stats, a two-panel chart, and the top-N most confident
    detections in each direction for quick manual spot-checking.

    NOTE: there's no ground truth for inference_images/ — these are real
    images being checked, not labeled test data — so this can only report
    DISTRIBUTION and COUNT statistics, never accuracy/precision/recall.
    Accuracy requires known labels; see Section 7 for that (on CIFAKE/
    AIGC_DETECTION's test folders, which do have labels).
    """
    if not results:
        print("No inference results to summarize (empty result set).")
        return None

    preds = [r["pred"] for r in results]
    n = len(preds)
    n_fake = sum(1 for p in preds if p > 0.5)
    n_real = n - n_fake
    mean_pred = sum(preds) / n
    sorted_preds = sorted(preds)
    median_pred = (sorted_preds[n // 2] if n % 2 == 1
                   else (sorted_preds[n // 2 - 1] + sorted_preds[n // 2]) / 2)
    std_pred = (sum((p - mean_pred) ** 2 for p in preds) / n) ** 0.5

    summary = {
        "total_images": n,
        "predicted_ai_generated": n_fake,
        "predicted_real": n_real,
        "pct_ai_generated": round(100 * n_fake / n, 2),
        "pct_real": round(100 * n_real / n, 2),
        "mean_confidence": round(mean_pred, 4),
        "median_confidence": round(median_pred, 4),
        "std_confidence": round(std_pred, 4),
        "min_confidence": round(min(preds), 4),
        "max_confidence": round(max(preds), 4),
    }

    print(f"\n=== Inference summary ({n} images) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"Wrote inference summary -> {out_csv}")

    # Top-N most confident detections in each direction — names the actual
    # images, not just aggregate counts, for quick manual spot-checking.
    sorted_results = sorted(results, key=lambda r: r["pred"], reverse=True)
    n_show = min(top_n, n)
    print(f"\nTop {n_show} most confident AI-GENERATED detections:")
    for r in sorted_results[:n_show]:
        print(f"  {r['pred']:.4f}  {r['image_path']}")
    print(f"\nTop {n_show} most confident REAL detections:")
    for r in sorted_results[-n_show:][::-1]:
        print(f"  {r['pred']:.4f}  {r['image_path']}")

    # Chart: detection counts + confidence-score distribution
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(["Real", "AI-Generated"], [n_real, n_fake], color=["#55A868", "#C44E52"])
    axes[0].set_ylabel("Number of images")
    axes[0].set_title(f"Detection Summary ({n} images)")
    for i, v in enumerate([n_real, n_fake]):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    axes[1].hist(preds, bins=20, range=(0, 1), color="#4C72B0", edgecolor="white")
    axes[1].axvline(0.5, color="red", linestyle="--", linewidth=1, label="Decision threshold (0.5)")
    axes[1].set_xlabel("Predicted confidence (AI-generated likelihood)")
    axes[1].set_ylabel("Number of images")
    axes[1].set_title("Confidence Score Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_chart, dpi=150)
    plt.close(fig)
    print(f"\nWrote inference chart -> {out_chart}")

    return summary


# PRIORITY OUTPUT: automatically run inference on whatever's in
# INFERENCE_INPUT_DIR. This is the primary, user-facing output of the whole
# pipeline — your own images take priority over Section 7's robustness
# analysis below, which uses the CIFAKE/AIGC_DETECTION *test* folders (not
# your images) purely as a supporting robustness signal, not as the main
# analysis output.
#
# If INFERENCE_INPUT_DIR is empty, no analysis is performed on it at all —
# no preds.json gets written, no error is raised, just a clear message.
_inference_images_found = [
    p for p in Path(INFERENCE_INPUT_DIR).rglob("*") if p.suffix.lower() in VALID_EXT
]
if _inference_images_found:
    print(f"\nFound {len(_inference_images_found)} image(s) in {INFERENCE_INPUT_DIR} "
          f"— running inference (this is the priority output)...")
    _inference_results = run_inference(str(INFERENCE_INPUT_DIR))
    summarize_inference_results(_inference_results)
else:
    print(f"\n{INFERENCE_INPUT_DIR} is empty — skipping inference. "
          f"No preds.json will be written. Drop images in there to get predictions.")

"""## 7. Robustness evaluation (deliverable #4 + #5)

NOTE: everything below uses the CIFAKE/AIGC_DETECTION *test* folders, NOT
INFERENCE_INPUT_DIR — this is a supporting robustness/error-analysis signal
about the model's general behavior, separate from (and secondary to) the
inference_images/ predictions produced just above.

Because we're streaming, the robustness eval collects a **capped sample** into memory
(default 2,000 images) for repeatable per-transform scoring. This is fine for eval —
we never need the full dataset in RAM at once during training.

Produces the clean-vs-transformed table. Run **twice**:
1. On the held-out CIFAKE test + SID_Set validation stream (in-distribution)
2. On the external WildFake benchmark CSV (out-of-distribution, never trained on)
"""

from itertools import islice

# EVAL_CAP is set in the SETTINGS block at the top of the file.
_active_source_count = sum([USE_CIFAKE, USE_SID_SET, USE_AIGC_DETECTION])
per_source_cap = EVAL_CAP // _active_source_count

eval_samples_raw = []  # each entry: (PIL Image, label, identifier)

if USE_CIFAKE:
    # Split the cap evenly across classes BEFORE truncating — concatenating
    # real+fake into one list and slicing the front off is the exact same
    # bug already fixed in KaggleDirStreamDataset: CIFAKE's test split lists
    # all 10,000 real paths before any of the 10,000 fake ones, so any
    # per_source_cap <= 10,000 (the current default of 2000 included)
    # silently produced a ZERO-fake, real-only eval pool — meaning every
    # robustness/error-analysis number was being computed with no fake
    # images ever tested.
    _cifake_eval_per_class_cap = per_source_cap // 2
    cifake_real_eval = [(p, 0) for p in sorted((cifake_root / 'test' / 'REAL').rglob('*'))
                        if p.suffix.lower() in IMG_EXTS][:_cifake_eval_per_class_cap]
    cifake_fake_eval = [(p, 1) for p in sorted((cifake_root / 'test' / 'FAKE').rglob('*'))
                        if p.suffix.lower() in IMG_EXTS][:_cifake_eval_per_class_cap]
    cifake_eval_paths = cifake_real_eval + cifake_fake_eval

    for path, label in cifake_eval_paths:
        try:
            eval_samples_raw.append((Image.open(path).convert('RGB'), label, portable_identifier(path, cifake_root / 'test')))
        except Exception as e:
            print(f'[warn] {path}: {e}')

if USE_AIGC_DETECTION:
    # Confirmed structure: {root}/test/{real,fake} — genuinely untouched by
    # training (which only ever reads from {root}/train/...). Same balanced
    # per-class construction as CIFAKE above, for the same reason: avoid a
    # real-only eval pool from concatenating then slicing.
    _aigc_det_eval_per_class_cap = per_source_cap // 2
    aigc_det_test_real_dir = aigc_detection_root / AIGC_DETECTION_TEST_SUBDIR / AIGC_DETECTION_REAL_SUBDIR
    aigc_det_test_fake_dir = aigc_detection_root / AIGC_DETECTION_TEST_SUBDIR / AIGC_DETECTION_FAKE_SUBDIR
    aigc_det_real_eval = [(p, 0) for p in sorted(aigc_det_test_real_dir.rglob('*'))
                          if p.suffix.lower() in IMG_EXTS][:_aigc_det_eval_per_class_cap]
    aigc_det_fake_eval = [(p, 1) for p in sorted(aigc_det_test_fake_dir.rglob('*'))
                          if p.suffix.lower() in IMG_EXTS][:_aigc_det_eval_per_class_cap]
    aigc_det_eval_paths = aigc_det_real_eval + aigc_det_fake_eval

    for path, label in aigc_det_eval_paths:
        try:
            eval_samples_raw.append((Image.open(path).convert('RGB'), label, portable_identifier(path, aigc_detection_root / 'test')))
        except Exception as e:
            print(f'[warn] {path}: {e}')

if USE_SID_SET:
    # NOTE: unlike CIFAKE above, this takes the first `per_source_cap` items
    # in WHATEVER ORDER the HF stream provides them — if that underlying
    # order happens to group by class (as CIFAKE's directory listing does),
    # this would reproduce the exact same real-only eval pool bug just fixed
    # for CIFAKE. Not fixed here since USE_SID_SET is currently False and
    # the stream's actual ordering hasn't been verified — if you re-enable
    # this, check the class balance of eval_samples_raw's SID_Set portion
    # (or shuffle/stratify before capping) before trusting the results.
    sid_eval_stream = load_dataset('saberzl/SID_Set', split='validation', streaming=True)
    for i, item in enumerate(islice(sid_eval_stream, per_source_cap)):
        identifier = portable_identifier(item.get('image_path', f'sid_set_validation_row_{i}'))
        eval_samples_raw.append((item['image'].convert('RGB'), int(item['label']), identifier))

print(f'Eval pool: {len(eval_samples_raw)} images ready.')


@torch.no_grad()
def run_transform_eval_pil(model, samples_raw: list, transform_name: str):
    """samples_raw: list of (PIL Image, label, identifier)"""
    preds, labels, identifiers = [], [], []
    for img, label, identifier in samples_raw:
        img_t = apply_named_transform(img, transform_name)
        rgb = _normalize(img_t).unsqueeze(0).to(DEVICE)
        freq = to_freq_tensor(img_t).unsqueeze(0).to(DEVICE)
        score = torch.sigmoid(model(rgb, freq)).item()
        preds.append(score)
        labels.append(label)
        identifiers.append(identifier)

    pred_labels = [1 if p > DECISION_THRESHOLD else 0 for p in preds]
    acc = sum(int(p == l) for p, l in zip(pred_labels, labels)) / len(labels)
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else float('nan')
    return acc, auc, preds, labels, identifiers


def build_robustness_table_streaming(model, samples_raw,
                                      out_csv=str(OUTPUT_DIR / 'robustness_table.csv')):
    rows = []
    for name in TRANSFORM_POOL:
        acc, auc, _, _, _ = run_transform_eval_pil(model, samples_raw, name)
        rows.append({'transform': name, 'accuracy': round(acc, 4), 'auc': round(auc, 4)})
        print(f'{name:20s}  acc={acc:.4f}  auc={auc:.4f}')

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['transform', 'accuracy', 'auc'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote robustness table -> {out_csv}')
    return rows


# Groups TRANSFORM_POOL's 15 individual entries into families, for a compact
# clean-vs-transformed comparison — the deliverable asks for a "compact table
# or visual summary," and 15 individual rows isn't compact. Built from
# whatever rows build_robustness_table_streaming already computed, so this
# costs zero additional model evaluations.
TRANSFORM_FAMILIES = {
    'clean': ['clean'],
    'jpeg_compression': ['jpeg_90', 'jpeg_70', 'jpeg_50', 'jpeg_30'],
    'blur': ['blur_0.5', 'blur_1.0', 'blur_2.0'],
    'resize': ['resize_0.5', 'resize_0.25'],
    'noise': ['noise_0.02', 'noise_0.05', 'noise_0.10'],
    'color_jitter': ['color_jitter'],
    'center_crop': ['center_crop_80'],
}


def summarize_robustness_compact(rows: list, out_csv=str(OUTPUT_DIR / 'robustness_summary_compact.csv')):
    """
    Collapses the 15-row per-transform table into a 7-row family-level
    summary (clean vs. each transform family), with each family's accuracy
    drop relative to clean — the actual number a reader cares about most.
    """
    rows_by_name = {r['transform']: r for r in rows}
    clean_acc = rows_by_name['clean']['accuracy']

    summary = []
    for family, members in TRANSFORM_FAMILIES.items():
        member_rows = [rows_by_name[m] for m in members if m in rows_by_name]
        avg_acc = sum(r['accuracy'] for r in member_rows) / len(member_rows)
        avg_auc = sum(r['auc'] for r in member_rows) / len(member_rows)
        drop = clean_acc - avg_acc
        summary.append({
            'transform_family': family,
            'avg_accuracy': round(avg_acc, 4),
            'avg_auc': round(avg_auc, 4),
            'accuracy_drop_from_clean': round(drop, 4),
        })

    print(f"\n{'Transform family':<20} {'Avg Acc':>9} {'Avg AUC':>9} {'Drop from clean':>17}")
    print("-" * 58)
    for s in summary:
        print(f"{s['transform_family']:<20} {s['avg_accuracy']:>9.4f} {s['avg_auc']:>9.4f} "
              f"{s['accuracy_drop_from_clean']:>17.4f}")

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['transform_family', 'avg_accuracy',
                                                'avg_auc', 'accuracy_drop_from_clean'])
        writer.writeheader()
        writer.writerows(summary)
    print(f'\nWrote compact robustness summary -> {out_csv}')
    return summary


def plot_robustness_chart(rows: list, out_png=str(OUTPUT_DIR / 'robustness_chart.png')):
    """
    Bar chart of accuracy per individual transform, with a horizontal
    reference line at the clean-image accuracy — the "visual summary"
    option for the deliverable. Saved as a static PNG so it can be embedded
    directly in a README or Devpost writeup.
    """
    import matplotlib
    matplotlib.use('Agg')  # no display needed — just render straight to file
    import matplotlib.pyplot as plt

    names = [r['transform'] for r in rows]
    accs = [r['accuracy'] for r in rows]
    clean_acc = next(r['accuracy'] for r in rows if r['transform'] == 'clean')

    colors = ['#4C72B0' if n != 'clean' else '#55A868' for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(names, accs, color=colors)
    ax.axhline(clean_acc, color='red', linestyle='--', linewidth=1,
               label=f'Clean baseline ({clean_acc:.3f})')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    ax.set_title('Accuracy: Clean vs. Transformed Images')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'Wrote robustness chart -> {out_png}')
    return out_png


def write_overall_analysis(model, samples_raw, transform_name='clean',
                            out_csv=str(OUTPUT_DIR / 'overall_analysis.csv'),
                            out_json=str(OUTPUT_DIR / 'overall_analysis.json')):
    """
    Writes EVERY image's result (not just the top-k errors error_analysis_pil
    focuses on) to both CSV and JSON, in the same row order and field values
    in both — so the two files can be cross-checked against each other
    directly, and against error_analysis.csv's extreme-case subset.
    """
    acc, auc, preds, labels, identifiers = run_transform_eval_pil(model, samples_raw, transform_name)

    rows = []
    for identifier, score, true_label in zip(identifiers, preds, labels):
        predicted_label = 1 if score > DECISION_THRESHOLD else 0
        rows.append({
            'image_path': identifier,
            'pred': round(score, 4),
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': predicted_label == true_label,
        })

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'pred', 'true_label',
                                                'predicted_label', 'correct'])
        writer.writeheader()
        writer.writerows(rows)

    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=2)

    print(f'Wrote full analysis of all {len(rows)} images -> {out_csv} and {out_json}')
    print(f'Overall on "{transform_name}": accuracy={acc:.4f}, AUC={auc:.4f}')
    return rows


# One representative transform per family (not all 15) — keeps the example
# folder small and readable while still showing what each category of
# degradation actually looks like.
TRANSFORM_EXAMPLES_TO_SHOW = ['jpeg_30', 'blur_2.0', 'resize_0.25',
                              'noise_0.10', 'color_jitter', 'center_crop_80']
N_TRANSFORM_EXAMPLE_IMAGES = 3


def _save_comparison_grid(named_images: list, out_path, title: str = "", thumb_size: int = 224):
    """named_images: list of (label_text, PIL Image) tuples, laid out left to right."""
    from PIL import ImageDraw

    n = len(named_images)
    label_h = 24
    title_h = 30 if title else 0
    grid = Image.new("RGB", (thumb_size * n, thumb_size + label_h + title_h), "white")
    draw = ImageDraw.Draw(grid)
    if title:
        draw.text((8, 6), title, fill="black")
    for i, (label, img) in enumerate(named_images):
        thumb = img.resize((thumb_size, thumb_size))
        grid.paste(thumb, (i * thumb_size, title_h))
        draw.text((i * thumb_size + 8, title_h + thumb_size + 4), label, fill="black")
    grid.save(out_path)


def save_transform_examples(samples_raw, n_images: int = N_TRANSFORM_EXAMPLE_IMAGES,
                             transforms_to_show: list = None,
                             out_dir=str(OUTPUT_DIR / 'transform_examples')):
    """
    Saves the clean version and several named transforms of a few sample
    images as individual PNG files, plus one side-by-side comparison grid
    per image — so before/after can be inspected visually, not just measured
    numerically via the accuracy/AUC tables.
    """
    transforms_to_show = transforms_to_show or TRANSFORM_EXAMPLES_TO_SHOW
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # eval_samples_raw is ordered real-then-fake (per source), so a plain
    # samples_raw[:n_images] prefix slice would only ever pick real examples
    # — deliberately split the request across both classes instead.
    n_real = (n_images + 1) // 2
    n_fake = n_images // 2
    real_examples = [s for s in samples_raw if s[1] == 0][:n_real]
    fake_examples = [s for s in samples_raw if s[1] == 1][:n_fake]
    chosen = real_examples + fake_examples

    for i, (img, label, identifier) in enumerate(chosen):
        label_str = 'fake' if label == 1 else 'real'
        stem = f"example_{i}_{label_str}"

        img.save(out_path / f"{stem}_clean.png")
        variants = [("clean", img)]
        for t_name in transforms_to_show:
            transformed = apply_named_transform(img, t_name)
            transformed.save(out_path / f"{stem}_{t_name}.png")
            variants.append((t_name, transformed))

        _save_comparison_grid(
            variants, out_path / f"{stem}_comparison_grid.png",
            title=f"{stem}  (source: {identifier})  true label: {label_str.upper()}"
        )

    print(f"Wrote {len(chosen)} before/after transform example set(s) -> {out_path}")
    return out_path


def error_analysis_pil(model, samples_raw, transform_name='clean', top_k=10,
                        out_csv=str(OUTPUT_DIR / 'error_analysis.csv')):
    acc, auc, preds, labels, identifiers = run_transform_eval_pil(model, samples_raw, transform_name)
    records = list(zip(identifiers, preds, labels))

    false_positives = sorted(
        [r for r in records if r[2] == 0 and r[1] > 0.5], key=lambda r: -r[1]
    )[:top_k]
    false_negatives = sorted(
        [r for r in records if r[2] == 1 and r[1] < 0.5], key=lambda r: r[1]
    )[:top_k]

    print(f'Top {top_k} false positives (real flagged as fake):')
    for identifier, score, _ in false_positives:
        print(f'  {score:.3f}  {identifier}')

    print(f'\nTop {top_k} false negatives (fake missed as real):')
    for identifier, score, _ in false_negatives:
        print(f'  {score:.3f}  {identifier}')

    # Write a durable CSV — the console output above scrolls away, but this
    # is what you'd actually open the referenced images from when writing
    # the Error Analysis Note deliverable.
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['error_type', 'identifier', 'predicted_score', 'true_label'])
        writer.writeheader()
        for identifier, score, true_label in false_positives:
            writer.writerow({'error_type': 'false_positive', 'identifier': identifier,
                              'predicted_score': round(score, 4), 'true_label': true_label})
        for identifier, score, true_label in false_negatives:
            writer.writerow({'error_type': 'false_negative', 'identifier': identifier,
                              'predicted_score': round(score, 4), 'true_label': true_label})
    print(f'\nWrote error analysis (with real file paths) -> {out_csv}')
    print(f'Overall on "{transform_name}": accuracy={acc:.4f}, AUC={auc:.4f}')

    return false_positives, false_negatives


model_eval = load_model(str(CHECKPOINT_DIR / 'best_model.pt'))

print('=== In-distribution robustness ===')
_robustness_rows = build_robustness_table_streaming(model_eval, eval_samples_raw,
                                                     str(OUTPUT_DIR / 'robustness_indist.csv'))
summarize_robustness_compact(_robustness_rows)
plot_robustness_chart(_robustness_rows)

print('\n=== Error analysis (clean) ===')
error_analysis_pil(model_eval, eval_samples_raw, transform_name='clean', top_k=10)

print('\n=== Full analysis of all eval images ===')
write_overall_analysis(model_eval, eval_samples_raw, transform_name='clean')

print('\n=== Before/after transform examples ===')
save_transform_examples(eval_samples_raw)

def build_local_benchmark_samples(csv_path: str):
    """Converts a labeled CSV into the `(PIL Image, label, identifier)` format used by eval."""
    samples = []
    rows = load_labeled_csv_rows(csv_path)
    for image_path, label in benchmark_rows_to_samples(rows):
        p = Path(image_path)
        try:
            samples.append((Image.open(p).convert("RGB"), label, portable_identifier(p)))
        except Exception as e:
            print(f"[warn] {p}: {e}")
    return samples


WILDFAKE_LABELS_CSV = Path(os.environ.get("WILDFAKE_LABELS_CSV", PROJECT_ROOT / "wildfake_labels.csv"))
if WILDFAKE_LABELS_CSV.exists():
    print('\n=== WildFake (out-of-distribution) robustness ===')
    wildfake_samples_raw = build_local_benchmark_samples(str(WILDFAKE_LABELS_CSV))
    if wildfake_samples_raw:
        wildfake_rows = build_robustness_table_streaming(
            model_eval,
            wildfake_samples_raw,
            str(OUTPUT_DIR / 'robustness_wildfake.csv'),
        )
        summarize_robustness_compact(
            wildfake_rows,
            str(OUTPUT_DIR / 'robustness_wildfake_summary_compact.csv'),
        )
        plot_robustness_chart(
            wildfake_rows,
            str(OUTPUT_DIR / 'robustness_wildfake_chart.png'),
        )
    else:
        print(f"No valid WildFake samples could be loaded from {WILDFAKE_LABELS_CSV}.")
else:
    print(
        f"\nWildFake labels CSV not found at {WILDFAKE_LABELS_CSV} — skipping external "
        f"benchmark evaluation. Provide a labeled CSV with image_path,label columns "
        f"to enable it."
    )

import pandas as pd

df = pd.read_csv(str(OUTPUT_DIR / "robustness_indist.csv"))
print(df.sort_values("accuracy"))  # worst transforms first