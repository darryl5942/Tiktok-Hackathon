# -*- coding: utf-8 -*-
"""Dataset wrappers and train/val split bookkeeping, extracted from aigc_detector_3.py.

Note: deliberately NOT named `datasets.py` — that would shadow the
HuggingFace `datasets` package (`from datasets import load_dataset`) used
elsewhere in this project, since the project root sits ahead of site-packages
on sys.path.

CIFAKE, AIGC_DETECTION, AIGC_DETECTION_TRANSFORMED -> KaggleDirStreamDataset
    (reads from the project-local kagglehub cache)
SID_Set -> HFStreamDataset (HuggingFace streaming, no disk write)

Neither source requires the full dataset to be in RAM or copied elsewhere.
"""

import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import IterableDataset

from image_transforms import apply_random_transform_stack, to_freq_tensor, _normalize

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

PROJECT_ROOT = Path(__file__).resolve().parent
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
            # without going through the manifest-based setup.
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
