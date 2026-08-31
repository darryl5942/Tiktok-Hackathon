# TikTok TechJam 2026 - AIGC Detector

Robust image-level AIGC detector for the TikTok TechJam problem statement: classify whether an image is AI-generated and stay reliable under common real-world transformations such as JPEG recompression, blur, resize, noise, color jitter, and center crop.

**Demo video**: [Robust Detection of AI-Generated Images Under Real-World Transformations — Demo](https://youtu.be/7vnF9jSAB24)

## What this project does

- Trains a two-branch detector that combines:
  - a frozen CLIP ViT-H/14 RGB encoder
  - a trainable ConvNeXt-Base frequency branch on FFT magnitude features
- Runs inference on an input image directory and writes `outputs/preds.json`
- Produces robustness summaries and error analysis for held-out evaluation data
- Optionally evaluates an external OOD benchmark when a labeled CSV is provided

## Repository Layout

- `aigc_detector_3.py` - orchestration: env/credential setup, dataset loading, training loop, inference, and evaluation
- `config.py` - shared model architecture and augmentation constants
- `image_transforms.py` - the robustness transform pool (JPEG/blur/resize/noise/color/crop) and augmentation logic
- `model.py` - the two-branch (CLIP + ConvNeXt frequency) detector architecture
- `data_pipeline.py` - dataset wrappers (Kaggle/HuggingFace streaming) and train/val split bookkeeping
- `techjam_cli.py` / `techjam_utils.py` - CLI entrypoint wrapper and small dependency-free shared helpers
- `requirements.txt` - Python dependencies
- `checkpoints/` - saved model checkpoints (gitignored — see Setup step 4 to download)
- `outputs/` - inference, robustness, and error-analysis artifacts
- `split_manifest.json` - persistent train/validation split assignments
- `tests/` - unit tests for `techjam_utils.py`

## Setup

> **Before you start — two things that will otherwise surprise you:**
> 1. **A free Kaggle account + API key is required, even just to run inference.** The script checks for `KAGGLE_USERNAME`/`KAGGLE_KEY` unconditionally on startup (used to download the training datasets its bundled robustness-evaluation pipeline reads from), and will refuse to run without them. Get a free key at [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token.
> 2. **The first run downloads ~12GB of Kaggle datasets** (CIFAKE + the AIGC detection dataset) before producing any output — this is a one-time cost (cached afterward), but budget several minutes for it depending on your connection, even if all you want is inference predictions on your own images.
>
> On Windows, if you hit a crash mid-download like `FileNotFoundError` during `_extract_archive` (path too long), your project folder's path is too deeply nested (common with OneDrive-synced folders). Set `AIGC_CACHE_DIR` to a short path before running, e.g.:
> ```powershell
> $env:AIGC_CACHE_DIR = "C:\aigc_cache"
> ```

1. Create and activate a Python 3.12 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the credentials required by the enabled data sources:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
HF_TOKEN=your_huggingface_token
```

4. Download the trained checkpoint. The `checkpoints/` directory is gitignored (the `.pt` file is ~3.4GB), so it is hosted separately on the Hugging Face Hub:

```bash
mkdir -p checkpoints
python -c "from huggingface_hub import hf_hub_download; import shutil; p = hf_hub_download(repo_id='darrylljl/tiktok-techjam-aigc-detector', filename='best_model.pt'); shutil.copy(p, 'checkpoints/best_model.pt')"
```

Model card: https://huggingface.co/darrylljl/tiktok-techjam-aigc-detector

## Run Modes

The script reads `AIGC_SKIP_TRAINING` from the environment:

- `AIGC_SKIP_TRAINING=1` - skip training and run inference/evaluation using `checkpoints/best_model.pt`
- `AIGC_SKIP_TRAINING=0` - train or resume, then save checkpoints and run the evaluation flow

Example:

```bash
AIGC_SKIP_TRAINING=1 python aigc_detector_3.py
```

There is also a small CLI wrapper for clearer entrypoints:

```bash
python techjam_cli.py train
python techjam_cli.py infer --input-dir inference_images
python techjam_cli.py eval --wildfake-labels-csv /path/to/wildfake_labels.csv
```

## Optional External Benchmark

If you have a labeled CSV for WildFake or another OOD benchmark, set:

```bash
WILDFAKE_LABELS_CSV=/path/to/wildfake_labels.csv
```

The CSV must contain:

- `image_path`
- `label` where `0=real` and `1=fake`

When present, the script writes:

- `outputs/robustness_wildfake.csv`
- `outputs/robustness_wildfake_summary_compact.csv`
- `outputs/robustness_wildfake_chart.png`

The CSV is validated before evaluation, so missing columns or invalid labels fail early with a useful error.

## Reproducing Results

### Fastest path: run inference with the trained checkpoint

1. Complete [Setup](#setup) steps 1-4 (this downloads `checkpoints/best_model.pt`).
2. Run inference over a directory of images:

```bash
AIGC_SKIP_TRAINING=1 python techjam_cli.py infer --input-dir inference_images
```

3. Output is written to `outputs/preds.json` as a JSON list, one entry per image:

```json
[
  {"image_path": "inference_images/example.png", "pred": 0.9460284113883972}
]
```

`pred` is the model's confidence (0-1) that the image is AI-generated.

### Full reproduction (training from scratch)

1. Ensure the expected datasets are available through Kaggle/Hugging Face credentials in `.env`.
2. Run the script with `AIGC_SKIP_TRAINING=0` to train or resume from checkpoints.
3. Review the outputs in `outputs/`:
   - `preds.json` for inference predictions
   - `robustness_indist.csv` and `robustness_summary_compact.csv` for held-out robustness
   - `error_analysis.csv` for false positives and false negatives
   - `robustness_chart.png` for the visual summary

## Current Observations

See the Robustness Evaluation Summary table below for the current held-out accuracy/AUC numbers by transform family (sourced directly from `outputs/robustness_summary_compact.csv`).

The detector calibrates its decision threshold from the validation set and saves it to `outputs/decision_threshold.json`, which makes the final classification rule less arbitrary than a fixed 0.5 cutoff.

The error analysis shows the model tends to struggle on heavily processed real images and some heavily degraded fake images, which is the main trade-off to address next.

## Robustness Evaluation Summary

Held-out validation accuracy/AUC by transform family, compared to clean images (source: `outputs/robustness_summary_compact.csv`, from a full end-to-end run of the current `best_model.pt` checkpoint):

| Transform | Accuracy | AUC | Accuracy drop vs. clean |
|---|---|---|---|
| Clean (no transform) | 0.9750 | 0.9970 | — |
| Color jitter | 0.9720 | 0.9967 | 0.0030 |
| JPEG compression | 0.9630 | 0.9943 | 0.0120 |
| Noise | 0.9530 | 0.9912 | 0.0220 |
| Center crop | 0.9550 | 0.9924 | 0.0200 |
| Blur | 0.9477 | 0.9891 | 0.0273 |
| Resize | 0.9235 | 0.9779 | 0.0515 |

Resize and blur are still the weakest transforms — both destroy the high-frequency artifacts the frequency branch relies on, so accuracy degrades the most under heavy downsampling or blurring — but the degradation is modest (worst case 0.9235 accuracy, still well above chance). See `outputs/robustness_chart.png` for the visual summary.

## Error Analysis Note

On the held-out evaluation set (clean transform, 1000-image eval pool):

- **False positives** (real images predicted as fake, confidence 0.78–0.97): e.g. `REAL/0026 (8).jpg` (0.973), `REAL/0007.jpg` (0.969), `REAL/0013 (10).jpg` (0.961) — these are real images the model is confidently, incorrectly calling AI-generated.
- **False negatives** (fake images predicted as real, confidence 0.003–0.35): e.g. `FAKE/137.jpg` (0.003), `FAKE/102 (9).jpg` (0.033), `FAKE/106 (3).jpg` (0.077) — fakes the model is confidently, incorrectly calling real. Full lists are in `outputs/error_analysis.csv`.
- **Trade-off**: the frequency branch is central to clean-image accuracy but is also the main failure point under transforms that suppress high-frequency content (blur, aggressive resize). A production system would need either a transform-invariant frequency representation or an ensemble with a spatial-artifact detector to close this gap.

## Team Member Contributions

_Fill in before submission — list each team member and their primary contribution area (e.g. model architecture, data pipeline, robustness evaluation, README/demo video)._
Chloe Cheo
- 
Darren Mah
- 
Darryl Lee
- Drafted the initial version of the AI detection pipeline
- Set up Hugging Face Hub hosting and uploaded the trained checkpoint (`best_model.pt`) so the model is reproducible without bloating git
- Added `.gitignore` for the checkpoint files
- Wrote the checkpoint-download setup steps, robustness evaluation table, and error analysis note in the README
- Drafted the Devpost written description
- Split `aigc_detector_3.py` into `config.py`, `image_transforms.py`, `model.py`, and `data_pipeline.py` for better code structure
- Fixed a bug where `techjam_cli.py infer --input-dir` was silently ignored (the script never read the `INFERENCE_INPUT_DIR` env var it set)
- Fixed a cross-platform bug in `portable_identifier()` that returned Windows backslash paths instead of portable forward-slash paths, caught by the existing test suite
- Made the dataset cache directory configurable via `AIGC_CACHE_DIR` to work around a Windows long-path crash during dataset extraction
- Verified the full pipeline end-to-end on GPU across two machines, and refreshed the README's robustness table and error analysis with real numbers from that run (replacing stale/inconsistent figures)
- Rotated leaked Kaggle/Hugging Face credentials found during setup and added `.cache/` to `.gitignore` to prevent future leaks
- Rewrote the Devpost description, correcting factual errors (backbone names, an unimplemented explainability claim) while strengthening the impact/motivation narrative
- Added and verified the public demo video link in the README and Devpost description
- Documented a specific limitation: training stacks 1-3 transforms per image, but the robustness evaluation only tests one transform at a time
Zacchaeus Tan
- Added `techjam_utils.py`, a shared helper module for env-flag parsing, portable path identifiers, labeled-CSV validation, benchmark-row conversion, and decision-threshold search
- Added `techjam_cli.py`, a CLI wrapper with explicit `train`/`infer`/`eval` modes
- Updated `aigc_detector_3.py` to calibrate and save a decision threshold after training and use it in evaluation/reporting, bias training augmentation more toward weak spots (blur, resize), normalize eval identifiers for machine-independent output, and validate the external benchmark CSV path
- Wrote the initial README covering setup, CLI usage, the external benchmark workflow, and limitations
- Added unit tests (`tests/test_techjam_utils.py`) for portable identifiers, threshold search, and CSV validation
- Verified the refactor via syntax compilation and the new unit tests


## Limitations

- The project is still a hackathon-scale prototype, not a production moderation system.
- Pure inference is currently coupled to the training-dataset download/credential check — running `techjam_cli.py infer` still requires Kaggle credentials and a ~12GB one-time dataset download, even though the required inference deliverable itself doesn't need that data. Given more time, inference-only mode would skip this entirely.
- External OOD evaluation depends on a local labeled CSV and is optional.
- The model is robust, but resize and blur remain the biggest weak spots.
- Training augmentation applies 1-3 stacked transforms per image (e.g. blur + JPEG + noise together, simulating an image degraded through multiple real-world processing steps), but the robustness evaluation only tests one transform at a time — it doesn't measure how the model holds up against the same kind of stacked/combined degradation it was actually trained on. Given more time, extending the evaluation to cover stacked-transform combinations (not just single transforms) would give a more complete, more realistic robustness picture.

## Notes

- `split_manifest.json` is committed so train/validation assignments stay stable across runs.
- Generated paths in CSV outputs are now normalized to avoid machine-specific absolute prefixes when possible.
