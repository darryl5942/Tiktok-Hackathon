# TikTok TechJam 2026 - AIGC Detector

Robust image-level AIGC detector for the TikTok TechJam problem statement: classify whether an image is AI-generated and stay reliable under common real-world transformations such as JPEG recompression, blur, resize, noise, color jitter, and center crop.

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

Held-out validation accuracy/AUC by transform family, compared to clean images (source: `outputs/robustness_summary_compact.csv`):

| Transform | Accuracy | AUC | Accuracy drop vs. clean |
|---|---|---|---|
| Clean (no transform) | 0.959 | 0.9929 | — |
| Color jitter | 0.9605 | 0.9925 | -0.0015 (slightly better) |
| JPEG compression | 0.9549 | 0.9910 | 0.0041 |
| Noise | 0.9543 | 0.9905 | 0.0047 |
| Center crop | 0.9340 | 0.9854 | 0.0250 |
| Blur | 0.9162 | 0.9660 | 0.0428 |
| Resize | 0.8667 | 0.9325 | 0.0923 |

Resize and blur are the weakest transforms — both destroy the high-frequency artifacts the frequency branch relies on, so accuracy degrades the most under heavy downsampling or blurring. See `outputs/robustness_chart.png` for the visual summary.

## Error Analysis Note

On the held-out evaluation set (20 misclassified examples: 10 false positives, 10 false negatives out of the full validation set):

- **False positives** (real images predicted as fake, confidence >0.99): almost all are real images that were heavily compressed and downsampled together (e.g. `center_crop-80_resize-x0.25_..._jpeg-q30`). Aggressive resize + low-quality JPEG recompression introduces compression artifacts that resemble the frequency-domain signatures the model associates with generated images.
- **False negatives** (fake images predicted as real, confidence <0.01): concentrated in heavily blurred and downsampled fake images (e.g. `resize-x0.25_..._blur-sigma_2`), plus a handful of clean fake images from unseen generator families (e.g. `FAKE\137.jpg`). Downsampling and blur wash out the generator-specific frequency artifacts the model relies on, causing it to default toward "real."
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
Zacchaeus Tan
- 


## Limitations

- The project is still a hackathon-scale prototype, not a production moderation system.
- Pure inference is currently coupled to the training-dataset download/credential check — running `techjam_cli.py infer` still requires Kaggle credentials and a ~12GB one-time dataset download, even though the required inference deliverable itself doesn't need that data. Given more time, inference-only mode would skip this entirely.
- External OOD evaluation depends on a local labeled CSV and is optional.
- The model is robust, but resize and blur remain the biggest weak spots.

## Notes

- `split_manifest.json` is committed so train/validation assignments stay stable across runs.
- Generated paths in CSV outputs are now normalized to avoid machine-specific absolute prefixes when possible.
