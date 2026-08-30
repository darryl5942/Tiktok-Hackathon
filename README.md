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

- `aigc_detector_3.py` - end-to-end training, inference, and evaluation script
- `requirements.txt` - Python dependencies
- `checkpoints/` - saved model checkpoints
- `outputs/` - inference, robustness, and error-analysis artifacts
- `split_manifest.json` - persistent train/validation split assignments

## Setup

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

1. Ensure the expected datasets are available through Kaggle/Hugging Face credentials.
2. Run the script with `AIGC_SKIP_TRAINING=0` to train or resume from checkpoints.
3. Review the outputs in `outputs/`:
   - `preds.json` for inference predictions
   - `robustness_indist.csv` and `robustness_summary_compact.csv` for held-out robustness
   - `error_analysis.csv` for false positives and false negatives
   - `robustness_chart.png` for the visual summary

## Current Observations

The current held-out robustness summary shows:

- clean accuracy around 0.9485
- JPEG compression is only slightly worse than clean
- blur and resize are the weakest transform families, which matches the problem statement’s emphasis on post-processing robustness

The detector now also calibrates its decision threshold from the validation set and saves it to `outputs/decision_threshold.json`, which makes the final classification rule less arbitrary than a fixed 0.5 cutoff.

The error analysis shows the model tends to struggle on heavily processed real images and some heavily degraded fake images, which is the main trade-off to address next.

## Limitations

- The project is still a hackathon-scale prototype, not a production moderation system.
- The main script is intentionally single-file for speed of iteration, so it is less modular than a production repo.
- External OOD evaluation depends on a local labeled CSV and is optional.
- The model is robust, but resize and blur remain the biggest weak spots.

## Notes

- `split_manifest.json` is committed so train/validation assignments stay stable across runs.
- Generated paths in CSV outputs are now normalized to avoid machine-specific absolute prefixes when possible.
