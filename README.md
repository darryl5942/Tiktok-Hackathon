# TikTok Hackathon AIGC Detector

An AI-generated image detector built for the TikTok TechJam hackathon. The project uses a two-branch PyTorch model that combines:

- an RGB image encoder
- a frequency-domain encoder based on FFT features
- a lightweight fusion layer for final classification

The main script, [`aigc_detector_3.py`](./aigc_detector_3.py), handles:

- dataset loading
- train/validation/test splitting
- model training
- checkpointing
- single-folder inference
- robustness evaluation under common image transformations

## What It Detects

The model is designed to classify images as:

- `0` = real
- `1` = AI-generated / fake

It was built to be resilient to real-world perturbations such as:

- JPEG compression
- blur
- resize
- noise
- color jitter
- center crop

## Project Structure

- [`aigc_detector_3.py`](./aigc_detector_3.py) - end-to-end training, inference, and evaluation script
- [`requirements.txt`](./requirements.txt) - Python dependencies
- `checkpoints/` - saved model weights and training state
- `outputs/` - inference predictions and evaluation CSVs
- `inference_images/` - drop images here for local inference
- `.cache/` - local Kaggle and Hugging Face caches

## Requirements

- Python 3.11 or newer
- PyTorch and torchvision
- Kaggle access for CIFAKE, if enabled
- Hugging Face access token for SID_Set, if enabled

Create and activate a virtual environment before installing packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you need CUDA-enabled PyTorch, install `torch` and `torchvision` from the official PyTorch index first, then install the rest of the requirements.

## Setup

Create a `.env` file in the project root if you want to use the remote datasets:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
HF_TOKEN=your_huggingface_token
```

Notes:

- `KAGGLE_USERNAME` and `KAGGLE_KEY` are required only when `USE_CIFAKE = True`
- `HF_TOKEN` is required only when `USE_SID_SET = True`
- the script writes caches into the project directory, so it stays self-contained

## Running The Script

Open [`aigc_detector_3.py`](./aigc_detector_3.py) and check these flags near the top:

- `USE_CIFAKE = True` to train/evaluate on CIFAKE
- `USE_SID_SET = True` to add the SID_Set stream
- `SKIP_TRAINING = True` to skip training and go straight to inference/eval using an existing checkpoint

Then run:

```bash
python aigc_detector_3.py
```

### Training

When `SKIP_TRAINING = False`, the script:

1. downloads or streams the enabled datasets
2. builds the train/validation split
3. trains the model
4. writes checkpoints into `checkpoints/`

The script uses:

- early stopping
- cosine learning-rate scheduling
- mixed precision when available
- a persistent split manifest for reproducibility

### Inference

To run inference on local images:

1. put images into [`inference_images/`](./inference_images) or pass another folder path to `run_inference()`
2. load a trained checkpoint from `checkpoints/best_model.pt`
3. write predictions to `outputs/preds.json`

Expected output format:

```json
[
  {
    "image_path": "path/to/image.png",
    "pred": 0.9732
  }
]
```

The `pred` value is the model's confidence score for the fake class.

### Robustness Evaluation

The script also evaluates the model on transformed versions of the held-out evaluation set and writes CSV reports to `outputs/`, including:

- `robustness_indist.csv`
- `robustness_table.csv` if you use the helper's default output path

## Configuration Notes

The script currently defaults to:

- CIFAKE enabled
- SID_Set disabled
- training enabled

If you only want inference on an existing checkpoint, set:

```python
SKIP_TRAINING = True
```

If you want to keep the run lightweight while testing the pipeline, you can also temporarily disable one of the data sources.

## Reproducibility

The project stores a persistent `split_manifest.json` in the repo root so that image assignments to train and validation remain stable across runs. This helps keep experiments consistent when the same files are seen again.

## Outputs

After a run, expect to see files such as:

- `checkpoints/best_model.pt`
- `checkpoints/latest_checkpoint.pt`
- `outputs/preds.json`
- `outputs/robustness_indist.csv`

## License

No license has been specified yet. Add one if you plan to share or reuse the project publicly.
