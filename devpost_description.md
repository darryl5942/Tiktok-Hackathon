# Devpost Written Description

_Copy/paste and adapt into the Devpost submission form._

## How our solution addresses the problem statement

Our solution is a robust, image-level AIGC (AI-generated content) detector: given
an image, it outputs a confidence score (0-1) for how likely the image is
AI-generated, and it is explicitly trained and evaluated to stay reliable under
the kinds of post-processing real content goes through in the wild — JPEG
recompression, resizing, blurring, noise, color jitter, and center cropping —
rather than only performing well on pristine, unmodified images.

The architecture is a two-branch classifier:

- **RGB branch**: a frozen CLIP ViT-H/14 encoder (~632M params, LAION-2B
  weights) extracts general-purpose semantic/visual features. It is never
  fine-tuned, which keeps the visual representation stable and prevents
  overfitting to any single generator's style.
- **Frequency branch**: a trainable ConvNeXt-Base (~88M params) operates on the
  log-magnitude FFT spectrum of each image, learning the frequency-domain
  artifacts that generative models tend to leave behind (upsampling
  checkerboard patterns, GAN/diffusion spectral signatures) — signals that are
  often invisible in the raw pixel/RGB domain.

The two branches' features are fused into a final classification head. The
decision threshold is calibrated from a held-out validation set (rather than a
fixed 0.5 cutoff) and saved alongside the model for reproducibility.

We evaluate both in-distribution robustness (same data sources as training,
under each transform family) and out-of-distribution generalization (a
completely unseen benchmark — WildFake — that the model is never trained on),
which directly targets the problem statement's dual requirement of accuracy
*and* robustness to real-world post-processing.

## Development tools used

- VS Code
- Python 3.12 (`.venv`)
- Git / GitHub for version control
- Hugging Face Hub for hosting the trained checkpoint (too large for git)

## Models / APIs used

- **CLIP ViT-H/14** (`vit_huge_patch14_clip_224.laion2b`, via `timm`) — frozen
  RGB feature extractor
- **ConvNeXt-Base** (via `timm`) — trainable frequency-domain feature
  extractor

## Libraries and frameworks used

- PyTorch / torchvision — model, training loop, inference
- `timm` — pretrained backbone architectures (CLIP ViT-H/14, ConvNeXt-Base)
- `scikit-learn` — ROC-AUC scoring, train/val split utilities
- `pandas`, `numpy` — data handling and metrics aggregation
- `pillow` — image I/O
- `datasets` + `huggingface_hub` — streaming dataset loading (SID_Set) and
  model checkpoint hosting
- `kagglehub` — dataset download (CIFAKE, AIGC Detection dataset)
- `matplotlib` — robustness summary chart
- `python-dotenv` — environment/credential loading

## Datasets and assets used

- **CIFAKE** (Kaggle, `birdy654/cifake-real-and-ai-generated-synthetic-images`)
  — real vs. AI-generated (Stable Diffusion) image pairs, core training data
- **AIGC Detection Dataset** (Kaggle, `shxrlenee/aigc-detection-dataset`),
  including its own pre-transformed subset — additional real/fake training
  data plus built-in post-processing variants
- **SID_Set** (Hugging Face, `saberzl/SID_Set`, streamed, optional/disabled by
  default) — additional AI-generated image source
- **WildFake subset** (COCO val2017 real images + DALL·E "Advanced" fake
  images) — held-out, out-of-distribution benchmark used only for evaluation,
  never for training, to test generalization to an unseen generator/photo
  source

## Robustness evaluation summary

| Transform | Accuracy | AUC | Accuracy drop vs. clean |
|---|---|---|---|
| Clean (no transform) | 0.959 | 0.9929 | — |
| Color jitter | 0.9605 | 0.9925 | -0.0015 |
| JPEG compression | 0.9549 | 0.9910 | 0.0041 |
| Noise | 0.9543 | 0.9905 | 0.0047 |
| Center crop | 0.9340 | 0.9854 | 0.0250 |
| Blur | 0.9162 | 0.9660 | 0.0428 |
| Resize | 0.8667 | 0.9325 | 0.0923 |

Resize and blur are the weakest transforms, since both suppress the
high-frequency artifacts the frequency branch relies on.

## Error analysis (brief)

- **False positives**: real images that were heavily downsampled and
  recompressed at low JPEG quality — the compression artifacts resemble the
  frequency signatures the model associates with generated content.
- **False negatives**: fake images that were heavily blurred/downsampled, plus
  a few clean fakes from generator families not represented in training —
  both wash out or omit the frequency artifacts the model depends on.

## Limitations / what we'd improve with more time

- Hackathon-scale prototype, not a production moderation system; the core
  script is intentionally single-file for iteration speed.
- Resize and blur remain the biggest robustness gaps — a transform-invariant
  frequency representation, or an ensemble with a spatial-artifact detector,
  would likely close this gap.
- External OOD evaluation (WildFake) currently depends on a manually supplied
  labeled CSV rather than an automated download.
- SID_Set is disabled by default due to streaming reliability; enabling it
  (and other generator sources) would likely improve generalization further.
