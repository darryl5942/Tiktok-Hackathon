# Devpost Written Description

_Copy/paste and adapt into the Devpost submission form._

## How our solution addresses the problem statement

Generative AI tools now produce images nearly indistinguishable from real
photos, and on a platform at TikTok's scale that creates real risk:
misinformation, impersonation, fraud, and erosion of trust in what people
see. But detecting AI-generated images in a lab isn't the hard part — it's
that every image reaching a real platform has already been re-compressed,
resized, cropped, or lightly edited (a JPEG re-encode on upload, a thumbnail
generated for a feed, a screenshot-and-repost). Most of the subtle
pixel-level artifacts that make an image "obviously fake" to a classifier
are exactly the artifacts that get destroyed first by that pipeline. A
detector that scores well on clean benchmark images but collapses after one
re-compression pass isn't a working solution — it's a demo that only works
inside its own dataset.

That's the problem we designed for: not "can we classify AI images," but
"can the model still tell the difference after the image has been through
what every real upload goes through." Our solution addresses this directly,
in three parts:

**A two-branch architecture built for different failure modes.** Instead of
a single network reading raw pixels, each image is processed through two
branches in parallel:
- a **frozen CLIP ViT-H/14 encoder** (~632M params, LAION-2B weights) that
  reads high-level semantic/visual content, and
- a **trainable ConvNeXt-Base** (~88M params) that reads the image's FFT
  log-magnitude spectrum.

Generator artifacts often show up as structured patterns in frequency space
even when they're subtle or invisible in raw pixels, and the two cues
degrade differently under different transforms — giving the model more than
one way to still be right after an image has been altered. A lightweight
cross-modal attention layer fuses both signals before the final
classification head. Total model size is ~720M parameters, comfortably
under the hackathon's 2B-parameter limit.

**Robustness built into training, not just measured afterward.** Every
training image is passed through 0 to 3 randomly-stacked real-world
transforms — JPEG compression, blur, resize, noise, color jitter, or center
crop, applied in sequence — before the model sees it, so the model never
gets to rely on cues that vanish once an image leaves the lab.

**Three-tier evaluation instead of one accuracy number.** We report (1)
in-distribution clean accuracy, (2) in-distribution accuracy after each
transform family, and (3) accuracy on a completely held-out external
benchmark (WildFake — COCO val2017 real images + DALL·E "Advanced" fakes)
that the model never trains on — separating "is it actually good" from "did
it just memorize this dataset's generators" from "does it survive
real-world post-processing."

On top of the model, the pipeline includes what a real deployment needs:
reproducible, deterministic train/val dataset splits (persisted so they stay
stable across runs); streaming data ingestion so nothing has to fit in
memory at once; mixed-precision training with early stopping and a decision
threshold calibrated from held-out data rather than a fixed 0.5 cutoff; and
checkpointing that resumes safely (detecting and warning about changes in
data sources/volume between runs) instead of silently retraining on drifted
data.

The direct beneficiaries are platform trust & safety and content moderation
teams, who need a detector that survives the platform's own image pipeline
rather than one that only works on pristine inputs. The output format
reflects that: a simple JSON scan of an input folder producing per-image
confidence scores (`image_path` + `pred`), designed to slot into a
moderation queue rather than requiring a bespoke integration layer. Beyond
platforms, this protects everyday users indirectly — every AI-generated
image caught before it spreads is one less piece of convincing
misinformation or impersonation reaching someone with no way to verify it
themselves.

## Headline results

- **Clean-data accuracy: 97.5%, ROC AUC: 0.997**
- **Accuracy retained under transforms**: 96.3% (JPEG), 95.3% (noise), 97.2%
  (color jitter), 95.5% (center crop) — holds up well; weakest cases are
  94.8% (blur) and 92.4% (resize), still far above chance
- Full per-transform breakdown in the Robustness Evaluation Summary below

## Development tools used

- VS Code
- Python 3.12, `venv`
- Local NVIDIA GPU (CUDA) for training/inference
- Git / GitHub for version control
- `.env`-based credential management for Kaggle and Hugging Face API access
- Hugging Face Hub for hosting the trained checkpoint (too large for git)

## Models / APIs used

- **CLIP ViT-H/14** (`vit_huge_patch14_clip_224.laion2b`, via `timm`, frozen)
  — RGB-branch backbone for semantic feature extraction
- **ConvNeXt-Base** (via `timm`, trainable) — frequency-branch backbone for
  detecting generator artifacts in FFT space
- Custom cross-modal attention fusion layer (our own implementation, not
  pretrained) — combines the two branches before classification

## Libraries and frameworks used

- PyTorch / torchvision — model definition, training loop, mixed-precision
  training, inference
- `timm` — pretrained backbone architectures (CLIP ViT-H/14, ConvNeXt-Base)
- `scikit-learn` — ROC-AUC scoring, train/val split utilities
- `pandas`, `numpy` — data handling and results aggregation
- `pillow` — image loading and preprocessing
- `kagglehub` — CIFAKE and AIGC Detection dataset downloads from Kaggle
- `datasets`, `huggingface_hub` — streaming SID_Set and hosting/downloading
  the trained checkpoint
- `matplotlib` — robustness summary chart
- `tqdm` — training/inference progress monitoring
- `python-dotenv` — environment/credential loading

## Datasets and assets used

- **CIFAKE** (Kaggle, `birdy654/cifake-real-and-ai-generated-synthetic-images`)
  — real vs. Stable-Diffusion-generated image pairs; core training,
  validation, and in-distribution test data
- **AIGC Detection Dataset** (Kaggle, `shxrlenee/aigc-detection-dataset`),
  including its own pre-transformed subset — additional real/fake training
  data spanning ADM, Stable Diffusion 1.5, and Midjourney generators, plus
  built-in post-processing variants
- **SID_Set** (Hugging Face, `saberzl/SID_Set`, streamed) — additional
  real/AI-generated image source; disabled by default in the current
  checkpoint due to streaming reliability during development (see
  Limitations)
- **WildFake subset** (COCO val2017 real images + DALL·E "Advanced" fake
  images) — used exclusively as an external, out-of-distribution benchmark;
  the model is never trained on this data, so it serves as an honest
  measure of generalization to a generator family the model hasn't seen

## Robustness evaluation summary

Held-out validation accuracy/AUC by transform family, from a full
end-to-end run of the current checkpoint:

| Transform | Accuracy | AUC | Accuracy drop vs. clean |
|---|---|---|---|
| Clean (no transform) | 0.9750 | 0.9970 | — |
| Color jitter | 0.9720 | 0.9967 | 0.0030 |
| JPEG compression | 0.9630 | 0.9943 | 0.0120 |
| Noise | 0.9530 | 0.9912 | 0.0220 |
| Center crop | 0.9550 | 0.9924 | 0.0200 |
| Blur | 0.9477 | 0.9891 | 0.0273 |
| Resize | 0.9235 | 0.9779 | 0.0515 |

Resize and blur are the weakest transforms — both destroy the high-frequency
artifacts the frequency branch relies on — but the degradation is modest
(worst case 92.4% accuracy, still well above chance).

## Error analysis (brief)

- **False positives** (real images predicted as fake, confidence up to
  0.97): images the model confidently, incorrectly calls AI-generated.
- **False negatives** (fake images predicted as real, confidence as low as
  0.003): fakes the model confidently, incorrectly calls real — frequency
  artifacts washed out by heavy blur/downsampling, or from generator
  families underrepresented in training.
- **Trade-off**: the frequency branch is central to clean-image accuracy but
  is also the main failure point under transforms that suppress
  high-frequency content (blur, aggressive resize). A production system
  would need either a transform-invariant frequency representation or an
  ensemble with a spatial-artifact detector to close this gap.

## Limitations / what we'd improve with more time

- Hackathon-scale prototype, not a production moderation system.
- Resize and blur remain the biggest robustness gaps — a transform-invariant
  frequency representation, or an ensemble with a spatial-artifact detector,
  would likely close this gap.
- No explainability layer yet (e.g. Grad-CAM-style saliency showing which
  pixels drove a "fake" verdict) — the problem statement lists this as an
  in-scope idea, and it's a natural next step: useful evidence for a human
  moderator rather than a bare score. It would also only explain a
  whole-image call, not localize a partially-edited region, which would need
  training on masked tampered-image data — a scoped extension, not a
  redesign.
- External OOD evaluation (WildFake) currently depends on a manually
  supplied labeled CSV rather than an automated download.
- SID_Set is disabled by default due to streaming reliability; enabling it
  (and other generator sources) would likely improve generalization further.
- Pure inference currently still requires Kaggle credentials and a one-time
  ~12GB dataset download, since the robustness-evaluation pipeline shares a
  run with inference — decoupling these would make the script easier for a
  reviewer to run standalone.
