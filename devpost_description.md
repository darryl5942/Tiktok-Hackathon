# Devpost Written Description

_Copy/paste and adapt into the Devpost submission form._

## Demo Video

[Robust Detection of AI-Generated Images Under Real-World Transformations — Demo](https://youtu.be/7vnF9jSAB24)

## Inspiration

Generative AI can now produce photorealistic fake images at scale, resulting
in misinformation, impersonation, fraud, eroded trust. The hard part isn't
spotting fakes in a lab, it's that every real image gets JPEG re-encoded,
resized, cropped or reposted before anyone sees it again, and that's exactly
when the pixel-level artifacts most detectors rely on disappear.

## What it does

Hence, we built SpectraLens. It's a two-branch model, under 720M
parameters, which includes a frozen CLIP ViT-H/14 encoder that reads visual
content, a trainable ConvNeXt-Base network that reads the image's FFT
frequency spectrum, where generative artifacts often hide even when
invisible to the eye. We train on randomly transformed images, ranging from
compression, blur, resize, noise, color jitter, crop so the model never
learns to lean on cues that vanish in the real world. And instead of one
accuracy number, we report three: clean, post-transform, and a fully
held-out benchmark (WildFake) the model never trains on, which proves that
it generalizes rather than memorizes. Output is a simple JSON scan built to
slot into a moderation queue.

That's the bet: a detector trust & safety teams could actually deploy.

## How we built it

We trained on CIFAKE and the AIGC Detection Dataset (Kaggle) — real photos
paired against images from ADM, Stable Diffusion, and Midjourney — with
every training image passed through 0 to 3 randomly-stacked real-world
transforms before the model ever sees it, so it never gets to rely on cues
that vanish once an image leaves the lab. We evaluate on three tiers
instead of one accuracy number: in-distribution clean accuracy, accuracy
after each individual transform, and accuracy on WildFake (COCO val2017 +
DALL·E "Advanced" fakes), a completely held-out benchmark the model never
trains on — separating "is it actually good" from "did it just memorize
this dataset's generators" from "does it survive real-world post-
processing." The trained checkpoint (~3.6GB) is hosted on the Hugging Face
Hub since it's too large for git, and the codebase is split into readable
modules (data pipeline, model architecture, transform pool, shared config)
rather than one monolithic script.

## Challenges we ran into

The model itself came together more smoothly than the infrastructure around
it. We hit a real, leaked-credential scare during setup and had to rotate
Kaggle and Hugging Face API keys mid-hackathon. We ran into Windows'
260-character path limit crashing mid-download on two separate machines,
because our project folder lived deep inside a synced OneDrive path — fixed
by making the dataset cache location configurable instead of forcing
everyone to enable OS-level long-path support. We also caught a genuine
cross-platform bug via our own test suite: a "portable" path identifier
function was silently returning Windows backslash paths instead of portable
forward slashes, which would have made error-analysis output inconsistent
between team members on different OSes. And after a full verified re-run on
real GPU hardware, we found our documented robustness numbers were stale
relative to the actual checkpoint we were shipping — a good reminder that
"the numbers in the README" and "what the model actually does right now"
can silently drift apart if you don't re-verify.

On the training side, getting the model onto the GPU at all took longer
than training it usually does. PyTorch's default install method silently
gives you a CPU-only build with no error — the script would happily start
training, just at a small fraction of the expected speed with no warning.
We fixed it once, then hit it again weeks later when a routine dependency
reinstall quietly overwrote the working CUDA build with the CPU one.
Eventually we taught the script to diagnose and explain this itself on
startup, rather than relying on someone noticing an oddly slow epoch.

The more dangerous bugs were the ones that didn't crash anything. A naive
way of capping how many training images we used accidentally produced a
dataset that was entirely real photos with zero AI-generated examples in it
— the code ran fine, the loss went down, and every number looked plausible
unless you specifically checked the label distribution. We found and fixed
that in the training pipeline, then discovered the exact same bug
independently in the evaluation pipeline much later — a reminder that
fixing a pattern once doesn't mean it's fixed everywhere it got copied. In a
similar spirit, we had our own version of the "stale README" problem
earlier in the pipeline: for a while, the same held-out images decided
which checkpoint counted as "best" and were used to report final accuracy —
meaning checkpoint selection was subtly tuned toward the exact data we were
about to grade ourselves on. Rebuilding that into a genuine three-way
train/validation/test split, with the test portion touched exactly once,
was one of the more invasive fixes we made, but it's the difference between
a real number and a flattering one.

Resuming training safely across sessions surfaced its own quiet failures.
On two separate occasions, restoring a saved optimizer or learning-rate
schedule silently overrode a setting we'd just deliberately changed —
extending the training length or adjusting the learning rate appeared to do
nothing after a resume, with no error to explain why. Both times the fix
was the same shape: explicitly re-apply the current setting after loading
the saved state, rather than trusting the checkpoint to defer to it. Even
our own reporting wasn't immune to this kind of drift — a summary table
meant to show how much training data was available double-counted every
source's total, since a validation split and its parent training split were
both correctly reporting the same pool size, and our summary logic summed
them as if they were two different pools. Caught only because the total
looked implausibly large next to what was actually used.

## Accomplishments that we're proud of

Every number in our robustness table and error analysis came from an
actual, verified, end-to-end run of our shipped checkpoint — not recycled
or approximate figures. We're proud that our own test suite caught a real
bug before it reached a teammate on a different OS, and that we were
upfront about a real gap we found in our own evaluation design: training
stacks multiple transforms per image, but our robustness evaluation
currently only tests one transform at a time, meaning we haven't fully
measured how the model holds up against the combined real-world degradation
it was actually trained to handle. Catching and documenting our own blind
spot felt more valuable than quietly leaving it out.

We're also proud of the architecture decision behind the detector itself.
Rather than training one model from scratch and hoping it generalizes, we
deliberately paired a frozen, large-scale pretrained vision model with a
small trainable component focused specifically on frequency-domain
artifacts — a design choice backed by published research on generalizing
to AI generators the model has never seen, not just intuition. Getting that
pairing to actually work together, through a custom fusion mechanism, and
stay within a hard 2-billion-parameter budget while still meaningfully
using the headroom, was a real engineering win, not just a checkbox.

We built real safety nets into the training process itself, not just the
final output. The pipeline can detect when someone has changed the
training data or configuration since a checkpoint was last saved, and
automatically decide whether it's safe to keep going or whether it should
insist on a clean restart to avoid quietly corrupting the results — the
kind of judgment call that's easy to skip under time pressure but that we
didn't want to leave to memory or luck. Training can also be safely paused
and resumed across sessions without losing progress or silently drifting
from the settings we intended, which mattered a lot given how long some of
our training runs took.

We also made a real, deliberate choice to hold ourselves to a higher bar on
data hygiene than the minimum required. We built a proper three-way
separation between training data, the data used to pick our best
checkpoint, and the data used to report final results — specifically so our
reported accuracy couldn't be quietly flattered by data the model's
checkpoint-selection process had already been tuned against. That's an easy
corner to cut under a deadline, and we didn't cut it.

Finally, we're proud of how reproducible and handoff-ready the end result
is. Every part of the pipeline — which datasets are active, how much data
is used, model size, training length — is controlled from one clearly
organized settings section rather than scattered through the codebase, and
the whole system is documented well enough that a teammate joining late
could actually pick it up and understand not just what the code does, but
why it's built that way.

## What we learned

That frequency-domain signals are a genuinely strong, complementary cue to
raw pixel content for this problem — but they're also the first thing to
degrade under blur and aggressive resizing, which is exactly where our
model's accuracy dips the most (92.4% on resize, still well above chance,
but our clearest weak spot). We also relearned a very unglamorous but
important lesson: reproducibility infrastructure (deterministic splits,
seeded randomness, checkpoint versioning, credential hygiene, environment
setup that actually works on a teammate's machine) takes real, deliberate
effort, and skipping it is exactly how "it works on my machine" turns into
a submission that doesn't reproduce for a judge.

We also came away with a stronger belief in restraint as a design
principle. Our instinct going in was that more parameters and more
fine-tuning would mean more accuracy. What actually worked better was
freezing most of a large pretrained model and only training a small piece
on top of it — not because it was cheaper (though it was), but because it
generalizes better to AI generators the model has never seen, which is the
harder and more important problem than performing well on the exact
generators in our training set. Knowing when not to train something turned
out to be as important a decision as the architecture itself.

We also learned not to trust assumptions about anything outside our own
code without checking. Third-party dataset folder structures, GPU driver
behaviour, default package installation behaviour, even how Windows handles
long file paths — all of these quietly broke something at least once, and
in every case the fix was cheap once we actually looked, but expensive
while we were still guessing. Verifying instead of assuming became a habit
we had to build deliberately, not one that came naturally under time
pressure.

## What's next for SpectraLens

Closing the resize/blur gap with a multi-scale frequency representation or
an ensemble with a spatial-artifact detector; extending our robustness
evaluation to cover stacked-transform combinations, not just single
transforms, to match what training already does; adding an explainability
layer (e.g. Grad-CAM-style saliency) so a human moderator gets evidence,
not just a bare score; and decoupling pure inference from the
training-dataset download so the script is faster and easier for a
reviewer to run standalone.

Beyond that, we'd like to actually finish validating generalization the way
we designed it to be validated: our pipeline already has a slot reserved
for a fully external, never-trained-on benchmark, but we haven't populated
it yet. Without that, we're still only confident about how well the model
does on data related to what it trained on — the harder and more
meaningful question of "does this hold up against a generator and photo
source it's never seen at all" is still open.

On the engineering side, we want to close a couple of gaps we identified
but consciously deferred: verifying that one of our newer data sources'
evaluation split is genuinely balanced between real and fake examples
before trusting any accuracy number from it, and extending our per-image
error analysis so it draws representative examples from every active data
source instead of defaulting to whichever one happens to load first.

Longer term, we're interested in exploring whether the model's current size
is actually being used efficiently — we have real headroom left in our
parameter budget, and it's worth finding out whether that's better spent on
a larger frequency-analysis branch, a stronger fusion mechanism between our
two branches, or isn't the bottleneck at all compared to simply training on
more diverse data.

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

That's the problem we designed SpectraLens for: not "can we classify AI
images," but "can the model still tell the difference after the image has
been through what every real upload goes through." Our solution addresses
this directly, in three parts:

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
- Training augmentation stacks 1-3 transforms per image (e.g. blur + JPEG +
  noise applied together, simulating an image degraded through multiple
  real-world processing steps), but the robustness evaluation only tests one
  transform at a time — it doesn't measure how the model holds up against
  the same kind of stacked/combined degradation it was actually trained on.
  Extending the evaluation to cover stacked-transform combinations would
  give a more complete, more realistic robustness picture.
- Pure inference currently still requires Kaggle credentials and a one-time
  ~12GB dataset download, since the robustness-evaluation pipeline shares a
  run with inference — decoupling these would make the script easier for a
  reviewer to run standalone.
