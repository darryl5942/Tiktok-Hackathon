# Demo Video Script (~3 minutes)

Screen-recording + voiceover. Swap in your own team's voice/wording — this is
a structural skeleton, not a script to read verbatim.

---

## 1. Problem framing (0:00-0:30)

> "Generative AI can now produce photorealistic fake images at scale — that's
> a real risk for misinformation, impersonation, and platform trust. But most
> detectors are only tested on clean, unmodified images. In the real world,
> images get compressed, resized, cropped, and filtered before anyone sees
> them again. We built a detector that's evaluated specifically on how well
> it holds up *after* that kind of real-world post-processing, not just on
> lab-clean data."

*(Show: title slide or just the README's "What this project does" section on
screen while talking.)*

## 2. Technical approach (0:30-1:15)

> "Our model has two branches. A frozen CLIP ViT-H/14 encoder reads the
> image's visual content. A trainable ConvNeXt-Base network looks at the
> image's frequency spectrum — the FFT — because generative models tend to
> leave subtle frequency-domain artifacts that aren't visible to the eye but
> show up clearly in frequency space. We combine both signals into a single
> confidence score. The whole model is under 720 million parameters combined,
> well within the 2B parameter limit."

*(Show: the architecture bullet points in the README, or a quick diagram if
your team has one.)*

## 3. Live demo — inference (1:15-2:00)

> "Here's the model running end-to-end on a directory of images."

*(Screen: open a terminal, run:)*

```bash
AIGC_SKIP_TRAINING=1 python techjam_cli.py infer --input-dir inference_images
```

> "It scores every image and writes out a JSON file with a confidence score
> per image — here's `outputs/preds.json`."

*(Screen: open `outputs/preds.json`, scroll through a few entries. Optionally
open 2-3 of the actual source images side-by-side so viewers can see real vs.
fake examples and their scores.)*

## 4. Robustness results (2:00-2:35)

> "The key question isn't just 'does it work on clean images' — it's 'does it
> still work after the image has been resized, blurred, compressed, or
> cropped.' Here's our robustness table across six transform families."

*(Screen: show the Robustness Evaluation Summary table from the README, or
`outputs/robustness_chart.png`.)*

> "Clean accuracy is about 96%. JPEG compression, noise, and color jitter
> barely move the needle. Blur and resize are our weakest spots — both wash
> out the high-frequency signal our frequency branch depends on, and that's
> the main trade-off we'd want to address with more time."

## 5. Error analysis + wrap-up (2:35-3:00)

> "Looking at our errors: false positives tend to be real images that were
> heavily downsampled and recompressed at low quality — the compression
> artifacts look similar to generator artifacts to the model. False negatives
> are mostly heavily blurred fake images, where the frequency signal is
> mostly gone. This tells us the next step is either a transform-invariant
> frequency representation, or pairing this with a spatial-artifact detector
> as an ensemble."

> "This is a hackathon-scale prototype, but the approach — combining a
> semantic branch with a frequency branch, and evaluating robustness
> explicitly rather than just clean accuracy — is a direction we think
> matters for real content-moderation pipelines. Thanks for watching."

---

## Notes for recording

- Keep total runtime to 2-4 minutes — judges are watching many of these.
- Screen-record the actual `techjam_cli.py infer` command running live rather
  than a pre-baked screenshot — "the demo runs reliably" is explicitly part
  of the Technical Execution rubric (35% of the score).
- Don't use any copyrighted/trademarked images beyond your own dataset
  samples (CIFAKE / AIGC Detection dataset images are fine since they're the
  datasets you're using; avoid pulling in random internet images).
- Upload to YouTube as **Public** (not Unlisted) before linking it in Devpost.
