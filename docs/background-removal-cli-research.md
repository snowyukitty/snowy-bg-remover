# Local-First Background Removal CLI Research and Design

Date: 2026-06-19

This document designs a production-grade local CLI background removal tool for
single-character/emote cutouts, with future GUI and agent/MCP integration in
mind.

## Implementation checkpoint

As of this checkpoint, the repository has a working local MVP:

- `cutout` single, batch, glob, and `--check` modes with stdout JSON
- `cutout models list/status/download` provisioning commands
- default `isnet-anime` ONNX adapter, pinned by SHA256
- registered `birefnet-general-lite` ONNX comparison adapter, pinned by SHA256
- quality `toonout` PyTorch BiRefNet adapter, pinned by SHA256 and paired with a
  pinned BiRefNet runtime revision
- offline-first cache under the platform cache directory
- soft-alpha-native topology with largest-core hysteresis reconstruction
- strict confidence failure for no subject, excessive coverage, high uncertainty,
  and multiple high-confidence cores
- bounded interior hole repair and detached-blob removal
- edge RGB decontamination by PyMatting foreground estimation with nearest opaque
  foreground color bleed fallback
- bounded closed-form alpha refinement for the quality path
- bbox thresholding to keep faint residual haze from expanding trim/framing boxes
- atomic PNG output and explain artifacts via `--explain-dir`
- 128px/32px/small-output path using linear-light premultiplied RGBA resize
- setup and operations guide for provisioning another machine:
  [`docs/SETUP.md`](SETUP.md)

Still pending for production hardening: corpus benchmarks, threshold calibration,
optional InSPyReNet comparison, stricter drop-effects policy, package
signing/notarization, and MCP/GUI adapters.

## 1. Problem Statement

Primary capability: convert one ChatGPT-generated single-character illustration
into a clean transparent PNG before it is published to Snowy Style Atlas'
`emote-wall`.

The hard part is that the generation prompt deliberately does not constrain the
background. Inputs can contain solid colors, gradients, scenery, texture,
white/near-white backgrounds, or faux checkerboards. The subject can also contain
white hair, pale clothes, loose sketch lines, props, and soft edges. Therefore
the tool must not rely on chroma keying or background color assumptions. It needs
a learned semantic/matting model, strict validation, and a failure path that does
not silently emit unusable transparent output.

Core contract:

- Single image: `cutout --input <path> --output <path>`
- Batch: `cutout --input-dir <dir> --output-dir <dir>` or `cutout --glob`
- Headless, deterministic, local/offline after model download
- macOS arm64 first, CPU fallback required
- Atomic output writes: temp file, fsync best effort, rename
- stdout is machine-readable JSON; logs go to stderr
- Exit code `0` only for a confident usable subject
- A distinct `no_confident_subject` signal for uncertain segmentation
- `--check` dry-run reports bbox/detectability and writes nothing
- Preserve original resolution unless explicit framing flags are used
- Output true RGBA alpha, soft anti-aliased edges, decontaminated edge colors
- Idempotent on already-transparent input

### Architecture correction: soft alpha is the source of truth

The engine must not threshold the model output first and then try to reconstruct
soft edges later. Modern segmentation/matting models often produce useful
continuous alpha already. The correct design is:

```text
soft alpha A
  -> derive seed mask from A >= high_threshold
  -> derive support mask from A >= low_threshold
  -> run topology on binary masks only
  -> reconstruct keep mask from seed over support
  -> fill safe interior holes
  -> A_final = A * keep_mask
  -> foreground/edge decontamination
  -> framing and output
```

Binary masks are topology tools, not the final alpha. This preserves native
anti-aliasing, hair wisps, sketchy linework, and transparent edges. Thresholds
should be fixed per profile and calibrated on the corpus. Per-image adaptivity
should mostly affect accept/reject confidence, not geometry, because geometry
adaptivity makes near-identical inputs hard to debug.

The key primitive is hysteresis-style morphological reconstruction: start from a
high-confidence foreground seed and expand only through the low-threshold
support. This keeps soft attached props reachable from the character core while
dropping detached sparkles, hearts, dust, and background blobs.

### AI-generated source-image artifact model

Most source images come from high-quality image generation (`gpt-img-2` at the
time of writing), but generated images can contain background artifacts that are
visually plausible and therefore hard for naive post-processing to distinguish:

- faux transparency: checkerboard or grid backgrounds rendered as real pixels
- stray decorative marks: sparkles, dust, color flecks, texture, brush noise
- near-subject halos: glow, rim light, shadow, bloom, watercolor bleed
- background-object confusion: signs, hearts, props, scenery, or repeated motifs
  near the character
- edge ambiguity: pale hair/clothes on pale backgrounds, transparent accessories,
  soft sketch lines, motion smear, line-art gaps
- compositional ambiguity: the model may create a secondary object that looks
  foreground-like but should be dropped for emote use

This changes the design from "remove background" to "extract the single intended
character plus held props." The pipeline must be artifact-aware:

- keep one dominant foreground component and expected attached props
- reject large uncertain areas rather than preserving all salient pixels
- remove isolated background speckles and decorative blobs
- fill interior holes in face/hair/clothing even if they are pale
- detect faux checkerboard as a background pattern, but never use it as a color
  key that can delete pale subject pixels
- expose review metadata (`removedBlobCount`, `holeFillArea`,
  `edgeUncertaintyScore`, `artifactFlags`) in JSON for batch QA

## 2. Research Summary

### rembg

`rembg` is the strongest direct reference for a Python local-first background
removal tool. It supports CLI, Python library, HTTP server, Docker, CPU/GPU
installation extras, automatic model downloads, multiple model sessions, batch
folder processing, FFmpeg-style raw RGB stream processing, alpha matting, and
custom model paths. Its README documents four subcommands: `i` for single files,
`p` for folders, `s` for HTTP server, and `b` for raw RGB24 streams. It also
documents session reuse for batch performance and a model cache under
`~/.u2net/` with `U2NET_HOME` override.

Source: [danielgatis/rembg](https://github.com/danielgatis/rembg)

Relevant design patterns to adopt:

- Use a `new_session(model_name)` style factory so model loading is decoupled
  from CLI parsing.
- Reuse one model session across batch files.
- Separate model prediction from alpha matting/post-processing.
- Support `only_mask`/mask output internally even if the public default writes
  RGBA PNG.
- Keep normal runs offline once model files exist.

Relevant gaps for our use case:

- The public CLI focuses on image/video output, not a strict JSON contract.
- No first-class confidence gate for "no confident subject".
- No first-class atomic output contract.
- The default cutout path can emit a result even when the mask is poor; our
  caller needs a loud failure.
- Edge quality and no-hole behavior need a purpose-built emote profile.

### bgremover

`luizomf/bgremover` is a thin argparse CLI around `rembg`. It supports `one` and
`many`, model switching, `uv`, and local use after model download. Its value is
mainly simplicity: small structure (`cli.py`, `runners.py`, `rembg_wrapper.py`)
and a subcommand split between single and folder processing.

Source: [luizomf/bgremover](https://github.com/luizomf/bgremover)

Useful lesson:

- A narrow CLI wrapper can be much easier to maintain than cloning an entire
  background removal stack, as long as the engine contract remains independent.

Limitation:

- It inherits most `rembg` behavior and does not solve confidence, atomic
  writes, JSON schema, idempotent alpha preservation, or emote framing.

### backgroundremover

`nadermx/backgroundremover` is a broader CLI covering images, videos, folders,
stdin/stdout, HTTP server, HEIC/HEIF support, custom backgrounds, alpha matting,
GPU fallback, batch videos, worker counts, and GPU batch size. It is PyTorch
based and includes operational notes for GPU availability, corrupted downloads,
large files, and transparency codecs.

Source: [nadermx/backgroundremover](https://github.com/nadermx/backgroundremover)

Useful lessons:

- Practical CLI tools need clear hardware fallback and model download recovery.
- Batch mode should expose concurrency/batch sizing, but conservative defaults
  matter because multiprocessing can break on some platforms.
- A server mode can share the same core pipeline later, but CLI should stay the
  primary interface for automation.

Limitations for our use case:

- The CLI surface is broad and video-oriented, not a tight ffmpeg-like image
  cutout contract.
- It still recommends contrast/background hygiene in troubleshooting; our inputs
  are uncontrolled, so confidence failure and model selection matter more.

### U2Net

U2Net is the older but highly practical baseline. The official repo provides
`u2net.pth` around 176 MB and `u2netp.pth` around 4.7 MB, with U2Net originally
targeting salient object detection. The official README notes that the pretrained
model should use 320x320 input for SOD performance, and it provides a separate
human segmentation model that is more robust for people but not hair-level
accurate.

Source: [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

Implication:

- U2Net is still useful as a lightweight fallback and compatibility baseline.
- For soft illustrated characters with fine linework, U2Net alone is not the
  quality target.

### IS-Net / DIS

The DIS paper introduced Dichotomous Image Segmentation: high-accuracy
foreground/background segmentation of natural images, including high-resolution
2K/4K images and fine-grained labels. The official repo includes IS-Net and an
`isnet-general-use.pth` model for general use. `rembg` exposes
`isnet-general-use` and `isnet-anime`, the latter being especially relevant for
illustration/anime-like character cutouts.

Sources:

- [xuebinqin/DIS](https://github.com/xuebinqin/DIS)
- [Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2203.03041)

Implication:

- IS-Net is a better conceptual fit than U2Net for one-main-subject cutouts.
- `isnet-anime` is the most practical first MVP default for generated
  character/emote images if using the `rembg` model ecosystem.

### SAM / Segment Anything

SAM is promptable segmentation, not a trimap-free alpha matting model. The
official repo supports point/box prompts, automatic mask generation, ONNX export
for the lightweight mask decoder, and a CLI script for automatic mask generation.
It was trained on 11M images and 1.1B masks and has strong zero-shot segmentation
capabilities.

Source: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

Implication:

- SAM is valuable for mask proposals and fallback, especially when a bounding box
  or center point can be supplied.
- SAM is not ideal as the first default for this CLI because it produces
  class-agnostic masks and requires mask ranking/selection; it does not directly
  solve alpha edge decontamination.
- A future advanced pipeline can use SAM/SAM2 as "proposal model" and a
  matting/refinement model as "alpha model".

### BiRefNet

BiRefNet targets high-resolution dichotomous segmentation. The official paper
uses a localization module plus reconstruction module with bilateral reference
from hierarchical patches and gradient maps, focusing on fine details. The repo
has active model variants including general use, lite, dynamic resolution,
HR/matting, and ONNX conversion notes. The README reports FP16 inference and GPU
memory/runtime measurements and notes 2025 improvements to memory and
foreground-refinement performance.

Sources:

- [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- [Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://arxiv.org/abs/2401.03407)

Implication:

- BiRefNet is the strongest quality candidate for fine hair, thin structures,
  and high-resolution subject boundaries.
- It is heavier operationally than IS-Net/U2Net and needs careful packaging on
  macOS arm64.
- It should be offered early as `--model birefnet-*` / `--quality`, while the
  default can be selected by benchmark on the actual emote corpus.

### InSPyReNet / transparent-background

InSPyReNet targets high-resolution salient object detection with an image
pyramid approach. The official paper describes a strict saliency-map pyramid and
pyramid blending between low-resolution and high-resolution scales to improve
boundary accuracy. The `transparent-background` package turns this into a
practical CLI/Python API and supports image, folder, video, GUI, saliency map,
RGBA output, thresholded hard masks, `static` vs `dynamic` resize behavior, and
foreground color extraction via PyMatting for RGBA output.

Sources:

- [plemeri/InSPyReNet](https://github.com/plemeri/InSPyReNet)
- [plemeri/transparent-background](https://github.com/plemeri/transparent-background)
- [Revisiting Image Pyramid Structure for High Resolution Salient Object Detection](https://arxiv.org/abs/2209.09475)

Implication:

- InSPyReNet is a serious benchmark candidate for `profile=emote-quality`,
  especially where high-resolution boundary details matter.
- Its `static` vs `dynamic` resize note maps directly to our need for a
  deterministic default plus an opt-in sharper mode.
- Its explicit RGBA path using foreground estimation reinforces that our output
  stage must decontaminate edge colors, not simply attach an alpha channel.

### BRIA RMBG

BRIA RMBG-1.4 is IS-Net based; RMBG-2.0 is BiRefNet based and outputs a
single-channel 8-bit grayscale alpha matte. The RMBG-2.0 model card describes
the training data as licensed, high-quality, high-resolution, manually labeled
images across objects, people, animals, text, solid/non-solid backgrounds, and
single/multiple foreground objects. It is a strong quality reference, but the
published Hugging Face weights are non-commercial unless a commercial agreement
is obtained.

Sources:

- [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)
- [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

Implication:

- RMBG-2.0 is a benchmark/reference candidate, but not a default production
  dependency unless license terms are acceptable.
- Its alpha-matte output contract is the right abstraction for our
  `ModelAdapter`: models should return probability/alpha maps, while our engine
  handles cleanup, validation, and output.

### BEN2

BEN2 introduces a Confidence Guided Matting pipeline: a base model predicts the
foreground, and a refiner targets low-confidence pixels to improve matting
quality. The public repo describes strengths in hair matting, 4K processing,
object segmentation, and edge refinement, with optional foreground refinement
that improves edges at additional latency.

Source: [PramaLLC/BEN2](https://github.com/PramaLLC/BEN2/)

Implication:

- BEN2's CGM idea is highly relevant even if we do not adopt the model directly:
  our engine should compute an uncertainty band and only spend expensive
  refinement on that band.
- The default emote pipeline should expose `--refine low-confidence` semantics
  rather than a global heavy refinement pass for every pixel.

### ToonOut

ToonOut is a BiRefNet fine-tune specialized for anime-style background removal.
Its paper and model card state that general realistic-image background removal
models can underperform on stylized content, especially hair wisps, line art, and
transparency. The dataset contains 1,228 annotated anime images; the paper
reports Pixel Accuracy improving from BiRefNet's 95.3% to ToonOut's 99.5% on
their test set, with especially large gains on "action" and "emotion" subsets.
The model card lists MIT weights and CC-BY 4.0 training data.

Sources:

- [ToonOut paper](https://arxiv.org/html/2509.06839v1)
- [joelseytre/toonout](https://huggingface.co/joelseytre/toonout)

Implication:

- ToonOut is the most domain-aligned candidate for anime/emote-style generated
  characters and should be tested early.
- The reported failure categories match our use case: close-up emotions, hair
  detail, held objects, and stylized linework.
- Because it is newer and has lower adoption than IS-Net/BiRefNet/RMBG, it
  should enter as a benchmarked adapter candidate, not an untested default.

### Alpha matting / foreground decontamination

`PyMatting` implements alpha matting and foreground estimation. This matters
because a good binary/soft mask alone is not enough: edge pixels may retain the
old background color, which becomes very visible when the emote is composited on
chat backgrounds and scaled to 32px.

Source: [PyMatting docs](https://pymatting.github.io/)

Implication:

- The tool should create a trimap from the model mask and run alpha estimation
  plus foreground color estimation for edge decontamination in quality mode.
- For already-transparent input, the existing alpha should be treated as a prior,
  not discarded.

### macOS arm64 acceleration

PyTorch supports Apple Silicon acceleration through the MPS backend. ONNX Runtime
has a CoreML Execution Provider for macOS/iOS devices. These are important
because NVIDIA CUDA is not relevant on macOS arm64.

Sources:

- [Apple: Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [ONNX Runtime CoreML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)

Implication:

- The device strategy should be `auto -> mps/coreml -> cpu`, depending on the
  backend adapter.
- For MVP, CPU correctness is mandatory. Hardware acceleration can be added per
  model adapter without changing the CLI contract.

## 3. High-Level Architecture

Recommended module structure:

```text
snowy_bg_remover/
  cli.py                 # argparse/typer entrypoint, JSON stdout contract
  contracts.py           # request/result dataclasses and schema version
  engine.py              # orchestration: load, infer, validate, refine, write
  image_io.py            # decode, EXIF transpose, RGBA handling, metadata
  model_registry.py      # model aliases, manifests, adapter discovery
  model_manager.py       # download, verify, cache, offline policy
  adapters/
    base.py              # ModelAdapter protocol
    rembg_adapter.py     # ISNet/U2Net/BiRefNet via rembg/ONNX
    birefnet_torch.py    # future PyTorch/MPS native adapter
    sam_adapter.py       # future proposal/fallback adapter
  masks.py               # thresholding, components, fill holes, bbox, metrics
  alpha.py               # trimap creation, matting, foreground decontamination
  framing.py             # trim, pad, square transparent canvas
  atomic_write.py        # temp file + rename
  batch.py               # file discovery, session reuse, concurrency policy
  errors.py              # typed failures mapped to exit codes
```

Pipeline:

```text
input path
  -> decode image and normalize orientation
  -> inspect existing alpha
  -> if valid alpha and not forced: validate/pass through or refine lightly
  -> select model/profile
  -> load/reuse model session
  -> predict soft alpha at model resolution
  -> upsample soft alpha to original resolution
  -> derive high-confidence seed and low-threshold support
  -> reconstruct keep mask from seed over support
  -> confidence gate
  -> fill safe interior holes and remove unreachable blobs
  -> gate soft alpha with keep mask
  -> optional uncertainty-band alpha refinement
  -> foreground color estimation and edge decontamination
  -> apply framing flags
  -> validate final alpha is non-empty and non-degenerate
  -> atomic PNG write
  -> JSON result
```

The model adapter should only produce a mask/probability map. All business
requirements live above the adapter: alpha handling, confidence, hole filling,
framing, JSON, and atomic writes.

## 4. CLI Design

The main command should be flat for the integration contract:

```bash
cutout --input raw.png --output transparent.png
```

Batch:

```bash
cutout --input-dir input_emotes --output-dir output_emotes
cutout --glob "output/*/*.raw.png" --output-dir cutouts
```

Dry run:

```bash
cutout --input raw.png --check
```

Emote framing:

```bash
cutout --input raw.png --output emote.png --trim --pad 12% --square
```

Quality and model controls:

```bash
cutout --input raw.png --output out.png --model isnet-anime
cutout --input raw.png --output out.png --device cpu
cutout --input raw.png --output out.png --high-threshold 0.85 --low-threshold 0.05
cutout --input raw.png --output out.png --explain-dir debug/raw-001
```

Model management:

```bash
cutout models list
cutout models status --model isnet-anime
cutout models download --model isnet-anime
cutout models download --model all
```

Recommended flags:

| Flag | Purpose |
| --- | --- |
| `--input`, `--output` | Single-file mode |
| `--input-dir`, `--output-dir` | Directory batch mode |
| `--glob` | Explicit batch discovery |
| `--check` | Dry-run segmentation/confidence report, no write |
| `--profile emote|general|portrait|quality` | Future: opinionated defaults |
| `--model auto|isnet-anime|birefnet-general-lite|...` | Model selection |
| `--device auto|cpu|coreml|cuda` | Backend selection |
| `--threads <n>` | Optional ONNX Runtime thread control |
| `--allow-download` | Explicit network opt-in; normal runs are offline |
| `--trim`, `--pad <px|%>`, `--square` | Emote framing |
| `--high-threshold`, `--low-threshold` | Hysteresis topology controls |
| `--no-decontaminate`, `--decontaminate-radius` | Edge RGB cleanup controls |
| `--explain-dir` | Debug artifacts for rejected/accepted images |
| `--jsonl` | Batch emits one JSON object per line |
| `--fail-fast` | Stop batch on first failure |

Default stdout shape for single image:

```json
{
  "schemaVersion": 1,
  "ok": true,
  "reason": null,
  "input": "raw.png",
  "output": "transparent.png",
  "width": 1024,
  "height": 1024,
  "bbox": [143, 58, 761, 908],
  "subjectCoverage": 0.42,
  "hadAlpha": false,
  "model": "isnet-anime",
  "device": "cpu",
  "elapsedMs": 1840,
  "message": "ok"
}
```

For batch, default should be a single summary JSON object with `items`. `--jsonl`
can be added for streaming agent pipelines.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `2` | CLI usage/config error |
| `10` | Input missing or unsupported |
| `11` | Decode/read failure |
| `12` | Model missing and downloads disabled/offline |
| `13` | Model hash/verification failure |
| `20` | `no_confident_subject` |
| `21` | Degenerate/empty alpha after processing |
| `30` | Output write/atomic rename failure |
| `40` | Internal error |

## 5. Model Strategy Comparison

| Model family | Strengths | Weaknesses | Best use |
| --- | --- | --- | --- |
| U2Net | Mature, small ecosystem, ONNX-friendly, fast enough, `u2netp` tiny | Older salient-object model; weaker fine details; can confuse pale subject/background | Compatibility fallback, low-resource mode |
| IS-Net | DIS-oriented, better high-accuracy subject segmentation; `isnet-anime` is relevant to generated characters | Still not a true matting model; quality depends on model variant | MVP default for `profile=emote` if benchmarking confirms |
| SAM/SAM2 | Excellent promptable mask proposals; point/box prompts and automatic mask generator; strong zero-shot | Not alpha matting; needs prompt/mask ranking; heavier; can select wrong object/background blob | Advanced fallback/proposal stage, user/GUI-assisted refinement |
| BiRefNet | Best quality candidate for fine detail and high-res DIS/matting; multiple general/lite/HR variants | Heavier dependencies and memory; packaging on macOS arm64 needs care | Quality mode, future default after corpus benchmark |
| InSPyReNet | High-resolution saliency with pyramid blending; practical `transparent-background` CLI; MIT | Saliency-object framing, not emote-specific; small-image stability notes need testing | Quality benchmark, possible high-res/detail adapter |
| RMBG-2.0 | Strong BiRefNet-based alpha matte, broad licensed training dataset | HF weights are non-commercial without agreement; `trust_remote_code` workflow needs production hardening | Benchmark/reference or licensed production adapter |
| BEN2 | Confidence-guided refinement concept; hair/edge focus; ONNX available | Base/open model vs commercial model distinction; less proven for anime/emotes | Design reference for low-confidence refinement and benchmark candidate |
| ToonOut | Anime/illustration-specialized BiRefNet fine-tune; MIT weights; strong reported anime results | Newer, lower adoption; needs local packaging and corpus validation | Top candidate for `profile=emote-quality` and potential default after tests |

Practical recommendation:

- MVP default profile: `emote` using `isnet-anime` first if the `rembg` adapter is
  used.
- Add `birefnet-general-lite`, `InSPyReNet`, and ToonOut as early benchmark
  candidates.
- Add a benchmark gate before switching default from IS-Net to BiRefNet/ToonOut.
- Keep U2Net as fallback/compatibility, not the quality target.
- Treat SAM as a future proposal/fallback model, not the default alpha model.
- Treat RMBG-2.0 as reference/benchmark unless licensing is explicitly cleared.

## 6. Confidence and Robustness Design

The tool should fail loudly rather than emit bad transparent PNGs.

Confidence inputs:

- `foregroundPixelCount`
- `subjectCoverage = foregroundPixelCount / (width * height)`
- largest connected component ratio
- bbox dimensions and area
- bbox touches image border ratio
- alpha mean/median inside bbox
- low-confidence edge band size
- agreement between existing alpha and predicted mask, when input has alpha
- optional two-model agreement for hard cases
- `artifactFlags` for likely faux checkerboard, edge glow, shadow, scattered
  speckles, large detached foreground-like objects, or unusual border contact
- `edgeUncertaintyScore`: fraction of pixels in the trimap unknown band
- `removedBlobCount` and total removed blob area
- `holeFillArea` and number of filled interior holes

Default `no_confident_subject` triggers:

- foreground coverage is too small or too large
- no connected component survives cleanup
- largest component is not dominant
- bbox is near full canvas with low edge confidence
- final alpha is empty or almost fully transparent
- model output is flat/low contrast
- a detached secondary component is too large to safely discard or keep
- uncertainty band is too large relative to subject bbox
- existing alpha and predicted mask strongly disagree, unless `--force` is set

Approximate initial thresholds for emotes:

- `minCoverage`: `0.03`
- `maxCoverage`: `0.92`
- `minLargestComponentRatio`: `0.70`
- `minMeanAlphaInBBox`: `0.45`
- `minBBoxSizePx`: `16`
- `maxTransparentOutputCoverage`: fail if alpha coverage `< 1%`
- `maxDetachedComponentCoverage`: `0.08`
- `maxUnknownBandCoverage`: `0.25` inside bbox before refinement

These should be configurable and then tuned against real generated samples.

## 6.1 Artifact-Aware Emote Pipeline

The default `profile=emote` pipeline should add an artifact-control stage between
raw model output and alpha refinement:

```text
soft alpha probability map
  -> high-confidence seed proposal
  -> low-threshold support proposal
  -> hysteresis / morphological reconstruction
  -> main subject selection
  -> attached prop preservation
  -> detached blob removal
  -> interior hole fill
  -> edge uncertainty band
  -> optional low-confidence refinement
  -> trimap/alpha refinement
```

Main subject selection:

- prefer the largest confident component near the visual center
- allow attached props if connected to the main component through a narrow bridge
  or close high-confidence edge band
- reject detached blobs unless they are inside/overlapping the main bbox and
  pass size/proximity heuristics
- expose `keptComponentCount` and `removedComponentCount`

Faux checkerboard handling:

- detect repeated alternating grid-like background patterns as an artifact flag
- use the learned model output to segment the subject; do not key by checker
  color
- if checkerboard pixels leak into the mask around transparent-looking regions,
  use local texture periodicity plus low model confidence to demote them in the
  uncertainty band

Glow/shadow handling:

- default emote mode should drop cast shadows and background glow unless they
  are tightly attached and inside the subject silhouette
- `--keep-shadow` can preserve soft contact shadows for non-emote use, but it
  should be off for 32px chat emotes
- Edge decontamination should remove old background color from semi-transparent
  boundary pixels

Interior detail handling:

- holes fully enclosed by the main subject are filled by default if below a
  configurable area threshold
- large semantic holes are suspicious for emotes and should become review flags
  instead of being blindly filled
- pale subject regions are protected by model confidence and connectivity, not
  by color

Protected core:

- pixels above `high_threshold` inside the reconstructed main subject are a
  protected core
- erosion, decontamination, shadow suppression, and blob cleanup must not remove
  this core
- cleanup gates the soft alpha around the core; it does not replace the kept
  region with a hard mask

## 7. Alpha and Image Pipeline

Rules:

- Always preserve original image resolution unless `--trim`, `--pad`, or
  `--square` changes the canvas.
- Decode with EXIF orientation correction.
- Work internally as RGB/RGBA float arrays, but write PNG as 8-bit RGBA first.
- Do not chroma-key.
- Convert model probability map to mask at original resolution.
- Keep the main component; remove stray background blobs.
- Fill interior holes inside the main subject to protect pale face/hair/clothes.
- Preserve thin details by avoiding aggressive erosion.
- Build a trimap from high-confidence foreground/background plus unknown edge
  band.
- Run alpha matting/foreground estimation in quality mode to reduce halos.
- Clamp and validate alpha before writing.
- Use alpha-aware RGB reconstruction for semi-transparent edge pixels so old
  background colors do not remain in the foreground RGB channels.
- Composite QA should test output against dark, light, saturated, and checker
  backgrounds to catch halos.
- If a 32px derivative is generated, downscale in linear-light premultiplied
  alpha space, then unpremultiply and convert back to sRGB. Naive straight-RGBA
  resizing can create the exact halos the tool is meant to prevent.

Existing alpha handling:

- If the input has nontrivial alpha and the alpha bbox is plausible, default to
  preserving it.
- If `--refine-alpha` is enabled, combine alpha and model mask as priors:
  existing alpha protects subject pixels; model mask removes accidental opaque
  background.
- Never turn an already-transparent good subject into an empty output.

Framing:

- `--trim`: crop to final alpha bbox after cleanup.
- `--pad 12%`: pad relative to bbox max dimension, or fixed px.
- `--square`: expand to square transparent canvas centered on subject.
- Framing runs after alpha validation.

32px derivative policy:

- The full-resolution transparent PNG should remain the master artifact.
- A future `--emit-size 32` or `--preview-size 32` can emit the product-size
  derivative.
- The benchmark should rank models primarily by 32px composite quality, not only
  native-resolution mask metrics.

## 8. Plugin and Extensibility Design

Use a small adapter protocol:

```python
class ModelAdapter(Protocol):
    spec: ModelSpec

    def load(self, device: DeviceSpec) -> None: ...
    def predict_mask(self, image: ImageInput, options: InferenceOptions) -> MaskResult: ...
    def unload(self) -> None: ...
```

`MaskResult`:

```python
{
  "mask": "float32 ndarray HxW, values 0..1",
  "nativeSize": [1024, 1024],
  "modelName": "isnet-anime",
  "rawScores": {},
  "elapsedMs": 1234
}
```

`ModelSpec`:

```json
{
  "name": "isnet-anime",
  "aliases": ["emote-v1"],
  "backend": "rembg-onnx",
  "task": "character-cutout",
  "cacheFiles": [
    {
      "filename": "isnet-anime.onnx",
      "sha256": "...",
      "url": "..."
    }
  ],
  "license": "upstream",
  "defaultThreshold": 0.5,
  "supportsBatch": false,
  "supportsDevice": ["cpu", "coreml"]
}
```

Discovery:

- Built-in adapters for stable production models.
- Optional Python entry points later:
  `snowy_bg_remover.models`.
- Plugin model manifests can live under:
  `~/Library/Application Support/snowy-bg-remover/plugins`.
- Do not allow arbitrary remote code execution during normal model load.
  If Hugging Face `trust_remote_code` is needed during experimentation, pin and
  vendor the adapter before production.

## 9. Production Considerations

### Offline-first model management

- Normal `cutout` runs should not download.
- Missing model returns exit `12` unless `--allow-download` is explicit.
- `cutout models download --model all` prefetches all built-in models.
- Store model files in a platform-appropriate cache directory:
  `~/Library/Caches/snowy-bg-remover/models` on macOS.
- Override with `SNOWY_CUTOUT_MODEL_CACHE`.
- Maintain per-model metadata with name, hash, source URL, license, and
  download date.
- Verify hash before inference.
- Use file locks to prevent concurrent downloads from corrupting a model file.

### GPU/CPU fallback

- Required baseline: CPU.
- PyTorch adapters: prefer `mps` on macOS arm64 if available, else CPU.
- ONNX adapters: prefer CoreML EP if packaged/tested, else CPU.
- Log selected device to stderr and include it in JSON.
- If GPU/MPS inference fails, retry once on CPU unless `--device` explicitly
  forced a device.

### Batch processing

- Build the file list first for deterministic order.
- Reuse one loaded session per model.
- Default concurrency should be conservative:
  - GPU/MPS/CoreML: serial model inference, concurrent decode/write only later.
  - CPU: small worker pool only after verifying memory use.
- Batch output should continue by default and summarize per-file failures.
- `--fail-fast` stops on first failure.
- Atomic output applies per image.

### Atomic writes

Algorithm:

1. Write PNG to `<output>.tmp-<pid>-<random>.png`.
2. Flush and fsync best effort.
3. Validate by reopening the temp image and checking alpha non-empty.
4. Rename/replace atomically into final output path.
5. On failure, delete temp file and return a write/internal error.

### Determinism

- Fix model mode to eval/inference.
- Avoid random augmentations.
- Keep thresholds explicit and included in debug metadata.
- Do not use time-dependent output filenames except temp files.
- Do not print progress bars to stdout.

### Testing

Golden fixture set:

- Already-transparent character
- White hair/outfit on white background
- Pastel character on gradient background
- Character over faux checkerboard
- Busy scenery background
- Thin hair/twin-tails/ears
- Held props, signs, hearts, megaphone
- AI-generated background speckles/dust
- faux checkerboard rendered as real pixels
- background glow/rim light/shadow around character
- detached decorative hearts/stars that should be dropped
- semitransparent accessories or soft watercolor edges
- No confident subject / empty canvas
- Multiple stray blobs behind subject
- Very small subject

Assertions:

- JSON schema and exit codes
- Atomic output behavior under simulated crash/write failure
- No fully transparent success
- bbox and coverage within expected ranges
- alpha channel exists and is non-flat
- no large interior holes
- deterministic repeated output hash for CPU path, or bounded tolerance for GPU
- batch order stable

Benchmark matrix:

| Candidate | Why test | Success criteria |
| --- | --- | --- |
| `isnet-anime` | Practical MVP baseline available through `rembg` | Low latency, acceptable cutouts on most corpus samples |
| `birefnet-general-lite` | Quality improvement with lighter cost | Better hair/linework than IS-Net without large latency hit |
| ToonOut | Closest domain match for anime/illustration | Best boundary quality and fewer action/emotion failures |
| InSPyReNet | Strong high-resolution saliency/boundary approach | Best or second-best edge stability on 1024px generated icons |
| RMBG-2.0 | Strong licensed-data reference | Reference score only unless licensing is cleared |
| BEN2 base | Confidence-guided matting reference | Useful edge/low-confidence behavior and acceptable packaging |

Recommended quality metrics:

- human-ranked visual score at 32px, 64px, and original size
- Boundary IoU or boundary F-score around the subject edge
- alpha coverage stability across repeated runs
- false kept background blob area
- false removed subject detail area
- halo score by compositing on black/white/saturated backgrounds
- failure rate with `no_confident_subject`
- median and p95 latency on macOS arm64 CPU and MPS/CoreML when available

## 10. MVP Recommendation

Build a narrow but strict CLI first.

MVP scope:

- Python 3.11/3.12 package.
- `cutout` command with single, batch, `--check`, JSON stdout, stderr logs.
- Model manager with explicit `models download`, hash verification, offline
  default for normal runs.
- First adapter: `rembg`/ONNX-compatible `isnet-anime`.
- Early quality adapters/options: `birefnet-general-lite`, ToonOut, and
  InSPyReNet if installation and runtime are acceptable on macOS arm64.
- Core post-processing independent of model:
  connected components, confidence gate, hole filling, alpha validation,
  artifact flags, framing, atomic writes.
- Alpha refinement:
  start with high-quality mask smoothing and foreground color decontamination;
  add PyMatting trimap refinement behind `--quality` or default `profile=emote`
  if latency is acceptable.
- Artifact-aware cleanup:
  remove AI-generated speckles and detached decorative blobs, detect faux
  checkerboard as an artifact, drop default shadows/glows for emotes, and report
  all major cleanup decisions in JSON.
- No GUI, no HTTP server, no MCP server in MVP. The stable JSON CLI contract is
  the integration surface for OpenClaw and future MCP wrapping.

MVP default command:

```bash
cutout --input input.png --output output.png --trim --pad 8% --square
```

Expected internal default:

```text
profile=emote
model=auto -> isnet-anime
offline=true
edgeDecontamination=true
confidenceGate=true
keepMainComponent=true
fillInteriorHoles=true
trim=false
square=false
jsonStdout=true
```

Quality roadmap:

1. Benchmark `isnet-anime`, `birefnet-general-lite`, `birefnet-general`, and a
   BiRefNet matting variant on real emote-wall outputs.
2. Add ToonOut and InSPyReNet to the same benchmark before locking the default.
3. Switch `profile=emote` default only after metrics show better quality at
   acceptable latency.
4. Add SAM/SAM2 proposal fallback only if confidence failures remain common.
5. Add MCP wrapper around the same engine after CLI schema stabilizes.
6. Add GUI as a thin front-end over the same library/engine, not a separate
   processing path.

## 11. Key Design Decision

Do not build a "best effort" remover. Build a strict cutout compiler:

- It either produces a validated transparent PNG with useful alpha, or it fails
  with a typed machine-readable reason.
- Model choice is pluggable, but the product quality comes from the pipeline:
  validation, cleanup, matting, decontamination, framing, atomic IO, and stable
  automation contracts.
