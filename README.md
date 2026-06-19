# snowy-bg-remover

Local-first background removal CLI for Snowy Style Atlas emote cutouts.

Current status: usable local MVP with a dedicated anime/emote quality path. The
pure library engine, CLI wrapper, JSON result schema, atomic PNG writes,
alpha-aware topology utilities, framing, offline model cache, ONNX adapter, and
PyTorch BiRefNet adapter are in place. The fast default model is `isnet-anime`.
The high-quality anime model is `toonout`, pinned by SHA256 and loaded locally
after provisioning.

Design principle: keep soft alpha as the source of truth. Binary masks are used
only for topology decisions such as connected support, detached artifact removal,
and safe interior hole filling.

```bash
python -m pip install -e .
python -m pip install -e ".[quality]"
python -m snowy_bg_remover.cli models download --model isnet-anime
python -m snowy_bg_remover.cli models download --model toonout
cutout --input raw.png --output transparent.png
cutout --input raw.png --output transparent.png --quality --device auto
cutout --input raw.png --check
cutout --input-dir input --output-dir output
cutout --glob "input/**/*.png" --output-dir output --jsonl --fail-fast
```

Normal `cutout` runs are offline by default. If a model is missing, the command
fails with JSON on stdout and exit code `12`; use `cutout models download` during
provisioning. `--allow-download` is available for explicit one-shot setup, but
should not be used in deterministic publish jobs.

If the editable install script directory is not on PATH, use:

```bash
python -m snowy_bg_remover.cli --input raw.png --check
```

## Installation

For a complete new-machine walkthrough, use [docs/SETUP.md](docs/SETUP.md).

```bash
python -m pip install -e .
python -m pip install -e ".[quality]"  # required for --quality / toonout
python -m snowy_bg_remover.cli models download --model isnet-anime
python -m snowy_bg_remover.cli models download --model toonout
```

The model file is not committed to this repository. It is downloaded into the
platform cache directory and verified by SHA256 before inference. Set
`SNOWY_CUTOUT_MODEL_CACHE` or pass `--model-cache` to override the cache path.

## Emote-Oriented Usage

For fast publish-style cutouts with transparent framing:

```bash
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output cutout.png \
  --trim \
  --pad 8% \
  --square
```

For the higher-quality anime/emote path:

```bash
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output cutout.png \
  --quality \
  --device auto \
  --trim \
  --pad 8% \
  --square
```

`--quality` currently selects `toonout`, forces model inference even when the
input already has alpha, enables bounded closed-form alpha refinement, and then
runs the same topology, foreground estimation, edge decontamination, and framing
stages as the fast path. This is slower and heavier than the ONNX path, but it
is materially better for anime hair, ears, linework, and AI-generated
pale-on-pale images.

For already-transparent clean masters, omit `--quality` and `--force-model`.
Meaningful input alpha is then treated as the source of truth, so the command can
validate, trim, pad, and square the image without re-segmenting it. Use
`--quality` on transparent input only when the alpha is damaged or you
intentionally want to re-run the model.

Useful quality knobs:

```bash
# Refine the trimap boundary at a larger working resolution.
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output cutout.png \
  --quality \
  --alpha-refine-size 768

# Disable alpha refinement for diagnosis or maximum reproducibility.
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output cutout.png \
  --quality \
  --no-alpha-refine

# Ignore very faint residual alpha when computing the trim/framing bbox.
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output cutout.png \
  --quality \
  --bbox-threshold 0.16
```

For a 32px derived emote:

```bash
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output emote.png \
  --trim \
  --pad 8% \
  --square \
  --emit-size 32
```

For 128px review output:

```bash
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output emote-128.png \
  --quality \
  --device auto \
  --trim \
  --pad 8% \
  --square \
  --emit-size 128
```

For batch processing:

```bash
python -m snowy_bg_remover.cli \
  --input-dir input \
  --output-dir output \
  --trim \
  --pad 8% \
  --square \
  --jsonl \
  --fail-fast
```

## Source Image Background Guidance

The tool uses learned semantic segmentation/matting, not chroma keying. A pure
blue or green background is not automatically better than white or black, and it
can create saturated color spill in semi-transparent hair, ears, soft linework,
or pale clothing.

For generated emote sources, prefer a simple non-semantic background:

```text
plain solid matte medium-gray background, no scenery, no texture, no pattern,
no checkerboard, no glow, no cast shadow
```

Medium gray is usually the safest generation backdrop because it avoids
high-contrast white/black edge ambiguity while producing less visible color
contamination than saturated chroma-key colors.

## Implemented contract

- single-image and batch CLI modes
- stdout JSON for success and failure
- deterministic nonzero exit codes for failure reasons
- `--check` dry-run without writing
- atomic PNG output writes through temp-file-and-rename
- true RGBA output with source alpha preserved/refined
- ONNX Runtime adapter for `isnet-anime` and BiRefNet general-lite
- PyTorch BiRefNet adapter for `toonout`
- offline model cache with atomic download and SHA256 verification
- soft-alpha-native topology: high-confidence seed, low-threshold support,
  morphological reconstruction, detached blob removal, and bounded interior
  hole repair
- largest high-confidence core selection with multi-core confidence rejection
- edge RGB decontamination via PyMatting foreground estimation, with nearest
  opaque foreground color bleed fallback
- uniform-background color suppression (`--no-bg-suppress` to disable): on the
  model path, removes flat backdrop pixels the matte leaves trapped in concave
  hair/ear gaps. Safe by construction — only reduces alpha for pixels that are
  both background-colored and reachable from the image border through other
  background-colored pixels, so interior neutral features (eyes, pearls, silver)
  are never touched. No-op unless a single dominant uniform border background is
  detected; auto-aborts if it would erase the whole subject.
- enclosed-pocket removal inside background suppression: the flat generated
  backdrop trapped between hair strands (not reachable from the border) is
  keyed globally by tight color match + near-background chroma, so neutral
  hair-gap backdrop is removed while tinted features (eyes, pearls) are kept.
- optional `--edge-contract N` matte contraction: erodes the silhouette inward by
  N px and re-softens it, removing the thin opaque pale halo left when pale hair
  fades into a low-contrast backdrop. Works on the model path and on already-cut
  RGBA input, so existing cutouts can be cleaned without re-segmenting.
- bounded closed-form alpha refinement for `--quality`, driven by a generated
  trimap around the model boundary and protected foreground core
- `--bbox-threshold` to keep faint outer haze from expanding trim/framing boxes
- optional `--explain-dir` artifacts: source alpha, final alpha, keep mask,
  protected core, and result JSON
- optional `--trim`, `--pad`, `--square`, and `--emit-size`
- batch model session reuse, `--fail-fast`, and optional `--threads`
- linear-light premultiplied RGBA resize for derived small outputs

## Model Strategy

| Model | Backend | Use |
| --- | --- | --- |
| `isnet-anime` | ONNX Runtime | Fast default for batch processing and low dependency footprint. |
| `toonout` | PyTorch BiRefNet | Quality path for anime/emote cutouts, especially hair, ears, and soft linework. |
| `birefnet-general-lite` | ONNX Runtime | Registered comparison model; not the default for anime because it can mis-segment stylized emotes. |

The tool intentionally keeps semantic foreground effects and held props when the
model treats them as foreground. This is better for emote expressiveness, but a
future `--drop-effects` mode can be added if the publish policy should remove
floating expression marks such as sparkles or emphasis strokes.

InSPyReNet / `transparent-background` was evaluated as a candidate quality
backend. It can produce finer natural-image edges, but on the current emote
samples it also removes or weakens semantic expression effects and held elements,
so it is not the default. It remains a good future optional model backend.

Key upstream projects used as references:

- rembg model registry and ONNX release packaging: https://github.com/danielgatis/rembg
- BiRefNet: https://github.com/ZhengPeng7/BiRefNet
- ToonOut anime BiRefNet fine-tune: https://huggingface.co/joelseytre/toonout
- PyMatting foreground estimation: https://github.com/pymatting/pymatting
- transparent-background / InSPyReNet pipeline reference: https://github.com/plemeri/transparent-background

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | success |
| `2` | CLI usage error |
| `10` | input file missing |
| `11` | image decode failure |
| `12` | model unavailable or missing from cache |
| `20` | no confident subject / low confidence |
| `21` | degenerate alpha |
| `30` | output write failure |
| `40` | internal error |

## Next implementation step

The next quality step is quantitative benchmarking and a stricter policy mode:

1. Build a real emote-wall corpus and synthetic-composite benchmark set.
2. Add a `--drop-effects` policy for floating non-character decorations.
3. Add InSPyReNet / transparent-background as an optional backend.
4. Calibrate confidence thresholds against false-accept rate at 128px and 32px.
