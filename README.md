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
cutout models download --model isnet-anime
cutout models download --model toonout
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
input already has alpha, and then runs the same topology, foreground estimation,
edge decontamination, and framing stages as the fast path. This is slower and
heavier than the ONNX path, but it is materially better for anime hair, ears,
linework, and AI-generated pale-on-pale images.

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
3. Add InSPyReNet / transparent-background as a soft-edge comparison point.
4. Calibrate confidence thresholds against false-accept rate at 32px.
