# snowy-bg-remover

Local-first background removal CLI for Snowy Style Atlas emote cutouts.

Current status: usable local MVP. The pure library engine, CLI wrapper, JSON
result schema, atomic PNG writes, alpha-aware topology utilities, framing,
offline model cache, and first ONNX adapter are in place. The default model is
`isnet-anime`, pinned by SHA256 and loaded locally after provisioning.

Design principle: keep soft alpha as the source of truth. Binary masks are used
only for topology decisions such as connected support, detached artifact removal,
and safe interior hole filling.

```bash
python -m pip install -e .
cutout models download --model isnet-anime
cutout --input raw.png --output transparent.png
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
python -m snowy_bg_remover.cli models download --model isnet-anime
```

The model file is not committed to this repository. It is downloaded into the
platform cache directory and verified by SHA256 before inference. Set
`SNOWY_CUTOUT_MODEL_CACHE` or pass `--model-cache` to override the cache path.

## Emote-Oriented Usage

For publish-style cutouts with transparent framing:

```bash
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output cutout.png \
  --trim \
  --pad 8% \
  --square
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
- ONNX Runtime adapter for `isnet-anime`
- offline model cache with atomic download and SHA256 verification
- soft-alpha-native topology: high-confidence seed, low-threshold support,
  morphological reconstruction, detached blob removal, and bounded interior
  hole repair
- largest high-confidence core selection with multi-core confidence rejection
- edge RGB decontamination via nearest opaque foreground color bleed
- optional `--explain-dir` artifacts: source alpha, final alpha, keep mask,
  protected core, and result JSON
- optional `--trim`, `--pad`, `--square`, and `--emit-size`
- batch model session reuse, `--fail-fast`, and optional `--threads`
- linear-light premultiplied RGBA resize for derived small outputs

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

The next quality step is benchmarking and optional second-adapter comparison:

1. Build a real emote-wall corpus and synthetic-composite benchmark set.
2. Add BiRefNet or BiRefNet-matting for higher quality soft alpha.
3. Add InSPyReNet / transparent-background as a soft-edge comparison point.
4. Calibrate confidence thresholds against false-accept rate at 32px.
