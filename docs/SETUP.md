# Setup and Operations Guide

This guide is the recommended path for using `snowy-bg-remover` on a new
machine.

The CLI is local-first: normal cutout runs do not use the network. The only
network-dependent step is provisioning the model cache with
`cutout models download`.

## Requirements

- Python 3.11 or 3.12
- Git
- Network access for initial model download
- Disk space for the local model cache
  - `isnet-anime`: small ONNX default model
  - `birefnet-general-lite`: ONNX comparison model
  - `toonout`: high-quality anime model, about 846 MB

The `--quality` path requires PyTorch and Transformers. Install the project with
the `quality` extra when you want ToonOut.

## Install on Windows PowerShell

```powershell
git clone https://github.com/snowyukitty/snowy-bg-remover.git
cd snowy-bg-remover

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[quality]"
```

If PowerShell blocks activation scripts, either allow local scripts for the
current user or run commands through the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m snowy_bg_remover.cli --help
```

## Install on macOS or Linux

```bash
git clone https://github.com/snowyukitty/snowy-bg-remover.git
cd snowy-bg-remover

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[quality]"
```

## Provision Models

Download the default fast model and the quality model:

```bash
python -m snowy_bg_remover.cli models download --model isnet-anime
python -m snowy_bg_remover.cli models download --model toonout
python -m snowy_bg_remover.cli models status --model all
```

Optional comparison model:

```bash
python -m snowy_bg_remover.cli models download --model birefnet-general-lite
```

Model files are verified by SHA256. ToonOut also primes the pinned BiRefNet
runtime code during `models download`, so missing `quality` dependencies or
runtime-cache problems fail during provisioning instead of during a later
offline publish job.

Default cache locations:

| OS | Cache |
| --- | --- |
| Windows | `%LOCALAPPDATA%\snowy-bg-remover\models` |
| macOS | `~/Library/Caches/snowy-bg-remover/models` |
| Linux | `$XDG_CACHE_HOME/snowy-bg-remover/models` or `~/.cache/snowy-bg-remover/models` |

Use `SNOWY_CUTOUT_MODEL_CACHE` or `--model-cache` to override the cache path.

## Quick Smoke Tests

Dry-run detectability without writing output:

```bash
python -m snowy_bg_remover.cli --input path/to/raw.png --check
```

Fast default cutout:

```bash
python -m snowy_bg_remover.cli \
  --input path/to/raw.png \
  --output path/to/cutout.png \
  --trim \
  --pad 8% \
  --square
```

High-quality anime/emote cutout:

```bash
python -m snowy_bg_remover.cli \
  --input path/to/raw.png \
  --output path/to/cutout.png \
  --quality \
  --device auto \
  --trim \
  --pad 8% \
  --square
```

Derived 128px review artifact:

```bash
python -m snowy_bg_remover.cli \
  --input path/to/raw.png \
  --output path/to/cutout-128.png \
  --quality \
  --device auto \
  --trim \
  --pad 8% \
  --square \
  --emit-size 128
```

Batch processing:

```bash
python -m snowy_bg_remover.cli \
  --input-dir input \
  --output-dir output \
  --quality \
  --device auto \
  --trim \
  --pad 8% \
  --square \
  --jsonl \
  --fail-fast
```

## Already-Transparent Inputs

By default, meaningful input alpha is treated as the source of truth. If an
input PNG is already transparent and you do not pass `--quality` or
`--force-model`, the engine uses the existing alpha, runs validation/framing, and
does not re-segment the image.

Use this for clean transparent masters:

```bash
python -m snowy_bg_remover.cli --input already-cutout.png --output framed.png --trim --pad 8% --square
```

`--quality` intentionally forces model inference and alpha refinement. This is
useful for raw AI images or damaged alpha, but it is not the safest default for a
known-good transparent master. For diagnosis, add `--no-alpha-refine`.

## Recommended Source-Image Background

The tool does not rely on chroma keying and should not need a special key color.
For generated emote sources, the best prompt policy is a simple, non-semantic
background:

```text
plain solid matte medium-gray background, no scenery, no texture, no pattern,
no checkerboard, no glow, no cast shadow
```

Avoid asking for blue/green chroma-key backgrounds unless the character palette
is guaranteed not to contain those colors. Bright saturated backgrounds can
increase edge color contamination and can leak into semi-transparent hair or
soft linework. Medium gray is usually safer because it is less likely to become
a semantic object and creates less visible fringe after matting.

## Output Contract

Every command writes machine-readable JSON to stdout. Logs and tracebacks go to
stderr. Important fields include:

```json
{
  "ok": true,
  "input": "raw.png",
  "output": "cutout.png",
  "width": 1024,
  "height": 1024,
  "bbox": [210, 96, 816, 930],
  "subjectCoverage": 0.42,
  "hadAlpha": false,
  "elapsedMs": 1830,
  "model": "toonout",
  "modelSha256": "...",
  "message": "ok"
}
```

Exit code `0` means a confident usable cutout. Nonzero means the caller should
keep the original input and surface the failure.

## Troubleshooting

`model_unavailable` or exit code `12`:

```bash
python -m snowy_bg_remover.cli models download --model isnet-anime
python -m snowy_bg_remover.cli models download --model toonout
```

`toonout requires quality dependencies`:

```bash
python -m pip install -e ".[quality]"
```

GPU or accelerator issues:

```bash
python -m snowy_bg_remover.cli --input raw.png --output cutout.png --quality --device cpu
```

Use `--device auto` for normal quality runs. Use `--device cpu` for the most
portable behavior. CUDA and MPS depend on the local PyTorch installation.

Hard-to-debug rejection:

```bash
python -m snowy_bg_remover.cli \
  --input raw.png \
  --output cutout.png \
  --quality \
  --explain-dir explain/raw
```

The explain directory contains source alpha, final alpha, keep mask, protected
core, and the result JSON.

## Development Verification

Before pushing changes:

```bash
python -m pytest -q
python -m compileall -q src tests
python -m snowy_bg_remover.cli models status --model all
git diff --check
```
