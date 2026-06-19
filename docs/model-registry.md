# Model Registry

This file records the built-in model allowlist for `cutout`.

Normal inference runs must be offline and must only load verified files from the
local cache. New models should be added here and in
`src/snowy_bg_remover/model_specs.py` with a pinned hash before they are used in
the publish pipeline.

## Built-In Models

| Model | Status | License | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `isnet-anime` | default | Apache-2.0 | ONNX Runtime | Anime/illustration baseline from SkyTNT anime-segmentation, using the rembg ONNX export and rembg-compatible preprocessing. |

## `isnet-anime`

- Model file: `isnet-anime.onnx`
- URL: `https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx`
- SHA256: `f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99`
- Cache command: `cutout models download --model isnet-anime`
- Default cache:
  - macOS: `~/Library/Caches/snowy-bg-remover/models`
  - Windows: `%LOCALAPPDATA%/snowy-bg-remover/models`
  - Linux: `$XDG_CACHE_HOME/snowy-bg-remover/models` or `~/.cache/snowy-bg-remover/models`
- Cache override: `SNOWY_CUTOUT_MODEL_CACHE`

Implementation notes:

- Input size: `1024x1024`
- Mean: `(0.485, 0.456, 0.406)`
- Std: `(1.0, 1.0, 1.0)`
- Output postprocess: min/max normalize, resize float alpha to original image
  size, then feed the soft alpha into topology cleanup.

References:

- SkyTNT anime-segmentation license: Apache-2.0.
- SkyTNT anime-segmentation README lists ISNet support, 1024px training/export
  path, and anime character dataset context.
- rembg `dis_anime.py` provides the public `isnet-anime` model name, ONNX URL,
  checksum, and preprocessing parameters.
- ONNX Runtime provides CPU execution and optional platform execution providers
  such as CoreML.

## Admission Rules For New Models

Before a model can become a built-in option:

1. License must be compatible with commercial/publish use.
2. Model file must be pinned by SHA256, not just a mutable branch or latest URL.
3. Normal `cutout` inference must load from local cache only.
4. Adapter must return a continuous alpha/probability map; threshold-first
   adapters are not acceptable.
5. Tests must cover missing cache, hash failure, successful adapter invocation,
   and confidence behavior on low-quality masks.
6. Benchmark results must include native-resolution alpha metrics and 32px
   composited emote metrics.
