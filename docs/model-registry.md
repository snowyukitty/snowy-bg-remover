# Model Registry

This file records the built-in model allowlist for `cutout`.

Normal inference runs must be offline and must only load verified files from the
local cache. New models should be added here and in
`src/snowy_bg_remover/model_specs.py` with a pinned hash before they are used in
the publish pipeline.

## Built-In Models

| Model | Status | License | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `isnet-anime` | fast default | Apache-2.0 | ONNX Runtime | Anime/illustration baseline from SkyTNT anime-segmentation, using the rembg ONNX export and rembg-compatible preprocessing. |
| `toonout` | quality default | MIT | PyTorch + pinned BiRefNet runtime | Anime/illustration-specialized BiRefNet fine-tune. Best current option for Snowy emote hair, ears, soft linework, and pale-on-pale AI images. |
| `birefnet-general-lite` | comparison | MIT | ONNX Runtime | Lightweight BiRefNet general model. Useful for benchmarking and fallback experiments, but not the anime default. |

Aliases:

| Alias | Resolves To |
| --- | --- |
| `auto` | `isnet-anime` |
| `emote` | `isnet-anime` |
| `quality` | `toonout` |

## `isnet-anime`

- Model file: `isnet-anime.onnx`
- URL: `https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx`
- SHA256: `f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99`
- Cache command: `cutout models download --model isnet-anime`
- Backend: ONNX Runtime
- Input size: `1024x1024`
- Mean: `(0.485, 0.456, 0.406)`
- Std: `(1.0, 1.0, 1.0)`
- Output postprocess: min/max normalize, resize float alpha to original image
  size, then feed the soft alpha into topology cleanup.

## `toonout`

- Model file: `birefnet_finetuned_toonout.pth`
- URL: `https://huggingface.co/joelseytre/toonout/resolve/cbf720eca394edcde66b861a8a8c20fbabe9c748/birefnet_finetuned_toonout.pth`
- SHA256: `8c7f8a0bc24400f4caade76622f75ff22ca1e93e169add9d2b70093e2487fbe5`
- Cache command: `cutout models download --model toonout`
- Backend: PyTorch BiRefNet adapter
- Runtime repo: `ZhengPeng7/BiRefNet`
- Runtime revision: `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`
- Input size: `1024x1024`
- Mean: `(0.485, 0.456, 0.406)`
- Std: `(0.229, 0.224, 0.225)`
- Output postprocess: sigmoid, min/max normalize, resize float alpha to original
  image size, then feed the soft alpha into topology cleanup.

Provisioning notes:

- Install with `python -m pip install -e ".[quality]"`.
- `models download` verifies the `.pth` file and primes the pinned BiRefNet
  runtime code. This is required for reliable offline use on a second machine.
- `--quality` selects ToonOut when `--model` is `auto` or `emote`, enables
  bounded alpha refinement, and forces model inference even when source alpha
  exists.

## `birefnet-general-lite`

- Model file: `birefnet-general-lite.onnx`
- URL: `https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx`
- SHA256: `5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333`
- Cache command: `cutout models download --model birefnet-general-lite`
- Backend: ONNX Runtime
- Input size: `1024x1024`
- Mean: `(0.485, 0.456, 0.406)`
- Std: `(0.229, 0.224, 0.225)`
- Output postprocess: sigmoid, min/max normalize, resize float alpha to original
  image size, then feed the soft alpha into topology cleanup.

## Cache Policy

Default cache:

| OS | Path |
| --- | --- |
| macOS | `~/Library/Caches/snowy-bg-remover/models` |
| Windows | `%LOCALAPPDATA%/snowy-bg-remover/models` |
| Linux | `$XDG_CACHE_HOME/snowy-bg-remover/models` or `~/.cache/snowy-bg-remover/models` |

Cache override:

- `SNOWY_CUTOUT_MODEL_CACHE`
- `--model-cache <path>`

The repository does not commit model weights. Publish jobs should provision
models explicitly and then run normal inference offline. `--allow-download` is
reserved for one-shot setup, not deterministic production runs.

## Admission Rules For New Models

Before a model can become a built-in option:

1. License must be compatible with commercial/publish use.
2. Model file must be pinned by SHA256, not just a mutable branch or latest URL.
3. Normal `cutout` inference must load from local cache only.
4. Adapter must return a continuous alpha/probability map; threshold-first
   adapters are not acceptable.
5. Tests must cover missing cache, hash failure, successful adapter invocation,
   and confidence behavior on low-quality masks.
6. Benchmark results must include native-resolution alpha metrics and 128px/32px
   composited emote metrics.

## References

- SkyTNT anime-segmentation license: Apache-2.0.
- rembg `dis_anime.py` provides the public `isnet-anime` model name, ONNX URL,
  checksum, and preprocessing parameters.
- BiRefNet code and model family: `ZhengPeng7/BiRefNet`.
- ToonOut weights: `joelseytre/toonout`.
- ONNX Runtime provides CPU execution and optional platform execution providers
  such as CoreML.
