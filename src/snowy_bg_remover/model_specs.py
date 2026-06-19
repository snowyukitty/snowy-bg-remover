from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    filename: str
    url: str
    hash_kind: str
    hash_value: str
    license: str
    source: str
    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    output_transform: str = "minmax"
    default_device: str = "cpu"
    backend: str = "onnx"
    runtime_repo: str | None = None
    runtime_revision: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "isnet-anime": ModelSpec(
        model_id="isnet-anime",
        filename="isnet-anime.onnx",
        url=(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
            "isnet-anime.onnx"
        ),
        hash_kind="sha256",
        hash_value="f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
        license="Apache-2.0",
        source="SkyTNT/anime-segmentation via danielgatis/rembg ONNX export",
        input_size=(1024, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(1.0, 1.0, 1.0),
    ),
    "birefnet-general-lite": ModelSpec(
        model_id="birefnet-general-lite",
        filename="birefnet-general-lite.onnx",
        url=(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
            "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
        ),
        hash_kind="sha256",
        hash_value="5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
        license="MIT",
        source="ZhengPeng7/BiRefNet via danielgatis/rembg ONNX export",
        input_size=(1024, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        output_transform="sigmoid-minmax",
    ),
    "toonout": ModelSpec(
        model_id="toonout",
        filename="birefnet_finetuned_toonout.pth",
        url=(
            "https://huggingface.co/joelseytre/toonout/resolve/"
            "cbf720eca394edcde66b861a8a8c20fbabe9c748/"
            "birefnet_finetuned_toonout.pth"
        ),
        hash_kind="sha256",
        hash_value="8c7f8a0bc24400f4caade76622f75ff22ca1e93e169add9d2b70093e2487fbe5",
        license="MIT",
        source="joelseytre/toonout BiRefNet anime fine-tune",
        input_size=(1024, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        output_transform="sigmoid-minmax",
        backend="torch-birefnet",
        runtime_repo="ZhengPeng7/BiRefNet",
        runtime_revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
    ),
}


MODEL_ALIASES = {
    "auto": "isnet-anime",
    "emote": "isnet-anime",
    "quality": "toonout",
}


def resolve_model_id(model_id: str) -> str:
    normalized = model_id.strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)


def get_model_spec(model_id: str) -> ModelSpec:
    resolved = resolve_model_id(model_id)
    try:
        return MODEL_SPECS[resolved]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"unknown model '{model_id}', supported models: {supported}") from exc
