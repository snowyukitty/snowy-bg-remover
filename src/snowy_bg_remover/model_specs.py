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
}


MODEL_ALIASES = {
    "auto": "isnet-anime",
    "emote": "isnet-anime",
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
