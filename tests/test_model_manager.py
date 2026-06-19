from __future__ import annotations

import hashlib

import pytest

from snowy_bg_remover.model_manager import (
    ModelManagerError,
    download_model,
    ensure_model,
    model_path,
    verify_model_file,
)
from snowy_bg_remover.model_specs import ModelSpec


def test_download_model_verifies_hash_and_writes_metadata(tmp_path) -> None:
    source = tmp_path / "source.onnx"
    payload = b"small-model-fixture"
    source.write_bytes(payload)
    spec = ModelSpec(
        model_id="fixture",
        filename="fixture.onnx",
        url=source.as_uri(),
        hash_kind="sha256",
        hash_value=hashlib.sha256(payload).hexdigest(),
        license="fixture",
        source="fixture",
        input_size=(4, 4),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    path = download_model(spec, tmp_path / "cache")

    assert path == model_path(spec, tmp_path / "cache")
    assert verify_model_file(path, spec)
    assert path.with_suffix(".onnx.json").exists()


def test_ensure_model_missing_offline_fails_loudly(tmp_path) -> None:
    with pytest.raises(ModelManagerError, match="missing from cache"):
        ensure_model("isnet-anime", cache_dir=tmp_path, allow_download=False, offline=True)
