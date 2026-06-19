from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

from snowy_bg_remover.model_manager import (
    ModelManagerError,
    download_model,
    ensure_model,
    model_path,
    prime_runtime_cache,
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


def test_prime_runtime_cache_instantiates_remote_model_class(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("config", (args, kwargs)))
            return {"runtime": "config"}

    class FakeAutoModelForImageSegmentation:
        @staticmethod
        def from_config(config, **kwargs):
            calls.append(("model", (config, kwargs)))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=FakeAutoConfig,
            AutoModelForImageSegmentation=FakeAutoModelForImageSegmentation,
        ),
    )

    spec = ModelSpec(
        model_id="fake-quality",
        filename="fake.pth",
        url="https://example.test/fake.pth",
        hash_kind="sha256",
        hash_value="0" * 64,
        license="MIT",
        source="test",
        input_size=(1024, 1024),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        backend="torch-birefnet",
        runtime_repo="example/runtime",
        runtime_revision="abc123",
    )

    prime_runtime_cache(spec)

    assert calls[0][0] == "config"
    config_args, config_kwargs = calls[0][1]
    assert config_args == ("example/runtime",)
    assert config_kwargs["revision"] == "abc123"
    assert config_kwargs["trust_remote_code"] is True
    assert config_kwargs["local_files_only"] is False
    assert calls[1] == (
        "model",
        ({"runtime": "config"}, {"trust_remote_code": True}),
    )
