from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .model_specs import MODEL_SPECS, ModelSpec, get_model_spec, resolve_model_id


class ModelManagerError(RuntimeError):
    pass


def default_model_cache_dir() -> Path:
    override = os.environ.get("SNOWY_CUTOUT_MODEL_CACHE")
    if override:
        return Path(override).expanduser()

    system = platform.system()
    if system == "Darwin":
        root = Path.home() / "Library" / "Caches"
    elif system == "Windows":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "snowy-bg-remover" / "models"


def hash_file(path: Path, hash_kind: str = "sha256") -> str:
    digest = hashlib.new(hash_kind)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_file(path: Path, spec: ModelSpec) -> bool:
    if not path.exists() or not path.is_file():
        return False
    actual = hash_file(path, spec.hash_kind)
    return actual.lower() == spec.hash_value.lower()


def model_dir(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    return (cache_dir or default_model_cache_dir()) / spec.model_id


def model_path(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    return model_dir(spec, cache_dir) / spec.filename


@contextmanager
def model_lock(lock_path: Path, timeout_s: float = 300.0) -> Iterator[None]:
    started = time.monotonic()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - started > timeout_s:
                raise ModelManagerError(f"timed out waiting for model lock: {lock_path}")
            time.sleep(0.2)
    try:
        os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


def write_metadata(path: Path, spec: ModelSpec) -> None:
    metadata = {
        "schemaVersion": 1,
        "downloadedAt": int(time.time()),
        "model": asdict(spec),
        "path": str(path),
    }
    metadata_path = path.with_suffix(path.suffix + ".json")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        dir=str(metadata_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, metadata_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def download_model(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    target = model_path(spec, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".download",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    digest = hashlib.new(spec.hash_kind)
    try:
        with urllib.request.urlopen(spec.url, timeout=60) as response:
            with os.fdopen(fd, "wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        actual = digest.hexdigest()
        if actual.lower() != spec.hash_value.lower():
            raise ModelManagerError(
                f"{spec.model_id} hash mismatch: expected {spec.hash_kind}:"
                f"{spec.hash_value}, got {actual}"
            )
        os.replace(tmp_path, target)
        write_metadata(target, spec)
        return target
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def prime_runtime_cache(spec: ModelSpec) -> None:
    if not spec.runtime_repo:
        return
    try:
        from transformers import AutoConfig
    except Exception as exc:
        raise ModelManagerError(
            f"{spec.model_id} requires the quality runtime dependencies; "
            "install snowy-bg-remover with the quality extra"
        ) from exc

    try:
        AutoConfig.from_pretrained(
            spec.runtime_repo,
            revision=spec.runtime_revision,
            trust_remote_code=True,
            local_files_only=False,
        )
    except Exception as exc:
        raise ModelManagerError(
            f"failed to prepare runtime code for {spec.model_id}: {exc}"
        ) from exc


def ensure_model(
    model_id: str,
    *,
    cache_dir: Path | None = None,
    allow_download: bool = False,
    offline: bool = True,
) -> Path:
    spec = get_model_spec(model_id)
    path = model_path(spec, cache_dir)
    if verify_model_file(path, spec):
        return path
    if path.exists():
        raise ModelManagerError(
            f"{spec.model_id} exists but failed {spec.hash_kind} verification: {path}"
        )
    if offline or not allow_download:
        raise ModelManagerError(
            f"{spec.model_id} is missing from cache; run "
            f"`cutout models download --model {spec.model_id}` first"
        )

    lock_path = path.with_suffix(path.suffix + ".lock")
    with model_lock(lock_path):
        if verify_model_file(path, spec):
            return path
        path = download_model(spec, cache_dir)
        prime_runtime_cache(spec)
        return path


def model_status(model_id: str, cache_dir: Path | None = None) -> dict:
    spec = get_model_spec(model_id)
    path = model_path(spec, cache_dir)
    exists = path.exists()
    verified = verify_model_file(path, spec) if exists else False
    actual_hash = hash_file(path, spec.hash_kind) if exists else None
    return {
        "model": spec.model_id,
        "filename": spec.filename,
        "path": str(path),
        "exists": exists,
        "verified": verified,
        "hashKind": spec.hash_kind,
        "expectedHash": spec.hash_value,
        "actualHash": actual_hash,
        "license": spec.license,
        "source": spec.source,
        "backend": spec.backend,
        "runtimeRepo": spec.runtime_repo,
        "runtimeRevision": spec.runtime_revision,
    }


def all_model_status(cache_dir: Path | None = None) -> list[dict]:
    return [model_status(model_id, cache_dir) for model_id in sorted(MODEL_SPECS)]


def download_by_id(
    model_id: str,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict:
    resolved = resolve_model_id(model_id)
    spec = get_model_spec(resolved)
    path = model_path(spec, cache_dir)
    if force and path.exists():
        path.unlink()
    path = ensure_model(
        spec.model_id,
        cache_dir=cache_dir,
        allow_download=True,
        offline=False,
    )
    prime_runtime_cache(spec)
    return model_status(spec.model_id, cache_dir)
