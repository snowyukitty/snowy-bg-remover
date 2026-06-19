from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..model_manager import ModelManagerError, ensure_model, hash_file
from ..model_specs import ModelSpec, get_model_spec
from .base import MaskResult


class TorchBiRefNetAdapterError(RuntimeError):
    pass


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def _postprocess_output(
    output: np.ndarray, transform: str, target_size: tuple[int, int]
) -> np.ndarray:
    pred = np.asarray(output)
    if pred.ndim == 4:
        pred = pred[:, 0, :, :]
    pred = np.squeeze(pred).astype(np.float32)
    if transform == "sigmoid-minmax":
        pred = _sigmoid(pred)
    ma = float(np.max(pred, initial=0.0))
    mi = float(np.min(pred, initial=0.0))
    if ma - mi < 1e-6:
        alpha = np.zeros(pred.shape, dtype=np.float32)
    else:
        alpha = (pred - mi) / (ma - mi)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_image = Image.fromarray(alpha.astype(np.float32), mode="F")
    resized = alpha_image.resize(target_size, Image.Resampling.LANCZOS)
    return np.clip(np.asarray(resized).astype(np.float32), 0.0, 1.0)


def _normalize_image(
    image: Image.Image,
    *,
    size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    arr = (arr - mean_arr) / std_arr
    return arr.transpose((2, 0, 1))[None, :, :, :].astype(np.float32)


class TorchBiRefNetAdapter:
    def __init__(
        self,
        model_id: str,
        *,
        cache_dir: Path | None = None,
        allow_download: bool = False,
        offline: bool = True,
        device: str = "cpu",
        threads: int | None = None,
    ) -> None:
        self.spec: ModelSpec = get_model_spec(model_id)
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self.offline = offline
        self.device = device
        self.threads = threads
        self.model = None
        self.model_path: Path | None = None
        self.actual_device = "cpu"

    def _select_device(self, torch) -> str:
        requested = self.device.lower()
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise TorchBiRefNetAdapterError("CUDA is not available")
            return "cuda"
        if requested == "mps":
            has_mps = (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )
            if not has_mps:
                raise TorchBiRefNetAdapterError("MPS is not available")
            return "mps"
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "coreml":
            raise TorchBiRefNetAdapterError("CoreML is only supported by ONNX models")
        raise TorchBiRefNetAdapterError(f"unknown device '{self.device}'")

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModelForImageSegmentation
        except Exception as exc:
            raise TorchBiRefNetAdapterError(
                "toonout requires quality dependencies: torch, transformers, timm, "
                "einops, and kornia"
            ) from exc

        if self.threads is not None:
            torch.set_num_threads(self.threads)

        try:
            self.model_path = ensure_model(
                self.spec.model_id,
                cache_dir=self.cache_dir,
                allow_download=self.allow_download,
                offline=self.offline,
            )
        except ModelManagerError as exc:
            raise TorchBiRefNetAdapterError(str(exc)) from exc

        try:
            config = AutoConfig.from_pretrained(
                self.spec.runtime_repo,
                revision=self.spec.runtime_revision,
                trust_remote_code=True,
                local_files_only=self.offline,
            )
            model = AutoModelForImageSegmentation.from_config(
                config,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise TorchBiRefNetAdapterError(
                f"failed to load BiRefNet runtime code for {self.spec.model_id}: {exc}"
            ) from exc

        try:
            state = torch.load(self.model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise TypeError("checkpoint did not contain a state dict")
            state = {
                key.removeprefix("module._orig_mod.").removeprefix("module."): value
                for key, value in state.items()
            }
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise TorchBiRefNetAdapterError(
                f"failed to load {self.spec.model_id} weights: {exc}"
            ) from exc

        self.actual_device = self._select_device(torch)
        model.eval().float().to(self.actual_device)
        self.model = model

    def predict_alpha(self, image: Image.Image) -> MaskResult:
        self.load()
        if self.model is None or self.model_path is None:
            raise TorchBiRefNetAdapterError("model session did not initialize")

        try:
            import torch
        except Exception as exc:  # pragma: no cover - load() already checks this.
            raise TorchBiRefNetAdapterError("torch is not available") from exc

        started = time.perf_counter()
        tensor_np = _normalize_image(
            image,
            size=self.spec.input_size,
            mean=self.spec.mean,
            std=self.spec.std,
        )
        tensor = torch.from_numpy(tensor_np).to(self.actual_device).float()
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        output_np = output.detach().float().cpu().numpy()
        alpha = _postprocess_output(
            output_np,
            self.spec.output_transform,
            image.convert("RGB").size,
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        return MaskResult(
            alpha=alpha,
            model_name=self.spec.model_id,
            device=self.actual_device,
            elapsed_ms=elapsed_ms,
            raw_scores={
                "modelId": self.spec.model_id,
                "modelFile": str(self.model_path),
                "modelHashKind": self.spec.hash_kind,
                "modelHash": hash_file(self.model_path, self.spec.hash_kind),
                "modelLicense": self.spec.license,
                "modelBackend": self.spec.backend,
                "runtimeRepo": self.spec.runtime_repo,
                "runtimeRevision": self.spec.runtime_revision,
                "inferenceMs": elapsed_ms,
            },
        )
