from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..model_manager import ModelManagerError, ensure_model, hash_file
from ..model_specs import ModelSpec, get_model_spec
from .base import MaskResult


class OnnxAdapterError(RuntimeError):
    pass


def _normalize_image(
    image: Image.Image,
    *,
    size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    input_name: str,
) -> dict[str, np.ndarray]:
    resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(resized).astype(np.float32)
    arr = arr / max(float(arr.max(initial=0.0)), 1e-6)
    for channel in range(3):
        arr[:, :, channel] = (arr[:, :, channel] - mean[channel]) / std[channel]
    tensor = arr.transpose((2, 0, 1))[None, :, :, :].astype(np.float32)
    return {input_name: tensor}


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


class OnnxSegmentationAdapter:
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
        self.session = None
        self.model_path: Path | None = None
        self.providers: list[str] = []

    def _providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        requested = self.device.lower()
        if requested == "cpu":
            return ["CPUExecutionProvider"]
        if requested == "coreml":
            if "CoreMLExecutionProvider" not in available:
                raise OnnxAdapterError("CoreMLExecutionProvider is not available")
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if requested == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise OnnxAdapterError("CUDAExecutionProvider is not available")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if requested == "auto":
            if "CoreMLExecutionProvider" in available and os.name != "nt":
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
        raise OnnxAdapterError(f"unknown device '{self.device}'")

    def load(self) -> None:
        if self.session is not None:
            return
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise OnnxAdapterError(
                "onnxruntime is not installed; install snowy-bg-remover with ONNX support"
            ) from exc

        try:
            self.model_path = ensure_model(
                self.spec.model_id,
                cache_dir=self.cache_dir,
                allow_download=self.allow_download,
                offline=self.offline,
            )
        except ModelManagerError as exc:
            raise OnnxAdapterError(str(exc)) from exc

        session_options = ort.SessionOptions()
        if self.threads is not None:
            session_options.inter_op_num_threads = self.threads
            session_options.intra_op_num_threads = self.threads

        self.providers = self._providers(ort)
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=self.providers,
        )

    def predict_alpha(self, image: Image.Image) -> MaskResult:
        self.load()
        if self.session is None or self.model_path is None:
            raise OnnxAdapterError("model session did not initialize")

        started = time.perf_counter()
        input_name = self.session.get_inputs()[0].name
        inputs = _normalize_image(
            image,
            size=self.spec.input_size,
            mean=self.spec.mean,
            std=self.spec.std,
            input_name=input_name,
        )
        outputs = self.session.run(None, inputs)
        alpha = _postprocess_output(
            outputs[0], self.spec.output_transform, image.convert("RGB").size
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        actual_providers = list(self.session.get_providers())
        return MaskResult(
            alpha=alpha,
            model_name=self.spec.model_id,
            device=actual_providers[0] if actual_providers else self.device,
            elapsed_ms=elapsed_ms,
            raw_scores={
                "modelId": self.spec.model_id,
                "modelFile": str(self.model_path),
                "modelHashKind": self.spec.hash_kind,
                "modelHash": hash_file(self.model_path, self.spec.hash_kind),
                "modelLicense": self.spec.license,
                "providers": actual_providers,
                "inferenceMs": elapsed_ms,
            },
        )
