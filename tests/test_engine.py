from __future__ import annotations

import json

import numpy as np
from PIL import Image

from snowy_bg_remover.adapters.registry import create_adapter
from snowy_bg_remover.adapters.base import MaskResult
from snowy_bg_remover.adapters.torch_birefnet import TorchBiRefNetAdapter
from snowy_bg_remover.cli import QUALITY_MODEL, apply_quality_defaults, main
from snowy_bg_remover.contracts import EXIT_MODEL_MISSING, EXIT_SUCCESS, CutoutOptions
from snowy_bg_remover.engine import process_image


def make_alpha_image(path) -> None:
    image = Image.new("RGBA", (16, 16), (255, 255, 255, 0))
    for y in range(4, 12):
        for x in range(4, 12):
            image.putpixel((x, y), (20, 80, 220, 255))
    image.save(path)


class FakeAdapter:
    def load(self) -> None:
        return None

    def predict_alpha(self, image: Image.Image) -> MaskResult:
        alpha = np.zeros((image.height, image.width), dtype=np.float32)
        alpha[4:12, 4:12] = 1.0
        return MaskResult(
            alpha=alpha,
            model_name="fake",
            device="cpu",
            elapsed_ms=1,
            raw_scores={"inferenceMs": 1},
        )


def test_process_existing_alpha_check_ok(tmp_path) -> None:
    input_path = tmp_path / "source.png"
    make_alpha_image(input_path)

    result = process_image(
        CutoutOptions(
            input_path=input_path,
            check=True,
            model_cache_dir=tmp_path / "empty-model-cache",
        )
    )

    assert result.ok
    assert result.exit_code == EXIT_SUCCESS
    assert result.hadAlpha
    assert result.model == "input-alpha"
    assert result.bbox == [4, 4, 8, 8]


def test_process_opaque_without_adapter_fails_loudly(tmp_path) -> None:
    input_path = tmp_path / "opaque.png"
    Image.new("RGB", (16, 16), (255, 255, 255)).save(input_path)

    result = process_image(
        CutoutOptions(
            input_path=input_path,
            check=True,
            model_cache_dir=tmp_path / "empty-model-cache",
        )
    )

    assert not result.ok
    assert result.reason == "model_unavailable"
    assert result.exit_code == EXIT_MODEL_MISSING


def test_process_opaque_with_adapter_succeeds(tmp_path) -> None:
    input_path = tmp_path / "opaque.png"
    output_path = tmp_path / "out.png"
    Image.new("RGB", (16, 16), (255, 255, 255)).save(input_path)

    result = process_image(
        CutoutOptions(input_path=input_path, output_path=output_path),
        adapter=FakeAdapter(),
    )

    assert result.ok
    assert result.model == "fake"
    assert result.bbox == [4, 4, 8, 8]
    assert output_path.exists()


def test_process_writes_png_atomically_on_success(tmp_path) -> None:
    input_path = tmp_path / "source.png"
    output_path = tmp_path / "out.png"
    make_alpha_image(input_path)

    result = process_image(CutoutOptions(input_path=input_path, output_path=output_path))

    assert result.ok
    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.mode == "RGBA"
        assert image.size == (16, 16)


def test_process_writes_explain_artifacts(tmp_path) -> None:
    input_path = tmp_path / "source.png"
    explain_dir = tmp_path / "explain"
    make_alpha_image(input_path)

    result = process_image(
        CutoutOptions(input_path=input_path, check=True, explain_dir=explain_dir)
    )

    assert result.ok
    assert (explain_dir / "source-alpha.png").exists()
    assert (explain_dir / "final-alpha.png").exists()
    assert (explain_dir / "keep-mask.png").exists()
    assert (explain_dir / "result.json").exists()


def test_process_rejects_multiple_high_confidence_cores(tmp_path) -> None:
    input_path = tmp_path / "two.png"
    image = Image.new("RGBA", (16, 16), (255, 255, 255, 0))
    for y in range(2, 6):
        for x in range(2, 6):
            image.putpixel((x, y), (255, 0, 0, 255))
    for y in range(10, 14):
        for x in range(10, 14):
            image.putpixel((x, y), (0, 0, 255, 255))
    image.save(input_path)

    result = process_image(CutoutOptions(input_path=input_path, check=True))

    assert not result.ok
    assert result.reason == "low_confidence"
    assert result.metrics["seedComponentCount"] == 2


def test_cli_check_outputs_machine_readable_json(tmp_path, capsys) -> None:
    input_path = tmp_path / "source.png"
    make_alpha_image(input_path)

    exit_code = main(
        ["--input", str(input_path), "--output", str(tmp_path / "unused.png"), "--check"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == EXIT_SUCCESS
    assert payload["ok"] is True
    assert payload["output"] is None
    assert payload["bbox"] == [4, 4, 8, 8]


def test_cli_batch_summary_writes_outputs(tmp_path, capsys) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    make_alpha_image(input_dir / "one.png")
    make_alpha_image(input_dir / "two.png")

    exit_code = main(
        ["--input-dir", str(input_dir), "--output-dir", str(output_dir), "--trim"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == EXIT_SUCCESS
    assert payload["ok"] is True
    assert payload["total"] == 2
    assert (output_dir / "one.png").exists()
    assert (output_dir / "two.png").exists()


def test_cli_glob_batch_rejects_output_collisions(tmp_path, capsys) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    make_alpha_image(left / "same.png")
    make_alpha_image(right / "same.png")

    exit_code = main(
        [
            "--glob",
            str(tmp_path / "**" / "*.png"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code != EXIT_SUCCESS
    assert payload["reason"] == "usage_error"
    assert "collision" in payload["message"]


def test_cli_batch_fail_fast_stops_after_first_failure(tmp_path, capsys) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    Image.new("RGB", (16, 16), (255, 255, 255)).save(input_dir / "one.png")
    Image.new("RGB", (16, 16), (255, 255, 255)).save(input_dir / "two.png")

    exit_code = main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "output"),
            "--model-cache",
            str(tmp_path / "empty-model-cache"),
            "--fail-fast",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == EXIT_MODEL_MISSING
    assert payload["matched"] == 2
    assert payload["processed"] == 1
    assert payload["failed"] == 1


def test_cli_models_list_outputs_json(capsys) -> None:
    exit_code = main(["models", "list"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == EXIT_SUCCESS
    assert payload["ok"] is True
    assert payload["models"][0]["model"] == "isnet-anime"


def test_cli_quality_defaults_select_quality_model_and_force_model() -> None:
    class Args:
        quality = True
        model = "auto"
        force_model = False

    args = Args()

    apply_quality_defaults(args)  # type: ignore[arg-type]

    assert args.model == QUALITY_MODEL
    assert args.force_model is True


def test_registry_dispatches_toonout_to_torch_adapter(tmp_path) -> None:
    adapter = create_adapter("toonout", cache_dir=tmp_path, device="cpu")

    assert isinstance(adapter, TorchBiRefNetAdapter)


def test_cli_models_status_missing_returns_model_exit(tmp_path, capsys) -> None:
    exit_code = main(
        ["models", "status", "--model", "isnet-anime", "--model-cache", str(tmp_path)]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == EXIT_MODEL_MISSING
    assert payload["ok"] is False
    assert payload["models"][0]["verified"] is False
