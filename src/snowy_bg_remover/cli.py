from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Sequence

from .contracts import (
    CutoutOptions,
    CutoutResult,
    EXIT_MODEL_MISSING,
    EXIT_SUCCESS,
    EXIT_USAGE,
)
from .adapters.registry import create_adapter
from .engine import process_image
from .model_manager import (
    all_model_status,
    default_model_cache_dir,
    download_by_id,
    model_status,
)
from .model_specs import MODEL_SPECS, resolve_model_id


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


class CutoutArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = CutoutArgumentParser(
        prog="cutout",
        description="Local-first character/emote background removal CLI.",
        add_help=True,
    )
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--glob", dest="glob_pattern")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--profile", default="emote")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--high-threshold", type=float, default=0.85)
    parser.add_argument("--low-threshold", type=float, default=0.05)
    parser.add_argument("--min-subject-coverage", type=float, default=0.01)
    parser.add_argument("--max-subject-coverage", type=float, default=0.98)
    parser.add_argument("--max-hole-area-ratio", type=float, default=0.02)
    parser.add_argument("--force-model", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--device", choices=["cpu", "auto", "coreml", "cuda"], default="cpu"
    )
    parser.add_argument("--threads", type=positive_int)
    parser.add_argument("--model-cache")
    parser.add_argument("--explain-dir")
    parser.add_argument("--no-decontaminate", action="store_true")
    parser.add_argument("--decontaminate-radius", type=positive_int, default=24)
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--pad")
    parser.add_argument("--square", action="store_true")
    parser.add_argument("--emit-size", type=positive_int)
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def build_models_parser() -> argparse.ArgumentParser:
    parser = CutoutArgumentParser(prog="cutout models", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--model-cache")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--model", default="all")
    status_parser.add_argument("--model-cache")

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--model", default="isnet-anime")
    download_parser.add_argument("--model-cache")
    download_parser.add_argument("--force", action="store_true")
    return parser


def usage_failure(message: str) -> int:
    result = CutoutResult(
        ok=False,
        reason="usage_error",
        input="",
        output=None,
        message=message,
    )
    print_json(result.to_json_dict())
    return EXIT_USAGE


def run_models(argv: Sequence[str]) -> int:
    parser = build_models_parser()
    try:
        args, unknown = parser.parse_known_args(argv)
    except (argparse.ArgumentError, ValueError) as exc:
        return usage_failure(str(exc))
    if unknown:
        return usage_failure(f"unknown arguments: {' '.join(unknown)}")

    cache_dir = Path(args.model_cache) if getattr(args, "model_cache", None) else None
    if args.command == "list":
        print_json(
            {
                "schemaVersion": 1,
                "ok": True,
                "cacheDir": str(cache_dir or default_model_cache_dir()),
                "models": [
                    {
                        "model": spec.model_id,
                        "filename": spec.filename,
                        "license": spec.license,
                        "source": spec.source,
                    }
                    for spec in MODEL_SPECS.values()
                ],
            }
        )
        return EXIT_SUCCESS

    if args.command == "status":
        try:
            if args.model == "all":
                models = all_model_status(cache_dir)
            else:
                models = [model_status(resolve_model_id(args.model), cache_dir)]
        except Exception as exc:
            result = CutoutResult(
                ok=False,
                reason="model_unavailable",
                input="",
                output=None,
                message=str(exc),
            )
            print_json(result.to_json_dict())
            return result.exit_code
        print_json(
            {
                "schemaVersion": 1,
                "ok": all(item["verified"] for item in models),
                "cacheDir": str(cache_dir or default_model_cache_dir()),
                "models": models,
            }
        )
        return (
            EXIT_SUCCESS
            if all(item["verified"] for item in models)
            else EXIT_MODEL_MISSING
        )

    if args.command == "download":
        requested = sorted(MODEL_SPECS) if args.model == "all" else [args.model]
        try:
            models = [
                download_by_id(model, cache_dir=cache_dir, force=args.force)
                for model in requested
            ]
        except Exception as exc:
            result = CutoutResult(
                ok=False,
                reason="model_unavailable",
                input="",
                output=None,
                message=str(exc),
            )
            print_json(result.to_json_dict())
            return result.exit_code
        print_json(
            {
                "schemaVersion": 1,
                "ok": True,
                "cacheDir": str(cache_dir or default_model_cache_dir()),
                "models": models,
            }
        )
        return EXIT_SUCCESS

    return usage_failure("unknown models command")


def discover_inputs(input_dir: str | None, glob_pattern: str | None) -> list[Path]:
    if input_dir:
        root = Path(input_dir)
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if glob_pattern:
        return sorted(
            Path(path)
            for path in glob.glob(glob_pattern, recursive=True)
            if Path(path).is_file() and Path(path).suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def batch_output_path(input_path: Path, args: argparse.Namespace) -> Path | None:
    if args.check:
        return None
    output_dir = Path(args.output_dir)
    if args.input_dir:
        try:
            relative = input_path.relative_to(Path(args.input_dir))
        except ValueError:
            relative = Path(input_path.name)
    else:
        relative = Path(input_path.name)
    return (output_dir / relative).with_suffix(".png")


def make_options(
    *,
    input_path: Path,
    output_path: Path | None,
    args: argparse.Namespace,
) -> CutoutOptions:
    return CutoutOptions(
        input_path=input_path,
        output_path=output_path,
        check=args.check,
        profile=args.profile,
        model=args.model,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        min_subject_coverage=args.min_subject_coverage,
        max_subject_coverage=args.max_subject_coverage,
        max_hole_area_ratio=args.max_hole_area_ratio,
        force_model=args.force_model,
        allow_download=args.allow_download,
        device=args.device,
        threads=args.threads,
        model_cache_dir=Path(args.model_cache) if args.model_cache else None,
        explain_dir=Path(args.explain_dir) if args.explain_dir else None,
        decontaminate_edges=not args.no_decontaminate,
        decontaminate_radius=args.decontaminate_radius,
        trim=args.trim,
        pad=args.pad,
        square=args.square,
        emit_size=args.emit_size,
    )


def validate_common(args: argparse.Namespace) -> str | None:
    if not 0.0 <= args.low_threshold <= 1.0:
        return "--low-threshold must be between 0 and 1"
    if not 0.0 <= args.high_threshold <= 1.0:
        return "--high-threshold must be between 0 and 1"
    if args.low_threshold >= args.high_threshold:
        return "--low-threshold must be lower than --high-threshold"
    if args.min_subject_coverage < 0 or args.max_subject_coverage > 1:
        return "subject coverage thresholds must be within [0, 1]"
    if args.min_subject_coverage >= args.max_subject_coverage:
        return "--min-subject-coverage must be lower than --max-subject-coverage"
    if args.max_hole_area_ratio < 0:
        return "--max-hole-area-ratio must be non-negative"
    return None


def is_batch(args: argparse.Namespace) -> bool:
    return bool(args.input_dir or args.glob_pattern)


def run_single(args: argparse.Namespace) -> int:
    if not args.input_path:
        return usage_failure("--input is required for single-image mode")
    if not args.check and not args.output_path:
        return usage_failure("--output is required unless --check is used")

    options = make_options(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path) if args.output_path else None,
        args=args,
    )
    result = process_image(options)
    print_json(result.to_json_dict())
    return result.exit_code


def run_batch(args: argparse.Namespace) -> int:
    if args.input_path or args.output_path:
        return usage_failure("do not mix --input/--output with batch mode")
    if args.input_dir and args.glob_pattern:
        return usage_failure("use either --input-dir or --glob, not both")
    if not args.check and not args.output_dir:
        return usage_failure("--output-dir is required unless --check is used")

    inputs = discover_inputs(args.input_dir, args.glob_pattern)
    if not inputs:
        return usage_failure("batch input did not match any supported image files")

    tasks: list[tuple[Path, Path | None]] = []
    seen_outputs: dict[Path, Path] = {}
    for input_path in inputs:
        output_path = batch_output_path(input_path, args)
        if output_path is not None:
            resolved_output = output_path.resolve()
            if resolved_output in seen_outputs:
                return usage_failure(
                    "batch output collision: "
                    f"{seen_outputs[resolved_output]} and {input_path} -> {output_path}"
                )
            seen_outputs[resolved_output] = input_path
        tasks.append((input_path, output_path))

    try:
        shared_adapter = create_adapter(
            args.model,
            cache_dir=Path(args.model_cache) if args.model_cache else None,
            allow_download=args.allow_download,
            offline=not args.allow_download,
            device=args.device,
            threads=args.threads,
        )
    except Exception as exc:
        result = CutoutResult(
            ok=False,
            reason="model_unavailable",
            input=str(args.input_dir or args.glob_pattern),
            output=str(args.output_dir or ""),
            message=str(exc),
        )
        print_json(result.to_json_dict())
        return result.exit_code

    results = []
    exit_code = EXIT_SUCCESS
    for input_path, output_path in tasks:
        options = make_options(
            input_path=input_path,
            output_path=output_path,
            args=args,
        )
        result = process_image(options, adapter=shared_adapter)
        results.append(result)
        if args.jsonl:
            print_json(result.to_json_dict())
        if exit_code == EXIT_SUCCESS and result.exit_code != EXIT_SUCCESS:
            exit_code = result.exit_code
        if args.fail_fast and result.exit_code != EXIT_SUCCESS:
            break

    if args.jsonl:
        return exit_code

    payload = {
        "schemaVersion": 1,
        "ok": exit_code == EXIT_SUCCESS,
        "matched": len(tasks),
        "processed": len(results),
        "total": len(results),
        "succeeded": sum(1 for result in results if result.ok),
        "failed": sum(1 for result in results if not result.ok),
        "items": [result.to_json_dict() for result in results],
    }
    print_json(payload)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if actual_argv[:1] == ["models"]:
        return run_models(actual_argv[1:])

    parser = build_parser()
    try:
        args, unknown = parser.parse_known_args(actual_argv)
    except (argparse.ArgumentError, ValueError) as exc:
        return usage_failure(str(exc))

    if unknown:
        return usage_failure(f"unknown arguments: {' '.join(unknown)}")

    common_error = validate_common(args)
    if common_error:
        return usage_failure(common_error)

    try:
        if is_batch(args):
            return run_batch(args)
        return run_single(args)
    except Exception as exc:
        result = CutoutResult(
            ok=False,
            reason="internal_error",
            input=str(args.input_path or args.input_dir or args.glob_pattern or ""),
            output=str(args.output_path or args.output_dir or ""),
            message=f"internal error: {exc}",
        )
        print_json(result.to_json_dict())
        return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
