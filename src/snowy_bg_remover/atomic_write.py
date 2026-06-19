from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image


def atomic_save_png(image: Image.Image, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp.png",
        dir=str(output_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            image.save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        with Image.open(tmp_path) as check:
            check.verify()
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise
