# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Archive schema-generated cudf-polars Quent events."""

from __future__ import annotations

import json
import re
import uuid
import zipfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SIDECAR_FILE_NAME = "model.qmi"
EXTENSION = "ndjson"
MODEL_QMI: dict[str, Any] = {
    "quent": {
        "version": "0.1.0",
        "commit": "0743198",
        "remote": "https://github.com/rapidsai/quent",
    },
    "model": {
        "name": "CudfPolars",
        "package": "cudf-polars-quent",
        "type_path": "_quent::CudfPolarsEvent",
        "source": {"path": "python/cudf_polars/quent/model.yaml"},
    },
}


def _entity_directory(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def to_export_line(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Remove the umbrella entity wrapper used by generated callback events."""
    data = event["data"]
    if len(data) != 1:
        raise ValueError(f"Expected one generated entity wrapper, got {data!r}")
    entity_name, payload = next(iter(data.items()))
    return _entity_directory(entity_name), {
        "id": event["id"],
        "timestamp": event["timestamp"],
        "data": payload,
    }


def write_quent_export(
    events: list[dict[str, Any]],
    export_root: Path,
    context_id: uuid.UUID,
    quent_archive: Path,
    *,
    sidecar: dict[str, Any] | None = None,
) -> Path:
    """Write gathered generated events to a Quent ZIP archive."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        directory, line = to_export_line(event)
        grouped.setdefault(directory, []).append(line)

    export_root.mkdir(parents=True, exist_ok=True)
    temporary = export_root / f".{context_id}.zip.tmp"
    context_dir = str(context_id)
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            f"{context_dir}/{SIDECAR_FILE_NAME}",
            json.dumps(MODEL_QMI if sidecar is None else sidecar, indent=2) + "\n",
        )
        for directory, lines in grouped.items():
            stream = f"{context_dir}/{directory}/{uuid.uuid4()}.{EXTENSION}"
            archive.writestr(
                stream,
                "\n".join(json.dumps(line, separators=(",", ":")) for line in lines)
                + "\n",
            )
    temporary.replace(quent_archive)
    return quent_archive


__all__ = [
    "EXTENSION",
    "MODEL_QMI",
    "SIDECAR_FILE_NAME",
    "to_export_line",
    "write_quent_export",
]
