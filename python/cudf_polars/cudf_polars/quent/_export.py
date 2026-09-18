# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package collector-produced cudf-polars Quent contexts."""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SIDECAR_FILE_NAME = "model.qmi"
EXTENSION = "ndjson"
INDEX_STREAM_ALIASES = {"Engine": "engine", "Worker": "worker"}


def to_index_line(line: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a generated FSM event for the legacy query-engine indexer."""
    data = line["data"]
    if isinstance(data, dict) and len(data) == 1:
        event, payload = next(iter(data.items()))
        if isinstance(payload, dict):
            legacy_payload = {
                key: value for key, value in payload.items() if key != "seq"
            }
            return {**line, "data": {event: legacy_payload or None}}
    return line


def write_quent_export(export_root: Path, quent_archive: Path) -> Path:
    """Package collector-produced contexts into one Quent ZIP archive."""
    quent_archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = quent_archive.with_name(f".{quent_archive.name}.tmp")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(export_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(export_root)
            archive.write(path, relative)
            alias = INDEX_STREAM_ALIASES.get(path.parent.name)
            if alias is not None and path.suffix == f".{EXTENSION}":
                lines = [
                    to_index_line(json.loads(line))
                    for line in path.read_text().splitlines()
                ]
                alias_path = relative.parent.parent / alias / relative.name
                archive.writestr(
                    str(alias_path),
                    "\n".join(json.dumps(line, separators=(",", ":")) for line in lines)
                    + "\n",
                )
    temporary.replace(quent_archive)
    return quent_archive


__all__ = [
    "EXTENSION",
    "INDEX_STREAM_ALIASES",
    "SIDECAR_FILE_NAME",
    "to_index_line",
    "write_quent_export",
]
