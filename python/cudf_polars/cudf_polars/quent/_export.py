# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Archive schema-generated cudf-polars Quent events."""

from __future__ import annotations

import json
import uuid
import zipfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SIDECAR_FILE_NAME = "model.qmi"
EXTENSION = "ndjson"

# Schema-generated streams are named after the entity ("Engine"), but
# `quent-open` discovers engines and their worker contexts by scanning the
# snake-case stream names its older `entity!` models export. Without these
# aliases the viewer builds and then lists no engines at all, so mirror the two
# streams that indexer reads. Keys are generated names; values are the aliases.
INDEX_STREAM_ALIASES = {"Engine": "engine", "Worker": "worker"}


def to_index_line(line: dict[str, Any]) -> dict[str, Any]:
    """
    Rewrite one event into the shape the legacy query-engine types deserialize.

    Those types spell every event as a newtype variant wrapping a struct, so an
    event without attributes has to be ``{"Exit": null}``. The schema generates
    a unit variant instead, which serializes to the bare string ``"Exit"`` and
    fails the indexer's whole stream, not just that event.
    """
    data = line["data"]
    if isinstance(data, str):
        return {**line, "data": {data: None}}
    return line


def _model_qmi() -> dict[str, Any]:
    """Return build provenance embedded in the generated extension."""
    from cudf_polars import _quent

    return json.loads(_quent.model_qmi())


def to_export_line(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Remove the umbrella entity wrapper used by generated callback events."""
    data = event["data"]
    if len(data) != 1:
        raise ValueError(f"Expected one generated entity wrapper, got {data!r}")
    entity_name, payload = next(iter(data.items()))
    return entity_name, {
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
            json.dumps(_model_qmi() if sidecar is None else sidecar, indent=2) + "\n",
        )
        for directory, lines in grouped.items():
            streams = [(directory, lines)]
            if (alias := INDEX_STREAM_ALIASES.get(directory)) is not None:
                streams.append((alias, [to_index_line(line) for line in lines]))
            for name, stream in streams:
                archive.writestr(
                    f"{context_dir}/{name}/{uuid.uuid4()}.{EXTENSION}",
                    "\n".join(
                        json.dumps(line, separators=(",", ":")) for line in stream
                    )
                    + "\n",
                )
    temporary.replace(quent_archive)
    return quent_archive


__all__ = [
    "EXTENSION",
    "INDEX_STREAM_ALIASES",
    "SIDECAR_FILE_NAME",
    "to_export_line",
    "to_index_line",
    "write_quent_export",
]
