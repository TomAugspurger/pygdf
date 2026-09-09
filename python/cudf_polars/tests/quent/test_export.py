# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for custom-schema Quent archive export."""

from __future__ import annotations

import json
import uuid
import zipfile
from typing import TYPE_CHECKING
from unittest.mock import ANY

import pytest

pytest.importorskip("cudf_polars._quent")

from cudf_polars.quent._export import (
    SIDECAR_FILE_NAME,
    to_export_line,
    write_quent_export,
)
from cudf_polars.quent._runtime import QuentSession

if TYPE_CHECKING:
    from pathlib import Path


def _generated_events() -> list[dict]:
    session = QuentSession()
    identifier = uuid.uuid4()
    engine = session.engine(identifier)
    engine.init(
        instance_name="test",
        implementation={
            "name": "cudf-polars",
            "version": "test",
            "backend": "spmd",
            "custom_attributes": {"backend": "spmd"},
        },
    )
    engine.exit()
    return [item["event"] for item in session.drain()]


def test_to_export_line_is_schema_generic() -> None:
    event = _generated_events()[0]
    directory, line = to_export_line(event)
    assert directory == "Engine"
    assert line["id"] == event["id"]
    assert "Init" in line["data"]


def test_write_quent_export_uses_custom_sidecar(tmp_path: Path) -> None:
    context_id = uuid.uuid4()
    archive_path = tmp_path / f"{context_id}.zip"
    write_quent_export(_generated_events(), tmp_path, context_id, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        sidecar = json.loads(archive.read(f"{context_id}/{SIDECAR_FILE_NAME}"))
        assert sidecar["model"]["name"] == "CudfPolars"
        assert sidecar["model"]["analyzer_package"] == "cudf-polars-quent-analyzer"
        assert sidecar["model"]["source"]["version"] == "0.1.0"
        assert sidecar["model"]["source"]["commit"]
        assert sidecar["model"]["source"]["remote"]
        streams = [
            name
            for name in archive.namelist()
            if name.startswith(f"{context_id}/Engine/")
        ]
        assert len(streams) == 1
        lines = [
            json.loads(line) for line in archive.read(streams[0]).decode().splitlines()
        ]
        assert [line["data"] for line in lines] == [
            {
                "Init": {
                    "instance_name": "test",
                    "implementation": {
                        "name": "cudf-polars",
                        "version": "test",
                        "backend": "spmd",
                        "custom_attributes": [
                            {"key": "backend", "value": {"String": "spmd"}}
                        ],
                    },
                }
            },
            "Exit",
        ]


def test_export_mirrors_streams_quent_open_indexes(tmp_path: Path) -> None:
    # quent-open finds engines by scanning snake-case stream names, so a missing
    # alias shows up as an empty engine list rather than an error.
    context_id = uuid.uuid4()
    archive_path = tmp_path / f"{context_id}.zip"
    write_quent_export(_generated_events(), tmp_path, context_id, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        streams = {name.split("/")[1] for name in archive.namelist() if "/" in name}
        assert {"Engine", "engine"} <= streams

        def payload(stream: str) -> list[dict]:
            name = next(
                item
                for item in archive.namelist()
                if item.startswith(f"{context_id}/{stream}/")
            )
            return [
                json.loads(line) for line in archive.read(name).decode().splitlines()
            ]

        generated = payload("Engine")
        assert [line["data"] for line in generated] == [{"Init": ANY}, "Exit"]
        # The alias carries the same events, but an attribute-less event has to
        # be a mapping for the legacy types to accept the stream.
        assert payload("engine") == [
            generated[0],
            {**generated[1], "data": {"Exit": None}},
        ]


def test_benchmark_writer_archives_generated_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cudf_polars.streaming.benchmarks import utils as benchmark_utils

    events = _generated_events()

    class FakeEngine:
        _quent_events = events

    run_id = uuid.uuid4()
    archive_path = tmp_path / "logs" / f"{run_id}.zip"
    monkeypatch.chdir(tmp_path)
    assert (
        benchmark_utils._write_quent_traces(
            FakeEngine(),  # type: ignore[arg-type]
            run_id,
            collect_traces=True,
            quent_archive=archive_path,
        )
        == archive_path
    )
    with zipfile.ZipFile(archive_path) as archive:
        assert f"{run_id}/{SIDECAR_FILE_NAME}" in archive.namelist()
