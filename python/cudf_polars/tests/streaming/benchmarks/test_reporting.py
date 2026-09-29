# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark reporting schema version 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cudf_polars.streaming.benchmarks.reporting import build_report


def _run(**overrides: Any) -> dict[str, Any]:
    plan = {
        "nodes": {
            "0": {
                "type": "Scan",
                "properties": {"paths": ["/private/data/a.parquet"]},
            }
        },
        "partition_info": {},
        "roots": ["0"],
    }
    return {
        "engine_name": "cudf-polars",
        "frontend": "ray",
        "dataset_path": "/data",
        "scale_factor": 10,
        "iterations": 1,
        "query_set": "pdsh",
        "io_mode": "lukewarm",
        "n_workers": 2,
        "extra_info": {
            "sku_name": "h100",
            "storage_configuration_name": "local",
            "benchmark_definition_name": "pdsh-10",
            "node_count": 1,
            "identifier_hash": "environment-hash",
            "container_image": "image:tag",
            "labels": ["nightly"],
        },
        "run_id": "00000000-0000-0000-0000-000000000001",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "command_line": "python -m benchmark",
        "config_options": {"executor": {"max_rows": 10}},
        "records": {
            1: [
                {
                    "query": 1,
                    "iteration": 0,
                    "duration": 0.25,
                    "status": "success",
                    "statistics": {"alloc": {"count": 2, "value": 3.5}},
                    "io_summaries": {"0": {"bytes_read": 100}},
                    "validation_result": {
                        "status": "Passed",
                        "message": None,
                        "details": None,
                    },
                    "traces": [
                        {"scope": "plan", "plan": plan},
                        {
                            "scope": "actor",
                            "event": "finished",
                            "row_count": 10,
                        },
                    ],
                }
            ]
        },
        "plans": {},
        "versions": {
            "cudf_polars": {"version": "26.12", "commit": "abc"},
            "polars": "1.40",
            "python": "3.12",
            "rapidsmpf": {"version": "26.12", "commit": "def"},
            "duckdb": None,
        },
        "hardware": {
            "gpus": [{"name": "H100", "total_memory": 80_000}],
            "cpu": {"model": "cpu"},
        },
        "validation_method": None,
        "roles": [{"type": "nightly", "date": "2026-01-01"}],
        "startup_duration_ms": 12.5,
        "shutdown_duration_ms": 3.5,
        **overrides,
    }


def test_build_report_matches_submission_shape(tmp_path: Path) -> None:
    """The JSON record is API-shaped and uses API units and enum spellings."""
    output = tmp_path / "results.jsonl"
    report = build_report(_run(), output_path=output)

    assert report["schema_version"] == 2
    assert report["cache_state"] == "warm"
    assert report["gpu_count"] == 2
    assert report["query_engine"] == {
        "engine_name": "cudf-polars",
        "version": "26.12",
        "commit_hash": "abc",
        "identifier_hash": "environment-hash",
        "container_image": "image:tag",
    }
    assert report["validation_status"] == "passed"
    log = report["query_logs"][0]
    assert log["runtime_ms"] == 250
    assert log["validation_result"]["status"] == "passed"
    assert log["engine_metrics"] == {"alloc.count": 2, "alloc.value": 3.5}
    assert log["extra_info"]["iteration"] == 0
    assert log["plan"]["nodes"]["0"]["properties"]["paths"] == []
    assert report["startup_duration_ms"] == 12.5


def test_build_report_writes_upload_ready_sidecars(tmp_path: Path) -> None:
    """Traces, plans, and raw JSON are represented by relative manifest paths."""
    output = tmp_path / "results.jsonl"
    report = build_report(_run(), output_path=output)
    artifacts = {item["filename"]: item for item in report["artifacts"]}
    run_id = "00000000-0000-0000-0000-000000000001"

    trace_name = f"{run_id}-traces.parquet"
    plan_name = f"{run_id}-plans.parquet"
    raw_name = f"{run_id}-raw.json"
    assert {trace_name, plan_name, raw_name} <= artifacts.keys()
    for item in artifacts.values():
        assert not Path(item["path"]).is_absolute()
        assert (tmp_path / item["path"]).is_file()

    traces = pl.read_parquet(tmp_path / artifacts[trace_name]["path"])
    assert traces.select("query_id", "iteration", "scope").row(0) == (
        "1",
        0,
        "actor",
    )
    plans = pl.read_parquet(tmp_path / artifacts[plan_name]["path"])
    assert json.loads(plans["plan"][0])["nodes"]["0"]["properties"]["paths"] == []
    raw = json.loads((tmp_path / artifacts[raw_name]["path"]).read_text())
    assert raw["query_logs"] == report["query_logs"]
    assert "artifacts" not in raw


def test_cpu_report_has_no_gpus(tmp_path: Path) -> None:
    """CPU engines report zero GPUs and their own package version."""
    run = _run(
        engine_name="polars-cpu",
        versions={
            "cudf_polars": {"version": "26.12", "commit": "abc"},
            "polars": "1.40",
        },
    )
    report = build_report(run, output_path=tmp_path / "results.jsonl")
    assert report["gpu_count"] == 0
    assert report["query_engine"]["version"] == "1.40"
    assert report["query_engine"]["commit_hash"] == ""
