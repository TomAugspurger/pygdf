# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize benchmark runs into the portable reporting format."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import polars as pl

SCHEMA_VERSION = 2

_CATALOG_FIELDS = {
    "sku_name",
    "storage_configuration_name",
    "benchmark_definition_name",
    "node_count",
}
_TRANSPORT_FIELDS = {
    *_CATALOG_FIELDS,
    "cache_state",
    "container_image",
    "identifier_hash",
    "is_official",
    "labels",
    "nsys-report",
    "quent-archive",
}


def _strip_scan_paths(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a plan without potentially enormous scan path lists."""
    result = copy.deepcopy(plan)
    for node in result.get("nodes", {}).values():
        if node.get("type") == "Scan":
            node.get("properties", {})["paths"] = []
    return result


def _plan_from_traces(
    traces: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Extract and sanitize the first plan trace."""
    for trace in traces or ():
        if trace.get("scope") == "plan" and trace.get("plan") is not None:
            return _strip_scan_paths(trace["plan"])
    return None


def _flatten_statistics(
    statistics: dict[str, Any] | None,
) -> dict[str, int | float]:
    """Flatten numeric engine statistics into API metric names."""
    flattened: dict[str, int | float] = {}
    for name, fields in (statistics or {}).items():
        if not isinstance(fields, dict):
            continue
        for field, value in fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            flattened[f"{name}.{field}"] = value
    return flattened


def _query_engine(run: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Build the API query-engine object."""
    engine_name = run["engine_name"]
    versions = run.get("versions", {})
    if engine_name == "polars-cpu":
        version = versions.get("polars") or ""
        commit = ""
    elif engine_name == "duckdb":
        version = versions.get("duckdb") or ""
        commit = ""
    else:
        cudf_polars = versions.get("cudf_polars") or {}
        if isinstance(cudf_polars, str):
            version = cudf_polars
            commit = ""
        else:
            version = cudf_polars.get("version") or ""
            commit = cudf_polars.get("commit") or ""

    result = {
        "engine_name": engine_name,
        "version": version,
        "commit_hash": commit,
    }
    for source, target in (
        ("identifier_hash", "identifier_hash"),
        ("container_image", "container_image"),
    ):
        if value := extra.get(source):
            result[target] = value
    return result


def _validation_status(query_logs: list[dict[str, Any]]) -> str:
    """Roll up per-query validation results."""
    statuses = [
        log["validation_result"]["status"]
        for log in query_logs
        if log.get("validation_result") is not None
    ]
    if "failed" in statuses:
        return "failed"
    if statuses and all(status == "passed" for status in statuses):
        return "passed"
    return "not-validated"


def _query_logs(
    run: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build query logs and return trace and plan sidecar rows."""
    query_logs: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    execution_order = 0
    saved_plans = run.get("plans", {})

    for query_id, records in run.get("records", {}).items():
        for record in records:
            iteration = record["iteration"]
            traces = record.get("traces")
            plan = _plan_from_traces(traces)
            if plan is None and (saved := saved_plans.get(str(query_id))) is not None:
                plan = _strip_scan_paths(saved)

            extra_info = {"iteration": iteration}
            if record.get("io_summaries") is not None:
                extra_info["io_summaries"] = record["io_summaries"]
            if record.get("statistics") is not None:
                extra_info["statistics"] = record["statistics"]
            if record.get("traceback") is not None:
                extra_info["traceback"] = record["traceback"]

            log: dict[str, Any] = {
                "query_name": str(query_id),
                "execution_order": execution_order,
                "status": record["status"],
                "extra_info": extra_info,
            }
            if record["status"] == "success":
                log["runtime_ms"] = record["duration"] * 1000
            if metrics := _flatten_statistics(record.get("statistics")):
                log["engine_metrics"] = metrics
            if validation := record.get("validation_result"):
                log["validation_result"] = {
                    **validation,
                    "status": validation["status"].lower(),
                }
            else:
                log["validation_result"] = None
            if plan is not None:
                log["plan_type"] = "cudf_polars"
                log["plan"] = plan
            query_logs.append(log)
            plan_rows.append(
                {
                    "query_id": str(query_id),
                    "iteration": iteration,
                    "plan": json.dumps(plan) if plan is not None else None,
                }
            )

            for trace in traces or ():
                if trace.get("scope") == "plan":
                    continue
                trace_rows.append(
                    {
                        **trace,
                        "query_id": str(query_id),
                        "iteration": iteration,
                    }
                )
            execution_order += 1

    return query_logs, trace_rows, plan_rows


def _relative_path(path: Path, base: Path) -> str:
    """Prefer a path relative to the JSONL directory."""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact(
    path: Path,
    base: Path,
    *,
    title: str,
    description: str,
    media_type: str,
) -> dict[str, str]:
    """Build one portable artifact manifest entry."""
    return {
        "path": _relative_path(path, base),
        "filename": path.name,
        "title": title,
        "description": description,
        "media_type": media_type,
        "scope": "run",
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Write sidecar rows with useful row-group statistics."""
    pl.DataFrame(rows).sort(["query_id", "iteration"]).write_parquet(
        path, statistics=True
    )


def build_report(
    run: dict[str, Any],
    *,
    output_path: Path,
    artifact_directory: Path | None = None,
) -> dict[str, Any]:
    """Convert a legacy in-memory benchmark run to schema version 2."""
    extra = dict(run.get("extra_info") or {})
    query_logs, trace_rows, plan_rows = _query_logs(run)
    hardware = run.get("hardware") or {}
    gpus = hardware.get("gpus") or []
    engine_name = run["engine_name"]

    diagnostic_extra = {
        key: value
        for key, value in {
            "frontend": run.get("frontend"),
            "dataset_path": run.get("dataset_path"),
            "scale_factor": run.get("scale_factor"),
            "iterations": run.get("iterations"),
            "validation_method": run.get("validation_method"),
            "versions": run.get("versions"),
            "hardware": hardware,
            "query_set": run.get("query_set"),
            "io_mode": run.get("io_mode"),
            **{
                key: value
                for key, value in extra.items()
                if key not in _TRANSPORT_FIELDS
            },
        }.items()
        if value is not None
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.get("run_id"),
        "cache_state": extra.get("cache_state")
        or {"lukewarm": "warm"}.get(run.get("io_mode"), run.get("io_mode")),
        "query_engine": _query_engine(run, extra),
        "run_at": run["timestamp"],
        "gpu_count": 0 if engine_name in {"polars-cpu", "duckdb"} else run["n_workers"],
        "query_logs": query_logs,
        "concurrency_streams": 1,
        "engine_config": run.get("config_options", {}),
        "extra_info": diagnostic_extra,
        "validation_status": _validation_status(query_logs),
        "command_line": run.get("command_line"),
        "roles": run.get("roles") or [],
        "labels": extra.get("labels") or [],
        "is_official": bool(extra.get("is_official", False)),
    }
    for key in _CATALOG_FIELDS:
        if extra.get(key) is not None:
            report[key] = extra[key]
    for key in ("startup_duration_ms", "shutdown_duration_ms"):
        if run.get(key) is not None:
            report[key] = run[key]
    if gpus:
        report["extra_info"]["gpu_name"] = gpus[0].get("name")
        report["extra_info"]["gpu_total_memory"] = gpus[0].get("total_memory")

    base = output_path.parent
    artifact_dir = artifact_directory or base / f"{output_path.stem}.assets"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(run.get("run_id") or "benchmark-run")
    artifacts: list[dict[str, str]] = []

    if trace_rows:
        trace_path = artifact_dir / f"{run_id}-traces.parquet"
        _write_parquet(trace_rows, trace_path)
        artifacts.append(
            _artifact(
                trace_path,
                base,
                title="IR node traces for benchmark run",
                description="cudf-polars execution traces.",
                media_type="application/vnd.apache.parquet",
            )
        )
    if any(row["plan"] is not None for row in plan_rows):
        plan_path = artifact_dir / f"{run_id}-plans.parquet"
        _write_parquet(plan_rows, plan_path)
        artifacts.append(
            _artifact(
                plan_path,
                base,
                title="Query plans",
                description="cudf-polars query plans by query iteration.",
                media_type="application/vnd.apache.parquet",
            )
        )

    raw_path = artifact_dir / f"{run_id}-raw.json"
    raw_path.write_text(json.dumps(report, indent=2, default=str))
    artifacts.append(
        _artifact(
            raw_path,
            base,
            title="Raw benchmark output",
            description="API-shaped cudf-polars benchmark output.",
            media_type="application/json",
        )
    )

    for path in _as_paths(extra.get("nsys-report")):
        artifacts.append(
            _artifact(
                path,
                base,
                title="Nsight Systems report",
                description="Nsight Systems profile report for this benchmark run.",
                media_type="application/octet-stream",
            )
        )
    if quent := extra.get("quent-archive"):
        artifacts.append(
            _artifact(
                Path(quent),
                base,
                title="Quent Archive",
                description="ZIP archive containing Quent traces.",
                media_type="application/zip",
            )
        )
    report["artifacts"] = artifacts
    return report


def _as_paths(value: Any) -> list[Path]:
    """Normalize an optional path or path list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [Path(value)]
    return [Path(path) for path in value]
