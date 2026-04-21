# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize resource-monitor traces from cudf-polars benchmark runs.

Reads one or more NDJSON log files (produced by ``--streaming-logs``),
filters for ``scope == "resource_monitor"`` records, and generates an
interactive HTML dashboard with Altair.

Usage::

    python plot_resource_monitor.py logs-*.ndjson -o dashboard.html

Each file may contain records from a different process (identified by
``pid``).  The tool extracts GPU device columns dynamically so it works
regardless of how many GPUs were present.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import altair as alt

import polars as pl


def _load_records(paths: list[Path]) -> pl.DataFrame:
    """Load resource_monitor records from one or more NDJSON log files."""
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("scope") == "resource_monitor":
                    record["source_file"] = path.name
                    rows.append(record)

    if not rows:
        print("No resource_monitor records found.", file=sys.stderr)
        sys.exit(1)

    return pl.DataFrame(rows)


def _gpu_indices(df: pl.DataFrame) -> list[int]:
    """Return sorted list of GPU indices present in the data."""
    pattern = re.compile(r"^gpu_(\d+)_memory_(used|total)$")
    indices = sorted(
        {int(m.group(1)) for col in df.columns if (m := pattern.match(col))}
    )
    return indices


def _resample(df: pl.DataFrame) -> pl.DataFrame:
    """Resample per-pid to 1-second bins via group_by_dynamic, averaging values."""
    gpu_indices = _gpu_indices(df)
    agg_exprs: list[pl.Expr] = [
        pl.col("host_cpu_percent").mean(),
        pl.col("process_rss").mean(),
        pl.col("host_memory_available").mean(),
        pl.col("host_memory_total").mean(),
        pl.col("host_memory_percent").mean(),
    ]
    for i in gpu_indices:
        for suffix in ("memory_used", "memory_total"):
            col = f"gpu_{i}_{suffix}"
            if col in df.columns:
                agg_exprs.append(pl.col(col).mean())

    return (
        df.sort("ts")
        .group_by_dynamic("ts", every="1s", period="1s", group_by="pid")
        .agg(agg_exprs)
    )


def _build_chart(df: pl.DataFrame) -> alt.VConcatChart:
    """Build a vertically-concatenated Altair chart."""
    # structlog's TimeStamper overwrites our numeric timestamp with a
    # formatted string like "2026-04-21 13:06:13.563153".
    ts = pl.col("timestamp")
    if df["timestamp"].dtype == pl.Utf8:
        ts = ts.str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S%.f")
    df = df.with_columns(ts.alias("ts"))

    df = _resample(df)

    t0 = df["ts"].min()
    df = df.with_columns(
        ((pl.col("ts") - t0).dt.total_microseconds() / 1e6).alias("elapsed_s"),
        (pl.col("process_rss") / 1e9).alias("process_rss_gb"),
        pl.col("pid").cast(pl.Utf8).alias("pid_str"),
    )

    gpu_indices = _gpu_indices(df)
    for i in gpu_indices:
        used_col = f"gpu_{i}_memory_used"
        if used_col in df.columns:
            df = df.with_columns(
                (pl.col(used_col) / 1e9).alias(f"gpu_{i}_used_gb"),
            )

    # Melt GPU used columns into long form for a single combined chart.
    gpu_used_cols = [
        f"gpu_{i}_used_gb" for i in gpu_indices if f"gpu_{i}_used_gb" in df.columns
    ]

    pdf = df.to_pandas()

    width = 700
    x = alt.X("elapsed_s:Q", title="Elapsed time (s)")
    color = alt.Color("pid_str:N", title="PID")

    cpu_chart = (
        alt.Chart(pdf, title="Host CPU %")
        .mark_line(point=True, size=1)
        .encode(x=x, y=alt.Y("host_cpu_percent:Q", title="CPU %"), color=color)
        .properties(width=width, height=200)
    )

    host_mem_chart = (
        alt.Chart(pdf, title="Process RSS")
        .mark_line(point=True, size=1)
        .encode(x=x, y=alt.Y("process_rss_gb:Q", title="RSS (GB)"), color=color)
        .properties(width=width, height=200)
    )

    charts = [cpu_chart, host_mem_chart]

    if gpu_used_cols:
        gpu_long = (
            df.select("elapsed_s", "pid_str", *gpu_used_cols)
            .unpivot(
                on=gpu_used_cols,
                index=["elapsed_s", "pid_str"],
                variable_name="gpu",
                value_name="used_gb",
            )
            .with_columns(
                pl.col("gpu").str.replace(r"gpu_(\d+)_used_gb", "GPU $1"),
                (pl.col("pid_str") + " / " + pl.col("gpu")).alias("pid_gpu"),
            )
        )
        gpu_pdf = gpu_long.to_pandas()

        gpu_chart = (
            alt.Chart(gpu_pdf, title="GPU memory used (per-process)")
            .mark_line(point=True, size=1)
            .encode(
                x=alt.X("elapsed_s:Q", title="Elapsed time (s)"),
                y=alt.Y("used_gb:Q", title="Used (GB)"),
                color=alt.Color("pid_gpu:N", title="PID / GPU"),
            )
            .properties(width=width, height=250)
        )
        charts.append(gpu_chart)

    return alt.vconcat(*charts)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize resource-monitor NDJSON traces as an HTML dashboard.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more NDJSON log files (e.g. logs-*.ndjson).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("resource_monitor.html"),
        help="Output HTML file (default: resource_monitor.html).",
    )
    args = parser.parse_args(argv)

    df = _load_records(args.files)
    chart = _build_chart(df)
    chart.save(str(args.output))
    print(f"Saved dashboard to {args.output}")


if __name__ == "__main__":
    main()
