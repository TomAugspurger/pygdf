#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Visualize query execution traces from benchmark JSONL output.

This script reads a JSONL benchmark results file and generates a formatted
query plan tree annotated with actual execution statistics from traces.

Usage:
    python visualize_traces.py pdsh_results.jsonl --query q1

Example output:
    SORT ('l_returnflag', 'l_linestatus') ('l_returnflag', '...', 'count_order') rows=4 chunks=1
      SELECT ('l_returnflag', '...', 'count_order') rows=4 chunks=1
        ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def format_row_count(value: int | None) -> str:
    """Format a row count as a readable string."""
    if value is None:
        return ""
    elif value < 1_000:
        return f"{value}"
    elif value < 1_000_000:
        return f"{round(value / 1_000, 2):g} K"
    elif value < 1_000_000_000:
        return f"{round(value / 1_000_000, 2):g} M"
    else:
        return f"{round(value / 1_000_000_000, 2):g} B"


def format_schema(schema: dict[str, str]) -> str:
    """Format schema as a tuple string, abbreviated if too long."""
    names = tuple(schema.keys())
    if len(names) > 6:
        names = names[:3] + ("...",) + names[-2:]
    return str(names).replace('"', "'")


def format_node_header(node: dict[str, Any]) -> str:
    """Format a node header based on its type and properties."""
    node_type = node["type"].upper()
    properties = node.get("properties", {})

    # Add type-specific info
    match node["type"]:
        case "GroupBy":
            keys = properties.get("keys", [])
            return f"GROUPBY {tuple(keys)}"
        case "Sort":
            by = properties.get("by", [])
            return f"SORT {tuple(by)}"
        case "Join":
            how = properties.get("how", "")
            left_on = properties.get("left_on", [])
            right_on = properties.get("right_on", [])
            return f"JOIN {how} {tuple(left_on)} {tuple(right_on)}"
        case "Filter":
            predicate = properties.get("predicate", "")
            return f"FILTER {predicate}"
        case "Scan":
            typ = properties.get("typ", "PARQUET").upper()
            return f"SCAN {typ}"
        case "SplitScan":
            return "SCAN PARQUET"
        case _:
            return node_type


def aggregate_traces(
    traces: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Aggregate trace data by ir_id.

    Returns a dict mapping ir_id to aggregated stats:
    - total_rows: sum of output rows across all chunks
    - chunk_count: number of trace events (chunks) for this node
    """
    aggregated: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"total_rows": 0, "chunk_count": 0}
    )

    for trace in traces:
        ir_id = trace["ir_id"]

        # Count chunks
        aggregated[ir_id]["chunk_count"] += 1

        # Sum rows from output frames
        frames_output = trace.get("frames_output", [])
        for frame in frames_output:
            shape = frame.get("shape", [0, 0])
            aggregated[ir_id]["total_rows"] += shape[0]

    return dict(aggregated)


def format_plan_tree(
    nodes: dict[str, dict[str, Any]],
    partition_info: dict[str, dict[str, Any]] | None,
    trace_stats: dict[int, dict[str, Any]],
    root_id: int,
    indent: str = "",
) -> str:
    """
    Recursively format the plan tree with trace statistics.

    Parameters
    ----------
    nodes
        Dict of node_id (string) -> node data.
    partition_info
        Dict of node_id (string) -> partition info, or None.
    trace_stats
        Dict of ir_id (int) -> aggregated trace stats.
    root_id
        The ID of the root node to start from.
    indent
        Current indentation string.

    Returns
    -------
    str
        Formatted tree representation.
    """
    node_id_str = str(root_id)
    node = nodes.get(node_id_str)

    if node is None:
        return f"{indent}??? (node {root_id} not found)\n"

    # Build the header line
    header = format_node_header(node)
    schema_str = format_schema(node.get("schema", {}))

    # Get stats from traces (using int id)
    stats = trace_stats.get(root_id, {})
    rows = stats.get("total_rows")
    chunks = stats.get("chunk_count")

    # If no trace stats, try partition_info for chunk count
    if chunks is None or chunks == 0:
        if partition_info and node_id_str in partition_info:
            chunks = partition_info[node_id_str].get("count")

    # Format the line
    line = f"{indent}{header} {schema_str}"
    if rows is not None and rows > 0:
        line += f" rows={format_row_count(rows)}"
    if chunks is not None and chunks > 0:
        line += f" chunks={chunks}"
    line += "\n"

    # Recurse to children
    children = node.get("children", [])
    for child_id in children:
        line += format_plan_tree(
            nodes, partition_info, trace_stats, child_id, indent + "  "
        )

    return line


def visualize_query(
    data: dict[str, Any],
    query_name: str,
    iteration: int = 0,
) -> str:
    """
    Generate a formatted query plan visualization.

    Parameters
    ----------
    data
        The parsed JSONL line containing records, plans, etc.
    query_name
        The query name (e.g., "1") to visualize.
    iteration
        Which iteration's traces to use (default 0).

    Returns
    -------
    str
        Formatted plan tree with trace statistics.
    """
    # Get the plan
    plans = data.get("plans", {})
    plan = plans.get(query_name)
    if plan is None:
        available = list(plans.keys())
        raise ValueError(f"Query '{query_name}' not found. Available: {available}")

    roots = plan.get("roots", [])
    nodes = plan.get("nodes", {})
    partition_info = plan.get("partition_info")

    if not roots:
        raise ValueError(f"No roots found in plan for query '{query_name}'")

    # Get traces for this query
    # Query number is extracted from query_name (e.g., "q1" -> "1")
    query_num = query_name.lstrip("q")
    records = data.get("records", {})
    query_records = records.get(query_num, [])

    traces: list[dict[str, Any]] = []
    if query_records and iteration < len(query_records):
        traces = query_records[iteration].get("traces", [])

    # Aggregate trace stats by ir_id
    trace_stats = aggregate_traces(traces)

    # Format the tree starting from each root
    output = ""
    for root_id in roots:
        output += format_plan_tree(nodes, partition_info, trace_stats, root_id)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize query execution traces from benchmark JSONL output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to the benchmark JSONL file",
    )
    parser.add_argument(
        "--query",
        "-q",
        default="1",
        help="Query name to visualize (default: 1)",
    )
    parser.add_argument(
        "--iteration",
        "-i",
        type=int,
        default=0,
        help="Iteration index to use for traces (default: 0)",
    )
    parser.add_argument(
        "--line",
        "-l",
        type=int,
        default=0,
        help="Line number in JSONL file (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    # Read the JSONL file
    if not args.jsonl_file.exists():
        print(f"Error: File not found: {args.jsonl_file}", file=sys.stderr)
        return 1

    with open(args.jsonl_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    if args.line >= len(lines):
        print(
            f"Error: Line {args.line} not found (file has {len(lines)} lines)",
            file=sys.stderr,
        )
        return 1

    data = json.loads(lines[args.line])

    try:
        output = visualize_query(data, args.query, args.iteration)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.output:
        args.output.write_text(output)
        print(f"Output written to {args.output}")
    else:
        print(output, end="")

    return 0


if __name__ == "__main__":
    sys.exit(main())
