# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid scan Parquet reader implementation."""

from __future__ import annotations

import concurrent.futures
import ctypes
import dataclasses
from typing import TYPE_CHECKING

import kvikio
import numpy as np

import pylibcudf as plc
import rmm

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from pylibcudf import expressions as plc_expr
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.typing import Schema


def _read_parquet_ranges_to_device_spans(
    path: str,
    ranges: list[plc.io.text.ByteRangeInfo],
    stream: Stream,
    mr: rmm.mr.PinnedHostMemoryResource,
) -> list[plc.gpumemoryview]:
    spans: list[plc.gpumemoryview] = []
    with kvikio.CuFile(path, "r") as file:
        for byte_range in ranges:
            ptr = mr.allocate(byte_range.size, stream)
            try:
                host_buffer = np.ctypeslib.as_array(
                    ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)),
                    shape=(byte_range.size,),
                )
                file.pread(
                    host_buffer,
                    size=byte_range.size,
                    file_offset=byte_range.offset,
                ).get()
                spans.append(
                    plc.gpumemoryview(rmm.DeviceBuffer.to_device(host_buffer, stream))
                )
            finally:
                mr.deallocate(ptr, byte_range.size, stream)
    return spans


def _hybrid_scan_apply_secondary_row_group_filters(
    path: str,
    hybrid_reader: plc.io.experimental.HybridScanReader,
    row_groups: list[int],
    per_file_options: plc.io.parquet.ParquetReaderOptions,
    stream: Stream,
    mr: rmm.mr.PinnedHostMemoryResource,
) -> list[int]:
    if not row_groups:
        return row_groups

    bloom_ranges, dict_ranges = hybrid_reader.secondary_filters_byte_ranges(
        row_groups, per_file_options
    )
    current = row_groups
    if dict_ranges and current:
        dict_data = _read_parquet_ranges_to_device_spans(path, dict_ranges, stream, mr)
        current = hybrid_reader.filter_row_groups_with_dictionary_pages(
            dict_data,  # type: ignore[arg-type]
            current,
            per_file_options,
            stream,
        )
    if bloom_ranges and current:
        bloom_data = _read_parquet_ranges_to_device_spans(
            path, bloom_ranges, stream, mr
        )
        current = hybrid_reader.filter_row_groups_with_bloom_filters(
            bloom_data,  # type: ignore[arg-type]
            current,
            per_file_options,
            stream,
        )
    return current


def _read_parquet_hybrid_single_file(
    schema: Schema,
    path: str,
    with_columns: list[str] | None,
    desired_order: list[str],
    filters: plc_expr.Expression | None,
    stream: Stream,
    footer_bytes: bytes,
    mr: rmm.mr.PinnedHostMemoryResource,
    *,
    row_groups: list[int] | None = None,
) -> DataFrame:
    from cudf_polars.dsl.ir import Scan

    per_file_options = Scan._build_parquet_reader_options(path, with_columns, filters)
    hybrid_reader = plc.io.experimental.HybridScanReader(footer_bytes, per_file_options)

    selected_row_groups = (
        list(row_groups)
        if row_groups is not None
        else hybrid_reader.all_row_groups(per_file_options)
    )
    if filters is not None:
        selected_row_groups = hybrid_reader.filter_row_groups_with_stats(
            selected_row_groups, per_file_options, stream
        )
        selected_row_groups = _hybrid_scan_apply_secondary_row_group_filters(
            path,
            hybrid_reader,
            selected_row_groups,
            per_file_options,
            stream,
            mr,
        )

    if not selected_row_groups:
        return Scan._empty_dataframe(schema, desired_order, stream)

    if filters is None:
        ranges = hybrid_reader.all_column_chunks_byte_ranges(
            selected_row_groups, per_file_options
        )
        column_data = _read_parquet_ranges_to_device_spans(path, ranges, stream, mr)
        table_w_meta = hybrid_reader.materialize_all_columns(
            selected_row_groups,
            column_data,  # type: ignore[arg-type]
            per_file_options,
            stream=stream,
        )
    else:
        filter_ranges = hybrid_reader.filter_column_chunks_byte_ranges(
            selected_row_groups, per_file_options
        )
        filter_data = _read_parquet_ranges_to_device_spans(
            path, filter_ranges, stream, mr
        )
        all_true = True
        row_mask = plc.Column.from_scalar(
            plc.Scalar.from_py(all_true, stream=stream),
            hybrid_reader.total_rows_in_row_groups(selected_row_groups),
            stream=stream,
        )
        filter_result = hybrid_reader.materialize_filter_columns(
            selected_row_groups,
            filter_data,  # type: ignore[arg-type]
            row_mask,
            plc.io.experimental.UseDataPageMask.NO,
            per_file_options,
            stream=stream,
        )

        payload_ranges = hybrid_reader.payload_column_chunks_byte_ranges(
            selected_row_groups, per_file_options
        )
        if payload_ranges:
            payload_data = _read_parquet_ranges_to_device_spans(
                path, payload_ranges, stream, mr
            )
            payload_result = hybrid_reader.materialize_payload_columns(
                selected_row_groups,
                payload_data,  # type: ignore[arg-type]
                row_mask,
                plc.io.experimental.UseDataPageMask.NO,
                per_file_options,
                stream=stream,
            )
            table_w_meta = plc.io.TableWithMetadata(
                plc.Table(
                    [*filter_result.tbl.columns(), *payload_result.tbl.columns()]
                ),
                filter_result.column_names(include_children=True)
                + payload_result.column_names(include_children=True),
            )
        else:
            table_w_meta = filter_result

    column_names = table_w_meta.column_names(include_children=False)
    if column_names:
        df = DataFrame.from_table(
            table_w_meta.tbl,
            column_names,
            [schema[name] for name in column_names],
            stream=stream,
        )
        available_order = [name for name in desired_order if name in df.column_map]
        if column_names != available_order:
            df = df.select(available_order)
        return df

    return Scan._empty_dataframe(
        schema,
        desired_order,
        stream,
        num_rows=hybrid_reader.total_rows_in_row_groups(selected_row_groups),
    )


@nvtx_annotate_cudf_polars(message="collect_parquet_metadata")
def collect_parquet_metadata(
    ir: IR,
    ir_execution_context: IRExecutionContext,
    *,
    pool: concurrent.futures.ThreadPoolExecutor | None = None,
) -> IRExecutionContext:
    """
    Return a new IR execution context with parquet metadata collected.

    This will traverse the IR graph, finding all parquet Scan nodes.

    It's safe to call ``result.parquet_metadata[tuple(paths)]`` for any ``paths``
    in the IR graph after calling this function.
    """
    from cudf_polars.dsl.ir import Scan

    pool = pool or concurrent.futures.ThreadPoolExecutor()

    parquet_metadata = {}
    futures = {}
    for node in traversal([ir]):
        if isinstance(node, Scan) and node.typ == "parquet":
            paths = tuple(node.paths)
            if paths not in ir_execution_context.parquet_metadata:
                # parquet_metadata[paths] = plc.io.parquet_metadata.read_parquet_metadata(
                #     plc.io.SourceInfo(paths)
                # )
                future = pool.submit(
                    plc.io.parquet_metadata.read_parquet_metadata,
                    plc.io.SourceInfo(paths),
                )
                futures[future] = paths
    for future in concurrent.futures.as_completed(futures):
        parquet_metadata[futures[future]] = future.result()
    return dataclasses.replace(ir_execution_context, parquet_metadata=parquet_metadata)
