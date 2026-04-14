# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Basic POC for hybrid scan + remote I/O.
#
# TODO:
# - [ ] Column projection
# - [ ] Row group pruning
# - [ ] Demo overlapping I/O (CUDA streams, multiple tables)
# - [ ] concurrent reads
# - [ ] request coalescing
# - [ ] Optimistically coalesce the footer read with the suffix read
# - [ ] rapidsmpf / rmm pinned memory resource

import argparse
import concurrent.futures
import io

import boto3
import botocore.config
import cupyx
import kvikio
import kvikio.defaults
import numpy as np
import nvtx
import pylibcudf as plc
import rmm.mr

DOMAIN = "POC"

TABLE_OBJECTS = {
    "lineitem": "scale-10/lineitem/part.0.parquet",
    "nation": "scale-10/nation/part.0.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid scan + remote I/O POC"
    )
    parser.add_argument(
        "--table",
        choices=list(TABLE_OBJECTS),
        default="lineitem",
        help="TPC-H table Parquet object to read (default: %(default)s)",
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download the whole file into memory for the baseline read_plain "
        "(default: stream from URL without a full download)",
    )
    return parser.parse_args()


def read_plain(
    url: str | io.BytesIO,
    filter_expression: plc.expressions.Operation,
    stream: rmm.pylibrmm.stream.Stream,
    mr: rmm.mr.DeviceMemoryResource,
) -> plc.io.TableWithMetadata:
    reader_options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([url])
    ).build()
    reader_options.set_filter(filter_expression)
    t = plc.io.parquet.read_parquet(
        reader_options,
        stream=stream,
        mr=mr,
    )
    return t


def read_hybrid_on_demand(
    url: str,
    filter_expression: plc.expressions.Operation | None,
    stream: rmm.pylibrmm.stream.Stream,
    mr: rmm.mr.DeviceMemoryResource,
    content_length: int,
    request_pool: concurrent.futures.ThreadPoolExecutor,
    footer_size_hint: int | None = None,
) -> plc.io.TableWithMetadata:
    # Get the content-length with a size-0 GET request
    # response = httpx.get(url, headers={"Range": "bytes=0-0"})
    # response.raise_for_status()
    # nbytes = int(response.headers["content-range"].split("/")[1])
    # Note: we might be able to get the content length from the `content-range`
    # header of the initial request to read the last 8 bytes?

    nbytes = content_length
    with nvtx.annotate("open_remote_file", domain=DOMAIN):
        remote_file = kvikio.RemoteFile.open_s3_presigned_url(url, nbytes)

    if footer_size_hint is None:
        # pinned_pool = cupy.get_default_pinned_memory_pool()
        # Allocate things on demand.
        suffix_len = 8
        with nvtx.annotate("alloc_suffix", domain=DOMAIN):
            suffix_buffer = cupyx.empty_pinned(suffix_len, dtype=np.uint8)

        with nvtx.annotate("read_suffix", domain=DOMAIN):
            remote_file.read(
                suffix_buffer, suffix_len, file_offset=nbytes - suffix_len
            )

        # IO-2: Read the footer bytes.
        footer_len = int.from_bytes(suffix_buffer[:-4], byteorder="little")
        with nvtx.annotate("alloc_footer", domain=DOMAIN):
            footer_buffer = cupyx.empty_pinned(
                footer_len - suffix_len, dtype=np.uint8
            )

        with nvtx.annotate("read_footer", domain=DOMAIN):
            remote_file.read(
                footer_buffer,
                footer_len - suffix_len,
                file_offset=nbytes - (footer_len + suffix_len),
            )

    else:
        raise NotImplementedError("Not implemented")
        # assert footer_size_hint > 8
        # with nvtx.annotate("alloc_footer_guess", domain=DOMAIN):
        #     footer_buffer = cupyx.empty_pinned(footer_size_hint, dtype=np.uint8)

        # with nvtx.annotate("read_footer_guess", domain=DOMAIN):
        #     remote_file.read(
        #         footer_buffer,
        #         footer_size_hint,
        #         file_offset=nbytes - footer_size_hint,
        #     )

        # footer_len = int.from_bytes(footer_buffer[-8:-4], byteorder="little")
        # if footer_len > footer_size_hint:
        #     # TODO: fallback to reading more.
        #     raise ValueError(f"Footer length {footer_len} is greater than the hint {footer_size_hint}")
        # footer_buffer = footer_buffer[-footer_len:-8]
        # print(f"{footer_len=}")
        # print(f"{footer_buffer.shape=}")
        # print(f"{footer_buffer=}")

    # TODO: figure out what to put here.
    per_file_options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([io.BytesIO()])
    ).build()
    if filter_expression is not None:
        per_file_options.set_filter(filter_expression)

    # TODO: see byte range filter? I don't think polars exposes that in any way.

    # TODO: memoryview on footer_bytes
    hybrid_reader = plc.io.experimental.HybridScanReader(
        footer_buffer.tobytes(), per_file_options
    )

    def read_column_chunk(
        r: plc.io.text.ByteRangeInfo, stage="filter"
    ) -> None:
        with nvtx.annotate(f"read_{stage}_column_chunk", domain=DOMAIN):
            with nvtx.annotate("alloc_row_group_buffer", domain=DOMAIN):
                row_group_buffer = cupyx.empty_pinned(r.size, dtype=np.uint8)
            with nvtx.annotate("read_row_group_buffer", domain=DOMAIN):
                remote_file.read(
                    row_group_buffer, r.size, file_offset=r.offset
                )
            return row_group_buffer

    # TODO: log stats selectivity
    # TODO: Filter row groups with byte range
    row_groups = hybrid_reader.all_row_groups(per_file_options)
    row_groups = hybrid_reader.filter_row_groups_with_stats(
        row_groups, per_file_options, stream=stream
    )
    # TODO: Filter row groups based on "secondary filters" (dictionaries, bloom filters)

    if filter_expression is not None:
        filter_ranges = hybrid_reader.filter_column_chunks_byte_ranges(
            row_groups, per_file_options
        )

        column_data = []
        row_group_buffers = []

        futures = [
            request_pool.submit(read_column_chunk, r, stage="filter")
            for r in filter_ranges
        ]
        with nvtx.annotate("filter-barrier", domain=DOMAIN):
            for future in futures:
                row_group_buffers.append(future.result())

        for r, row_group_buffer in zip(
            filter_ranges, row_group_buffers, strict=True
        ):
            with nvtx.annotate("convert_to_gpu_memoryview", domain=DOMAIN):
                column_data.append(
                    plc.gpumemoryview(
                        rmm.DeviceBuffer.to_device(row_group_buffer, stream)
                    )
                )

        # TODO: when can we free row_group_buffers?

        # We need an all true row mask for the filter columns.
        with nvtx.annotate("filter mask", domain=DOMAIN):
            n_rows = hybrid_reader.total_rows_in_row_groups(row_groups)
            # n.b. this row_mask is mutated inplace by materialize_filter_columns
            row_mask = plc.Column.from_scalar(
                plc.Scalar.from_py(True, stream=stream),
                n_rows,
                stream=stream,
                mr=mr,
            )
            filter_t = hybrid_reader.materialize_filter_columns(
                row_groups,
                column_data,
                row_mask,
                plc.io.experimental.UseDataPageMask.NO,
                per_file_options,
                stream=stream,
            )

        # This actually does prune one row group, somehow.
        # TODO: figure how how
        # I wonder if it's because it doesn't actually read the filter column.
        # So if you want the filter column, how do you do that?
        with nvtx.annotate("payload-ranges", domain=DOMAIN):
            payload_ranges = hybrid_reader.payload_column_chunks_byte_ranges(
                row_groups, per_file_options
            )

        payload_buffers = []
        payload_data = []
        futures = [
            request_pool.submit(read_column_chunk, r, stage="payload")
            for r in payload_ranges
        ]
        with nvtx.annotate("payload-barrier", domain=DOMAIN):
            for future in futures:
                payload_buffers.append(future.result())

        for r, payload_buffer in zip(
            payload_ranges, payload_buffers, strict=True
        ):
            with nvtx.annotate("convert_to_gpu_memoryview", domain=DOMAIN):
                payload_data.append(
                    plc.gpumemoryview(
                        rmm.DeviceBuffer.to_device(payload_buffer, stream)
                    )
                )

        with nvtx.annotate("materialize_payload_columns", domain=DOMAIN):
            payload_t = hybrid_reader.materialize_payload_columns(
                row_groups,
                payload_data,
                row_mask,
                plc.io.experimental.UseDataPageMask.NO,
                per_file_options,
                stream=stream,
            )

        # Now make a new table with both the filter and payload columns
        payload_t = plc.io.TableWithMetadata(
            plc.Table(filter_t.columns + payload_t.columns),
            filter_t.column_names(include_children=True)
            + payload_t.column_names(include_children=True),
        )

    else:
        with nvtx.annotate("payload-ranges", domain=DOMAIN):
            payload_ranges = hybrid_reader.all_column_chunks_byte_ranges(
                row_groups, per_file_options
            )

        payload_buffers = []
        payload_data = []
        futures = [
            request_pool.submit(read_column_chunk, r, stage="payload")
            for r in payload_ranges
        ]
        with nvtx.annotate("payload-barrier", domain=DOMAIN):
            for future in futures:
                payload_buffers.append(future.result())

        for r, payload_buffer in zip(
            payload_ranges, payload_buffers, strict=True
        ):
            with nvtx.annotate("convert_to_gpu_memoryview", domain=DOMAIN):
                payload_data.append(
                    plc.gpumemoryview(
                        rmm.DeviceBuffer.to_device(payload_buffer, stream)
                    )
                )

        payload_t = hybrid_reader.materialize_all_columns(
            row_groups, payload_data, per_file_options, stream=stream, mr=mr
        )

    return payload_t


def main():
    args = parse_args()
    table = args.table
    memory = args.memory

    with nvtx.annotate("setup", domain=DOMAIN):
        stream = rmm.pylibrmm.stream.Stream()
        mr = rmm.mr.CudaAsyncMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        kvikio.defaults.set("num_threads", 64)
        request_pool = concurrent.futures.ThreadPoolExecutor(max_workers=64)

        bucket_name = "rapids-tpch"
        object_name = TABLE_OBJECTS[table]
        region_name = "us-east-2"
        expiration = 3600
        if table == "lineitem":
            # TODO: write a selective filter
            # For `scale-10/lineitem/part.0.parquet`, we could use the expression
            # `l_orderkey < 999939` (or <= maybe; apparently the value 999939 appears
            # in the first and second row groups).
            # Use 999930 to also filter some rows, not just row groups.
            l_orderkey = plc.Scalar.from_py(
                999930,
                stream=stream,
                dtype=plc.types.DataType(plc.types.TypeId.INT64),
            )
            filter_expression = plc.expressions.Operation(
                plc.expressions.ASTOperator.LESS,
                plc.expressions.ColumnNameReference("l_orderkey"),
                plc.expressions.Literal(l_orderkey),
            )
        elif table == "nation":
            region_key = plc.Scalar.from_py(
                0,
                stream=stream,
                dtype=plc.types.DataType(plc.types.TypeId.INT32),
            )
            filter_expression = plc.expressions.Operation(
                plc.expressions.ASTOperator.EQUAL,
                plc.expressions.ColumnNameReference("n_regionkey"),
                plc.expressions.Literal(region_key),
            )
        else:
            raise ValueError(f"Unknown table: {table}")

        s3_client = boto3.client(
            "s3",
            region_name=region_name,
            config=botocore.config.Config(signature_version="s3v4"),
        )
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration,
        )

        content_length = s3_client.head_object(
            Bucket=bucket_name, Key=object_name
        )["ContentLength"]

    if memory:
        sink = io.BytesIO()
        with nvtx.annotate("read-remote-boto3", domain=DOMAIN):
            s3_client.download_fileobj(
                Bucket=bucket_name, Key=object_name, Fileobj=sink
            )
        sink.seek(0)
        with nvtx.annotate("read-host", domain=DOMAIN):
            with nvtx.annotate("read", domain=DOMAIN):
                expected = read_plain(sink, filter_expression, stream, mr)
            with nvtx.annotate("synchronize", domain=DOMAIN):
                stream.synchronize()

        with nvtx.annotate("read-pinned", domain=DOMAIN):
            with nvtx.annotate("alloc", domain=DOMAIN):
                buf = cupyx.empty_pinned(content_length, dtype=np.uint8)
                buf[:] = np.frombuffer(sink.getvalue(), dtype=np.uint8)

    with nvtx.annotate("read-remote-url", domain=DOMAIN):
        with nvtx.annotate("read", domain=DOMAIN):
            expected = read_plain(url, filter_expression, stream, mr)
        with nvtx.annotate("synchronize", domain=DOMAIN):
            stream.synchronize()

    with nvtx.annotate("read-remote-hybrid", domain=DOMAIN):
        with nvtx.annotate("read", domain=DOMAIN):
            result_on_demand = read_hybrid_on_demand(
                url,
                filter_expression,
                stream,
                mr,
                content_length,
                request_pool,
            )
        with nvtx.annotate("synchronize", domain=DOMAIN):
            stream.synchronize()

    # Validate that they match with pyarrow
    expected_columns = expected.column_names()
    result_columns = result_on_demand.column_names()
    a = expected.tbl.to_arrow()
    # TODO: reorder in the reader.
    c = result_on_demand.tbl.to_arrow()

    indices = [result_columns.index(c) for c in expected_columns]
    c = c.select(indices)

    try:
        assert a.equals(c)
    except AssertionError:
        print("Hybrid on demand result does not match expected result")


if __name__ == "__main__":
    main()
