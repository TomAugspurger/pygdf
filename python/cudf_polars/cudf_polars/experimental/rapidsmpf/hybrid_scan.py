# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid-scan parquet metadata helpers for the RapidsMPF runtime."""

from __future__ import annotations

import dataclasses
import io
import os
import queue
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cupy as cp
import kvikio
import nvtx

import pylibcudf as plc
import rmm

from cudf_polars.dsl import tracing as dsl_tracing

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping

    from cudf_polars.dsl.ir import IR, IRExecutionContext, Scan
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.utils.config import ParquetOptions

__all__ = [
    "CachedParquetFileMetadata",
    "CachedParquetMetadata",
    "HybridScanReadResult",
    "get_cached_parquet_metadata",
    "make_cached_parquet_file_metadata",
    "make_cached_parquet_metadata",
    "parquet_metadata_from_footer_bytes",
    "parquet_metadata_from_footer_bytes_many",
    "populate_parquet_metadata_cache",
    "read_parquet_with_hybrid_scan",
    "split_row_groups_for_partition",
]


_PARQUET_MAGIC = b"PAR1"
_DEFAULT_FOOTER_PREFETCH_WORKERS = 64
_DEFAULT_READ_READY_QUEUE_SIZE = 2


def _log_io_event(message: str, *, scope: str, **kwargs: Any) -> None:
    """Emit a structured IO event via structlog tracing."""
    dsl_tracing.log(message, category="IO", scope=scope, **kwargs)


@dataclasses.dataclass(frozen=True)
class CachedParquetFileMetadata:
    """Cached metadata for one parquet file."""

    path: str
    file_size: int
    footer_bytes: bytes
    hybrid_metadata: plc.io.experimental.hybrid_scan.FileMetaData


@dataclasses.dataclass(frozen=True)
class CachedParquetMetadata:
    """Query-lifetime parquet metadata cached for one Scan node's paths."""

    files: tuple[CachedParquetFileMetadata, ...]
    parquet_metadata: plc.io.parquet_metadata.ParquetMetadata

    @property
    def paths(self) -> tuple[str, ...]:
        """Return the scan paths this metadata describes."""
        return tuple(file.path for file in self.files)

    def file(self, path: str) -> CachedParquetFileMetadata:
        """Return cached metadata for *path*."""
        for file_metadata in self.files:
            if file_metadata.path == path:
                return file_metadata
        raise KeyError(path)


@dataclasses.dataclass(frozen=True)
class HybridScanReadResult:
    """Result of a single-step hybrid parquet scan."""

    table: plc.Table | None
    column_names: list[str]
    rows_per_source: list[int]


@dataclasses.dataclass(frozen=True)
class _CoalescedRange:
    """A merged byte-range block to read from storage."""

    offset: int
    size: int


@dataclasses.dataclass(frozen=True)
class _RangeSlice:
    """Mapping from original range index to a coalesced block slice."""

    block_index: int
    block_offset: int
    size: int


@dataclasses.dataclass(frozen=True)
class _ReadRangesResult:
    """Result payload for range reads plus perf counters."""

    column_data: list[plc.gpumemoryview]
    original_range_count: int
    coalesced_range_count: int
    requested_bytes: int
    read_bytes: int
    backing_refs: tuple[Any, ...]


@dataclasses.dataclass
class _PreparedReadRanges:
    """Prepared per-path read plan used for flattened IO scheduling."""

    path: str
    file_size: int | None
    byte_ranges: list[Any]
    coalesced_ranges: list[_CoalescedRange]
    slices: list[_RangeSlice]
    targets: list[Any]
    owners: list[Any]
    requested_bytes: int
    read_bytes: int
    max_workers: int
    max_coalesce_gap: int
    sync_before_read: bool
    use_slab_allocation: bool
    started_ns: int


@dataclasses.dataclass
class _PathScanPlan:
    """Per-path scan plan for two-phase read/materialize execution."""

    path: str
    reader: Any
    options: plc.io.parquet.ParquetReaderOptions
    row_groups: list[Any]
    prepared_reads: _PreparedReadRanges | None


def _footer_as_parquet_buffer(footer_bytes: bytes) -> io.BytesIO:
    """Wrap footer bytes in a minimal in-memory parquet file."""
    if not footer_bytes:
        raise ValueError("Parquet footer bytes must not be empty")
    buf = io.BytesIO()
    buf.write(_PARQUET_MAGIC)
    buf.write(footer_bytes)
    buf.write(len(footer_bytes).to_bytes(4, "little"))
    buf.write(_PARQUET_MAGIC)
    buf.seek(0)
    return buf


def parquet_metadata_from_footer_bytes(
    footer_bytes: bytes,
) -> plc.io.parquet_metadata.ParquetMetadata:
    """Create pylibcudf parquet metadata from one raw parquet footer payload."""
    return parquet_metadata_from_footer_bytes_many((footer_bytes,))


def parquet_metadata_from_footer_bytes_many(
    footers: Iterable[bytes],
) -> plc.io.parquet_metadata.ParquetMetadata:
    """Create aggregate pylibcudf parquet metadata from raw footer payloads."""
    buffers = [_footer_as_parquet_buffer(footer) for footer in footers]
    if not buffers:
        raise ValueError("At least one parquet footer is required")
    # pylibcudf accepts BytesIO sources at runtime, but the SourceInfo stub
    # currently omits them.
    source_info = plc.io.SourceInfo(buffers)  # type: ignore[arg-type]
    return plc.io.parquet_metadata.read_parquet_metadata(source_info)


def make_cached_parquet_file_metadata(
    path: str,
    file_size: int,
    footer_bytes: bytes,
    options: plc.io.parquet.ParquetReaderOptions,
) -> CachedParquetFileMetadata:
    """Build cached metadata for one parquet file from prefetched footer bytes."""
    reader = plc.io.experimental.hybrid_scan.HybridScanReader(footer_bytes, options)
    return CachedParquetFileMetadata(
        path=path,
        file_size=file_size,
        footer_bytes=footer_bytes,
        hybrid_metadata=reader.parquet_metadata(),
    )


def make_cached_parquet_metadata(
    files: Iterable[CachedParquetFileMetadata],
) -> CachedParquetMetadata:
    """Build scan-level cached metadata from per-file cached metadata."""
    file_metadata = tuple(files)
    if not file_metadata:
        raise ValueError("At least one parquet file metadata object is required")
    return CachedParquetMetadata(
        files=file_metadata,
        parquet_metadata=parquet_metadata_from_footer_bytes_many(
            file.footer_bytes for file in file_metadata
        ),
    )


def get_cached_parquet_metadata(
    paths: list[str],
    context: IRExecutionContext,
) -> CachedParquetMetadata:
    """Return cached parquet metadata for *paths*, composing single-file entries."""
    key = tuple(paths)
    if key in context.parquet_metadata:
        return context.parquet_metadata[key]

    files: list[CachedParquetFileMetadata] = []
    missing: list[str] = []
    for path in paths:
        single_file_metadata = context.parquet_metadata.get((path,))
        if single_file_metadata is None:
            missing.append(path)
        else:
            files.append(single_file_metadata.file(path))
    if missing:
        raise KeyError(
            "Missing hybrid-scan parquet metadata for paths: "
            + ", ".join(map(str, missing))
        )

    metadata = make_cached_parquet_metadata(files)
    context.parquet_metadata[key] = metadata
    return metadata


@nvtx.annotate("read_exact", domain="cudf_polars", color="red")
def _read_exact(file_handle: Any, size: int, offset: int) -> bytes:
    """Read exactly *size* bytes from *file_handle* at *offset*."""
    buf = bytearray(size)
    nread = file_handle.pread(buf, size=size, file_offset=offset).get()
    if nread != size:
        raise OSError(f"Expected to read {size} bytes, got {nread}")
    return bytes(buf)


@nvtx.annotate("open_file", domain="cudf_polars", color="red")
def _open_file(
    path: str, *, nbytes: int | None = None
) -> kvikio.RemoteFile | kvikio.CuFile:
    """Open *path* for footer reads using kvikio."""
    if plc.io.SourceInfo._is_remote_uri(path):
        scheme = urllib.parse.urlparse(path).scheme.lower()
        if scheme == "s3":
            return kvikio.RemoteFile.open_s3_url(
                path,
                nbytes=nbytes if nbytes is not None else _s3_object_size(path),
                aws_region_name=os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION"),
            )
        if scheme in {"http", "https"}:
            return kvikio.RemoteFile.open(
                path,
                nbytes=nbytes if nbytes is not None else _http_object_size(path),
            )
        return kvikio.RemoteFile.open(path, nbytes=nbytes)
    return kvikio.CuFile(path, "r")


def _object_size_from_content_range(content_range: str) -> int | None:
    """Extract the object size from an HTTP Content-Range header."""
    _, _, total = content_range.partition("/")
    if not total or total == "*":
        return None
    return int(total)


def _http_object_size(path: str) -> int | None:
    """Return object size using an HTTP range GET instead of HEAD."""
    request = urllib.request.Request(
        path,
        headers={"Range": "bytes=0-0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request) as response:
            if content_range := response.headers.get("Content-Range"):
                return _object_size_from_content_range(content_range)
            if getattr(response, "status", None) == 206:
                return None
            if content_length := response.headers.get("Content-Length"):
                return int(content_length)
    except urllib.error.URLError:
        return None
    return None


def _s3_object_size(path: str) -> int | None:
    """Return the S3 object size from boto3 when available."""
    parsed = urllib.parse.urlparse(path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        return None
    try:
        import boto3
    except ModuleNotFoundError:
        return None

    client = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION"),
    )
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except Exception:
        try:
            response = client.get_object(Bucket=bucket, Key=key, Range="bytes=0-0")
        except Exception:
            return None
        try:
            if content_range := response.get("ContentRange"):
                return _object_size_from_content_range(content_range)
        finally:
            response["Body"].close()
        return None
    return int(response["ContentLength"])


def _file_size(path: str, file_handle: Any) -> int:
    """Return the file size for a local or remote kvikio handle."""
    if plc.io.SourceInfo._is_remote_uri(path):
        return file_handle.nbytes()
    return Path(path).stat().st_size


def _read_footer_bytes(path: str) -> tuple[int, bytes]:
    """Read parquet footer payload bytes for *path*."""
    file_handle = _open_file(path)
    try:
        file_size = _file_size(path, file_handle)
        if file_size < 8:
            raise ValueError(f"Parquet file is too small: {path}")
        suffix = _read_exact(file_handle, 8, file_size - 8)
        footer_len = struct.unpack("<I", suffix[:4])[0]
        if suffix[4:] != _PARQUET_MAGIC:
            raise ValueError(f"Invalid parquet footer magic in {path}")
        footer_start = file_size - footer_len - 8
        if footer_start < 4:
            raise ValueError(f"Invalid parquet footer length in {path}: {footer_len}")
        footer_bytes = _read_exact(file_handle, footer_len, footer_start)
        return file_size, footer_bytes
    finally:
        file_handle.close()


def _reader_options_for_path(path: str) -> plc.io.parquet.ParquetReaderOptions:
    """Build metadata-reader options for one parquet file."""
    return (
        plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([path]))
        .decimal_width(plc.TypeId.DECIMAL128)
        .build()
    )


def _fetch_cached_parquet_file_metadata(path: str) -> CachedParquetFileMetadata:
    """Fetch and materialize cached metadata for one parquet file."""
    start = time.perf_counter_ns()
    file_size, footer_bytes = _read_footer_bytes(path)
    cached = make_cached_parquet_file_metadata(
        path,
        file_size,
        footer_bytes,
        _reader_options_for_path(path),
    )
    end = time.perf_counter_ns()
    _log_io_event(
        "Hybrid Scan Metadata Cache Entry",
        scope="metadata",
        path=path,
        start=start,
        duration=end - start,
        file_size=file_size,
        footer_size=len(footer_bytes),
    )
    return cached


def _reader_options_for_scan(
    path: str,
    column_names: list[str] | None,
    filter_expr: Any | None,
) -> plc.io.parquet.ParquetReaderOptions:
    """Build per-file parquet reader options for hybrid scan materialization."""
    options = (
        plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([path]))
        .decimal_width(plc.TypeId.DECIMAL128)
        .build()
    )
    if column_names is not None:
        options.set_column_names(column_names)
    if filter_expr is not None:
        options.set_filter(filter_expr)
    return options


def _coalesce_ranges(
    byte_ranges: list[Any],
    *,
    max_gap: int = 0,
) -> tuple[list[_CoalescedRange], list[_RangeSlice]]:
    """Coalesce sorted-overlap ranges and map originals back to merged blocks."""
    if not byte_ranges:
        return [], []

    indexed = [
        (idx, int(byte_range.offset), int(byte_range.size))
        for idx, byte_range in enumerate(byte_ranges)
    ]
    slices: list[_RangeSlice | None] = [None] * len(indexed)
    coalesced: list[_CoalescedRange] = []

    current_start: int | None = None
    current_end = 0
    current_origins: list[tuple[int, int, int]] = []

    def flush_block() -> None:
        nonlocal current_start, current_end, current_origins
        if current_start is None:
            return
        block_index = len(coalesced)
        coalesced.append(
            _CoalescedRange(offset=current_start, size=current_end - current_start)
        )
        for original_index, original_offset, original_size in current_origins:
            slices[original_index] = _RangeSlice(
                block_index=block_index,
                block_offset=original_offset - current_start,
                size=original_size,
            )
        current_start = None
        current_end = 0
        current_origins = []

    for original_index, offset, size in sorted(indexed, key=lambda item: item[1]):
        if size == 0:
            slices[original_index] = _RangeSlice(block_index=-1, block_offset=0, size=0)
            continue
        end = offset + size
        if current_start is None:
            current_start = offset
            current_end = end
            current_origins.append((original_index, offset, size))
            continue
        if offset <= current_end + max_gap:
            current_end = max(current_end, end)
            current_origins.append((original_index, offset, size))
            continue
        flush_block()
        current_start = offset
        current_end = end
        current_origins.append((original_index, offset, size))

    flush_block()
    remapped = [item for item in slices if item is not None]
    if len(remapped) != len(slices):  # pragma: no cover
        raise AssertionError("Missing byte-range mapping after coalescing")
    return coalesced, remapped


def _cupy_uint8_view(owner: Any, ptr: int, size: int) -> Any:
    """Create a uint8 GPU array view over *owner* memory."""
    memory = cp.cuda.UnownedMemory(ptr, size, owner)
    pointer = cp.cuda.MemoryPointer(memory, 0)
    return cp.ndarray((size,), dtype=cp.uint8, memptr=pointer)


@nvtx.annotate("allocate_coalesced_targets", domain="cudf_polars", color="red")
def _allocate_coalesced_targets(
    coalesced_ranges: list[_CoalescedRange],
    stream: Any,
    *,
    use_slab_allocation: bool,
) -> tuple[list[Any], list[Any]]:
    """Allocate GPU targets for each coalesced range."""
    if not coalesced_ranges:
        return [], []

    if use_slab_allocation:
        total_bytes = sum(block.size for block in coalesced_ranges)
        slab = rmm.DeviceBuffer(size=total_bytes, stream=stream)
        targets: list[Any] = []
        owners: list[Any] = [slab]
        cursor = 0
        for block in coalesced_ranges:
            view = _cupy_uint8_view(slab, slab.ptr + cursor, block.size)
            targets.append(view)
            owners.append(view)
            cursor += block.size
        return targets, owners

    targets = []
    owners = []
    for block in coalesced_ranges:
        buffer = rmm.DeviceBuffer(size=block.size, stream=stream)
        view = _cupy_uint8_view(buffer, buffer.ptr, block.size)
        targets.append(view)
        owners.extend((buffer, view))
    return targets, owners


def split_row_groups_for_partition(
    total_row_groups: int,
    split_index: int,
    total_splits: int,
) -> list[int]:
    """Return row-group indices assigned to one split partition."""
    if total_splits <= 0:  # pragma: no cover
        raise ValueError("total_splits must be positive")
    if split_index < 0 or split_index >= total_splits:  # pragma: no cover
        raise ValueError("split_index out of range")
    if total_row_groups <= 0:
        return []

    base = total_row_groups // total_splits
    remainder = total_row_groups % total_splits
    count = base + (1 if split_index < remainder else 0)
    start = split_index * base + min(split_index, remainder)
    return list(range(start, start + count))


def _read_byte_ranges_to_device(
    path: str,
    byte_ranges: list[Any],
    stream: Any,
    *,
    file_size: int | None = None,
    max_coalesce_gap: int = 0,
    max_workers: int = 128,
    sync_before_read: bool = False,
    use_slab_allocation: bool = False,
) -> _ReadRangesResult:
    """Read byte ranges from *path* into device buffers."""
    prepared_reads = _prepare_read_ranges_to_device(
        path,
        byte_ranges,
        stream,
        file_size=file_size,
        max_coalesce_gap=max_coalesce_gap,
        max_workers=max_workers,
        sync_before_read=sync_before_read,
        use_slab_allocation=use_slab_allocation,
    )
    for ready in _iter_prepared_reads_as_ready(
        (prepared_reads,),
        max_workers=max_workers,
    ):
        return _finalize_prepared_range_reads(ready, stream=stream)
    raise RuntimeError("Expected prepared reads to be available")


def _prepare_read_ranges_to_device(
    path: str,
    byte_ranges: list[Any],
    stream: Any,
    *,
    file_size: int | None = None,
    max_coalesce_gap: int = 0,
    max_workers: int = 128,
    sync_before_read: bool = False,
    use_slab_allocation: bool = False,
) -> _PreparedReadRanges:
    """Prepare one path's coalesced read plan without issuing reads yet."""
    coalesce_start = time.perf_counter_ns()
    requested_bytes = sum(int(byte_range.size) for byte_range in byte_ranges)
    coalesced_ranges, slices = _coalesce_ranges(byte_ranges, max_gap=max_coalesce_gap)
    read_bytes = sum(block.size for block in coalesced_ranges)
    coalesce_end = time.perf_counter_ns()
    _log_io_event(
        "Hybrid Scan Byte Range Coalescing",
        scope="Coalesce",
        path=path,
        byte_range_count=len(byte_ranges),
        coalesced_byte_range_count=len(coalesced_ranges),
        total_bytes=requested_bytes,
        coalesced_total_bytes=read_bytes,
        timestamp=coalesce_start,
        duration=coalesce_end - coalesce_start,
    )
    if sync_before_read:
        stream.synchronize()
    targets, owners = _allocate_coalesced_targets(
        coalesced_ranges, stream, use_slab_allocation=use_slab_allocation
    )
    return _PreparedReadRanges(
        path=path,
        file_size=file_size,
        byte_ranges=byte_ranges,
        coalesced_ranges=coalesced_ranges,
        slices=slices,
        targets=targets,
        owners=owners,
        requested_bytes=requested_bytes,
        read_bytes=read_bytes,
        max_workers=max_workers,
        max_coalesce_gap=max_coalesce_gap,
        sync_before_read=sync_before_read,
        use_slab_allocation=use_slab_allocation,
        started_ns=time.perf_counter_ns(),
    )


@nvtx.annotate("read_coalesced_blocks_many", domain="cudf_polars", color="red")
def _iter_prepared_reads_as_ready(
    prepared_reads: tuple[_PreparedReadRanges, ...],
    *,
    max_workers: int,
) -> Any:
    """Yield per-path read plans as soon as all of their blocks are read."""
    if not prepared_reads:
        return

    ready_queue: queue.Queue[Any] = queue.Queue(
        maxsize=max(
            1,
            min(_DEFAULT_READ_READY_QUEUE_SIZE, len(prepared_reads)),
        )
    )
    sentinel = object()
    total_paths = len(prepared_reads)

    def _producer() -> None:
        handles: list[kvikio.RemoteFile | kvikio.CuFile] = []
        handles_lock = threading.Lock()
        local_state = threading.local()
        state_lock = threading.Lock()
        remaining_by_id = {
            id(prepared): sum(block.size > 0 for block in prepared.coalesced_ranges)
            for prepared in prepared_reads
        }
        flat_tasks: list[tuple[_PreparedReadRanges, int, _CoalescedRange]] = []
        for prepared in prepared_reads:
            for block_index, block in enumerate(prepared.coalesced_ranges):
                if block.size > 0:
                    flat_tasks.append((prepared, block_index, block))
            if remaining_by_id[id(prepared)] == 0:
                ready_queue.put(prepared)

        def _get_handle(
            path: str, file_size: int | None
        ) -> kvikio.RemoteFile | kvikio.CuFile:
            handles_by_path = getattr(local_state, "handles_by_path", None)
            if handles_by_path is None:
                handles_by_path = {}
                local_state.handles_by_path = handles_by_path
            handle = handles_by_path.get(path)
            if handle is None:
                handle = _open_file(path, nbytes=file_size)
                handles_by_path[path] = handle
                with handles_lock:
                    handles.append(handle)
            return handle

        def _mark_block_complete(prepared: _PreparedReadRanges) -> None:
            with state_lock:
                key = id(prepared)
                remaining = remaining_by_id[key] - 1
                remaining_by_id[key] = remaining
            if remaining == 0:
                ready_queue.put(prepared)

        @nvtx.annotate("read_coalesced_block", domain="cudf_polars", color="red")
        def _read_coalesced_block(
            task: tuple[_PreparedReadRanges, int, _CoalescedRange],
        ) -> None:
            prepared, block_index, block = task
            started = time.perf_counter_ns()
            handle = _get_handle(prepared.path, prepared.file_size)
            nread = handle.pread(
                prepared.targets[block_index],
                size=block.size,
                file_offset=block.offset,
            ).get()
            if nread != block.size:
                raise OSError(f"Expected to read {block.size} bytes, got {nread}")
            finished = time.perf_counter_ns()
            _log_io_event(
                "Hybrid Scan Range Request",
                scope="RangeRequest",
                path=prepared.path,
                offset=block.offset,
                size=block.size,
                timestamp=started,
                duration=finished - started,
            )
            _mark_block_complete(prepared)

        try:
            if flat_tasks:
                worker_count = max(1, min(max_workers, len(flat_tasks)))
                if worker_count == 1:
                    for task in flat_tasks:
                        _read_coalesced_block(task)
                else:
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        list(executor.map(_read_coalesced_block, flat_tasks))
        except BaseException as exc:
            ready_queue.put(exc)
        finally:
            for handle in handles:
                handle.close()
            ready_queue.put(sentinel)

    producer = threading.Thread(target=_producer, name="hybrid_scan_read_producer")
    producer.start()
    completed = 0
    try:
        while completed < total_paths:
            item = ready_queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            completed += 1
            yield item
    finally:
        while producer.is_alive():
            with suppress(queue.Empty):
                ready_queue.get(timeout=0.01)
        producer.join()


def _finalize_prepared_range_reads(
    prepared_reads: _PreparedReadRanges,
    *,
    stream: Any,
) -> _ReadRangesResult:
    """Build gpumemoryviews and emit final per-path byte-range metrics."""
    result: list[plc.gpumemoryview] = []
    empty = rmm.DeviceBuffer(size=0, stream=stream)
    prepared_reads.owners.append(empty)
    for slice_info in prepared_reads.slices:
        if slice_info.size == 0:
            result.append(plc.gpumemoryview(empty))
            continue
        view = prepared_reads.targets[slice_info.block_index][
            slice_info.block_offset : slice_info.block_offset + slice_info.size
        ]
        result.append(plc.gpumemoryview(view))
        prepared_reads.owners.append(view)
    elapsed_ns = time.perf_counter_ns() - prepared_reads.started_ns

    dsl_tracing.log(
        "Hybrid Scan Byte Ranges",
        scope="hybrid_scan_read",
        path=prepared_reads.path,
        original_range_count=len(prepared_reads.byte_ranges),
        coalesced_range_count=len(prepared_reads.coalesced_ranges),
        requested_bytes=prepared_reads.requested_bytes,
        read_bytes=prepared_reads.read_bytes,
        over_read_bytes=max(
            prepared_reads.read_bytes - prepared_reads.requested_bytes, 0
        ),
        duration_ns=elapsed_ns,
        max_workers=prepared_reads.max_workers,
        max_coalesce_gap=prepared_reads.max_coalesce_gap,
        sync_before_read=prepared_reads.sync_before_read,
        use_slab_allocation=prepared_reads.use_slab_allocation,
    )
    return _ReadRangesResult(
        column_data=result,
        original_range_count=len(prepared_reads.byte_ranges),
        coalesced_range_count=len(prepared_reads.coalesced_ranges),
        requested_bytes=prepared_reads.requested_bytes,
        read_bytes=prepared_reads.read_bytes,
        backing_refs=tuple(prepared_reads.owners),
    )


@nvtx.annotate("read_parquet_with_hybrid_scan", domain="cudf_polars", color="red")
def read_parquet_with_hybrid_scan(
    paths: list[str],
    column_names: list[str] | None,
    filter_expr: Any | None,
    cached_metadata: CachedParquetMetadata,
    stream: Any,
    parquet_options: ParquetOptions,
    row_group_indices_by_path: dict[str, list[int]] | None = None,
) -> HybridScanReadResult:
    """Read parquet data with pylibcudf's single-step hybrid scan reader."""
    scan_plans: list[_PathScanPlan] = []
    tables: list[plc.Table] = []
    rows_per_source: list[int] = []
    output_names: list[str] | None = None

    for path in paths:
        row_group_start = time.perf_counter_ns()
        file_metadata = cached_metadata.file(path)
        options = _reader_options_for_scan(path, column_names, filter_expr)
        with nvtx.annotate(
            "HybridScanReader.from_parquet_metadata", domain="cudf_polars", color="red"
        ):
            reader = (
                plc.io.experimental.hybrid_scan.HybridScanReader.from_parquet_metadata(
                    file_metadata.hybrid_metadata,
                    options,
                )
            )
        with nvtx.annotate("reader.all_row_groups", domain="cudf_polars", color="red"):
            row_groups = reader.all_row_groups(options)
        total_row_group_counts = len(row_groups)
        if row_group_indices_by_path is not None:
            selected = row_group_indices_by_path.get(path)
            if selected is not None:
                row_groups = [
                    row_groups[index]
                    for index in selected
                    if 0 <= index < len(row_groups)
                ]
        selected_row_group_counts = len(row_groups)
        filtered_row_group_counts = 0
        if filter_expr is not None:
            with nvtx.annotate(
                "reader.filter_row_groups_with_stats", domain="cudf_polars", color="red"
            ):
                pre_filter_count = len(row_groups)
                row_groups = reader.filter_row_groups_with_stats(
                    row_groups,
                    options,
                    stream,
                )
            filtered_row_group_counts = pre_filter_count - len(row_groups)
        row_group_end = time.perf_counter_ns()
        _log_io_event(
            "Hybrid Scan Row Group Decision",
            scope="RowGroup",
            path=path,
            total_row_group_counts=total_row_group_counts,
            selected_row_group_counts=selected_row_group_counts,
            filtered_row_group_counts=filtered_row_group_counts,
            timestamp=row_group_start,
            duration=row_group_end - row_group_start,
        )
        if not row_groups:
            scan_plans.append(
                _PathScanPlan(
                    path=path,
                    reader=reader,
                    options=options,
                    row_groups=row_groups,
                    prepared_reads=None,
                )
            )
            continue

        with nvtx.annotate(
            "reader.all_column_chunks_byte_ranges", domain="cudf_polars", color="red"
        ):
            byte_ranges = reader.all_column_chunks_byte_ranges(row_groups, options)
        if not byte_ranges:
            raise NotImplementedError(
                "Hybrid scan does not yet support zero-column parquet reads."
            )
        scan_plans.append(
            _PathScanPlan(
                path,
                reader,
                options,
                row_groups,
                _prepare_read_ranges_to_device(
                    path,
                    byte_ranges,
                    stream,
                    file_size=file_metadata.file_size,
                    max_coalesce_gap=parquet_options.hybrid_scan_coalesce_max_gap,
                    max_workers=parquet_options.hybrid_scan_max_read_workers,
                    sync_before_read=parquet_options.hybrid_scan_sync_before_read,
                    use_slab_allocation=parquet_options.hybrid_scan_use_slab_allocation,
                ),
            )
        )

    rows_per_source = [0] * len(scan_plans)
    tables_by_index: list[plc.Table | None] = [None] * len(scan_plans)
    plan_by_prepared_id = {
        id(plan.prepared_reads): (index, plan)
        for index, plan in enumerate(scan_plans)
        if plan.prepared_reads is not None
    }
    pending_reads = tuple(
        plan.prepared_reads for plan in scan_plans if plan.prepared_reads is not None
    )
    with nvtx.annotate("read_byte_ranges_to_device", domain="cudf_polars", color="red"):
        for prepared in _iter_prepared_reads_as_ready(
            pending_reads,
            max_workers=parquet_options.hybrid_scan_max_read_workers,
        ):
            plan_index, plan = plan_by_prepared_id[id(prepared)]
            read_result = _finalize_prepared_range_reads(prepared, stream=stream)
            with nvtx.annotate(
                "reader.materialize_all_columns", domain="cudf_polars", color="red"
            ):
                table_with_metadata = plan.reader.materialize_all_columns(
                    plan.row_groups,
                    read_result.column_data,
                    plan.options,
                    stream,
                )
            names = table_with_metadata.column_names(include_children=False)
            if output_names is None:
                output_names = names
            tables_by_index[plan_index] = table_with_metadata.tbl
            rows_per_source[plan_index] = table_with_metadata.tbl.num_rows()

    for index, plan in enumerate(scan_plans):
        if plan.prepared_reads is None:
            rows_per_source[index] = 0
            continue
        table = tables_by_index[index]
        if table is None:  # pragma: no cover
            raise AssertionError(f"Missing materialized table for path {plan.path}")
        tables.append(table)

    if not tables:
        return HybridScanReadResult(
            table=None,
            column_names=column_names or [],
            rows_per_source=rows_per_source,
        )
    table = (
        tables[0]
        if len(tables) == 1
        else plc.concatenate.concatenate(tables, stream=stream)
    )
    return HybridScanReadResult(
        table=table,
        column_names=output_names or [],
        rows_per_source=rows_per_source,
    )


def _validate_supported_scan_partitioning(
    scan: Scan,
    partition_info: MutableMapping[IR, PartitionInfo] | None,
) -> None:
    """Reject lowered parquet scan shapes hybrid scan does not support yet."""
    from cudf_polars.experimental.base import IOPartitionFlavor

    if partition_info is None:
        return
    info = partition_info.get(scan)
    if info is None or info.io_plan is None:
        return
    if info.io_plan.flavor == IOPartitionFlavor.SPLIT_FILES:
        return


@nvtx.annotate("populate_parquet_metadata_cache", domain="cudf_polars", color="red")
def populate_parquet_metadata_cache(
    ir: IR,
    context: IRExecutionContext,
    partition_info: MutableMapping[IR, PartitionInfo] | None = None,
    *,
    max_workers: int = _DEFAULT_FOOTER_PREFETCH_WORKERS,
) -> None:
    """Populate ``context.parquet_metadata`` for parquet scans in *ir*."""
    from cudf_polars.dsl.ir import Scan
    from cudf_polars.dsl.traversal import traversal

    scan_paths: dict[tuple[str, ...], tuple[str, ...]] = {}
    for node in traversal([ir]):
        if not isinstance(node, Scan) or node.typ != "parquet":
            continue
        _validate_supported_scan_partitioning(node, partition_info)
        key = tuple(node.paths)
        if not key or key in context.parquet_metadata:
            continue
        scan_paths[key] = key

    if not scan_paths:
        return

    paths = tuple(dict.fromkeys(path for key in scan_paths for path in key))
    missing_paths = tuple(
        path for path in paths if (path,) not in context.parquet_metadata
    )
    if missing_paths:
        worker_count = min(max_workers, len(missing_paths))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            cached_files = dict(
                zip(
                    missing_paths,
                    executor.map(_fetch_cached_parquet_file_metadata, missing_paths),
                    strict=True,
                )
            )

        for path, file_metadata in cached_files.items():
            context.parquet_metadata[(path,)] = make_cached_parquet_metadata(
                (file_metadata,)
            )

    for key in scan_paths:
        context.parquet_metadata[key] = get_cached_parquet_metadata(list(key), context)
