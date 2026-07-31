# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Prototype byte-range recorder and pinned-host cache for hybrid-scan prefetch.

This is a process-scoped prototype for manually recording parquet byte-range
reads issued via kvikio ``pread``, dumping them to JSON, and pre-populating a
pinned (or host) cache so subsequent ``pread_ranges`` calls can skip HTTP/CuFile
I/O on exact ``(path, offset, size)`` hits.

Cache entries allocated from a ``PinnedMemoryResource`` do **not** participate
in rapidsmpf ``reserve_memory`` accounting and may oversubscribe the pinned
pool. Prefer clearing the cache between experiments.
"""

from __future__ import annotations

import ctypes
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

try:  # pragma: no cover; kvikio is optional
    import kvikio
except ImportError:
    kvikio = None

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rapidsmpf.memory.pinned_memory_resource import PinnedMemoryResource
    from rmm.pylibrmm.stream import Stream


def byte_view(buf: memoryview) -> memoryview:
    """Return an unsigned-byte view; ctypes-backed buffers report format ``<B``."""
    return buf if buf.format == "B" else buf.cast("B")


@dataclass(frozen=True)
class ByteRangeRequest:
    """A single file byte-range request."""

    path: str
    offset: int
    size: int

    def as_key(self) -> tuple[str, int, int]:
        """Return the cache key for this request."""
        return (self.path, self.offset, self.size)


class _CacheEntry:
    """Owns host bytes for one cached range (pinned MR or plain bytearray)."""

    __slots__ = ("array", "nbytes", "pinned")

    array: memoryview
    nbytes: int
    # (memory resource, pointer, stream) when backed by a pinned pool.
    pinned: tuple[PinnedMemoryResource, int, Stream] | None

    def __init__(
        self,
        data: memoryview,
        *,
        pinned_mr: PinnedMemoryResource | None = None,
        stream: Stream | None = None,
    ) -> None:
        self.nbytes = len(data)
        if pinned_mr is not None and stream is not None:
            ptr = pinned_mr.allocate(self.nbytes, stream)
            self.pinned = (pinned_mr, ptr, stream)
            self.array = byte_view(
                memoryview((ctypes.c_uint8 * self.nbytes).from_address(ptr))
            )
            self.array[:] = byte_view(data)
        else:
            self.pinned = None
            self.array = memoryview(bytearray(byte_view(data)))

    def __del__(self) -> None:
        # Guard against partial init (e.g. if allocate raised).
        pinned = getattr(self, "pinned", None)
        if pinned is not None:
            mr, ptr, stream = pinned
            mr.deallocate(ptr, self.nbytes, stream)


class ByteRangeCache:
    """Exact-match ``(path, offset, size)`` host/pinned cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, int, int], _CacheEntry] = {}
        self._pinned_mr: PinnedMemoryResource | None = None
        self._stream: Stream | None = None

    def configure(
        self,
        pinned_mr: PinnedMemoryResource | None = None,
        stream: Stream | None = None,
    ) -> None:
        """Configure the memory resource used for subsequent ``put`` calls."""
        with self._lock:
            self._pinned_mr = pinned_mr
            self._stream = stream

    def get(self, path: str, offset: int, size: int) -> memoryview | None:
        """Return a view of cached bytes, or ``None`` on miss."""
        with self._lock:
            entry = self._entries.get((path, offset, size))
            return None if entry is None else entry.array

    def put(self, path: str, offset: int, size: int, data: memoryview) -> None:
        """Store a copy of ``data`` under ``(path, offset, size)``."""
        if len(data) != size:
            raise ValueError(
                f"data length {len(data)} does not match size={size} "
                f"for {path!r} offset={offset}"
            )
        entry = _CacheEntry(
            data,
            pinned_mr=self._pinned_mr,
            stream=self._stream,
        )
        with self._lock:
            self._entries[(path, offset, size)] = entry

    def clear(self) -> None:
        """Drop all cached entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Return the number of cached ranges."""
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: tuple[str, int, int]) -> bool:
        """Return whether ``key`` is present."""
        with self._lock:
            return key in self._entries


_cache = ByteRangeCache()
_recording_enabled = False
_recorded: list[ByteRangeRequest] = []
_recorder_lock = threading.Lock()


def get_byte_range_cache() -> ByteRangeCache:
    """Return the process-global byte-range cache."""
    return _cache


def enable_byte_range_recording() -> None:
    """Enable recording of byte-range reads from ``pread_ranges``."""
    global _recording_enabled  # noqa: PLW0603
    with _recorder_lock:
        _recording_enabled = True


def disable_byte_range_recording() -> None:
    """Disable recording of byte-range reads."""
    global _recording_enabled  # noqa: PLW0603
    with _recorder_lock:
        _recording_enabled = False


def clear_recorded_byte_ranges() -> None:
    """Clear all recorded byte-range requests."""
    with _recorder_lock:
        _recorded.clear()


def record_byte_range(path: str, offset: int, size: int) -> None:
    """Record one byte-range request when recording is enabled."""
    if not _recording_enabled:
        return
    with _recorder_lock:
        if _recording_enabled:
            _recorded.append(ByteRangeRequest(path=path, offset=offset, size=size))


def get_recorded_byte_ranges() -> list[dict[str, Any]]:
    """Return recorded ranges as JSON-serializable dicts."""
    with _recorder_lock:
        return [asdict(r) for r in _recorded]


def dump_recorded_byte_ranges(path: str | Path) -> None:
    """Write recorded byte ranges to ``path`` as JSON."""
    Path(path).write_text(
        json.dumps(get_recorded_byte_ranges(), indent=2) + "\n",
        encoding="utf-8",
    )


def load_byte_range_requests(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list of ``{path, offset, size}`` requests."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"expected a JSON list in {path}, got {type(data).__name__}")
    return data


def _coerce_range_requests(
    ranges: Sequence[Mapping[str, Any]] | str | Path,
) -> list[ByteRangeRequest]:
    if isinstance(ranges, str | Path):
        ranges = load_byte_range_requests(ranges)
    return [
        ByteRangeRequest(
            path=str(item["path"]),
            offset=int(item["offset"]),
            size=int(item["size"]),
        )
        for item in ranges
    ]


def _open_handle(path: str) -> Any:
    if kvikio is None:  # pragma: no cover
        raise ImportError("kvikio is required to populate the byte-range cache")
    if plc.io.SourceInfo._is_remote_uri(path):
        return kvikio.RemoteFile.open(path)
    return kvikio.CuFile(path)


def populate_byte_range_cache(
    ranges: Sequence[Mapping[str, Any]] | str | Path,
    *,
    pinned_mr: PinnedMemoryResource | None = None,
    stream: Stream | None = None,
    batch_size: int = 256,
    progress: bool = False,
) -> int:
    """
    Manually fetch byte ranges into the process-global host/pinned cache.

    Reads are issued in concurrent batches per file; a serial read-and-wait
    loop is far too slow for the thousands of small ranges a scan produces.

    Parameters
    ----------
    ranges
        Either a list of ``{"path", "offset", "size"}`` dicts or a path to a
        JSON file produced by :func:`dump_recorded_byte_ranges`.
    pinned_mr
        Optional pinned memory resource. When provided (with ``stream``),
        cache entries are allocated from this pool.
    stream
        CUDA stream paired with ``pinned_mr`` for allocate/deallocate.
    batch_size
        Number of reads submitted before waiting on the batch.
    progress
        Print per-batch progress.

    Returns
    -------
    int
        Number of newly populated cache entries (duplicates and existing hits
        are skipped).
    """
    cache = get_byte_range_cache()
    if pinned_mr is not None or stream is not None:
        cache.configure(pinned_mr=pinned_mr, stream=stream)

    requests = _coerce_range_requests(ranges)
    # Preserve first-seen order while deduplicating.
    unique: dict[tuple[str, int, int], ByteRangeRequest] = {}
    for req in requests:
        unique.setdefault(req.as_key(), req)

    by_path: dict[str, list[ByteRangeRequest]] = {}
    for req in unique.values():
        if req.as_key() in cache:
            continue
        by_path.setdefault(req.path, []).append(req)

    total = sum(len(v) for v in by_path.values())
    populated = 0
    for path, path_reqs in by_path.items():
        handle = _open_handle(path)
        try:
            for start in range(0, len(path_reqs), batch_size):
                batch = path_reqs[start : start + batch_size]
                buf = memoryview(bytearray(sum(r.size for r in batch)))
                views = []
                futures = []
                offset = 0
                for req in batch:
                    view = buf[offset : offset + req.size]
                    futures.append(
                        handle.pread(view, size=req.size, file_offset=req.offset)
                    )
                    views.append(view)
                    offset += req.size
                for future in futures:
                    future.get()
                for req, view in zip(batch, views, strict=True):
                    cache.put(req.path, req.offset, req.size, view)
                populated += len(batch)
                if progress:
                    print(
                        f"byte-range cache: populated {populated}/{total} ranges",
                        flush=True,
                    )
        finally:
            close = getattr(handle, "close", None)
            if callable(close):
                close()
    return populated
