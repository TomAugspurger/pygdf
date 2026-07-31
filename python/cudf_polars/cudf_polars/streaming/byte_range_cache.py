# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Prototype byte-range recorder and pinned-host cache for hybrid-scan prefetch.

This is a process-scoped prototype for manually recording parquet byte-range
reads issued via kvikio ``pread``, dumping them to JSON, and pre-populating a
pinned (or host) cache so subsequent ``pread_ranges`` calls can reuse an
already-packed buffer on an exact ordered range-group hit.

Cache entries allocated from a ``PinnedMemoryResource`` do **not** participate
in rapidsmpf ``reserve_memory`` accounting and may oversubscribe the pinned
pool. Prefer clearing the cache between experiments.
"""

from __future__ import annotations

import ctypes
import json
import threading
from dataclasses import dataclass
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
class ByteRange:
    """One file byte range."""

    offset: int
    size: int


@dataclass(frozen=True)
class ByteRangeRequest:
    """One ordered group of ranges packed into a single host buffer."""

    path: str
    ranges: tuple[ByteRange, ...]

    def as_key(self) -> tuple[str, tuple[tuple[int, int], ...]]:
        """Return the cache key for this packed request."""
        return (
            self.path,
            tuple((r.offset, r.size) for r in self.ranges),
        )

    @property
    def nbytes(self) -> int:
        """Return the packed buffer size."""
        return sum(r.size for r in self.ranges)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable form with ``ranges`` as a list."""
        return {
            "path": self.path,
            "ranges": [{"offset": r.offset, "size": r.size} for r in self.ranges],
        }


class _CacheEntry:
    """Owns one packed range-group buffer (pinned MR or plain bytearray)."""

    __slots__ = ("array", "nbytes", "pinned")

    array: memoryview
    nbytes: int
    # (memory resource, pointer, stream) when backed by a pinned pool.
    pinned: tuple[PinnedMemoryResource, int, Stream] | None

    def __init__(
        self,
        nbytes: int,
        *,
        pinned_mr: PinnedMemoryResource | None = None,
        stream: Stream | None = None,
    ) -> None:
        self.nbytes = nbytes
        if pinned_mr is not None and stream is not None:
            ptr = pinned_mr.allocate(self.nbytes, stream)
            self.pinned = (pinned_mr, ptr, stream)
            self.array = byte_view(
                memoryview((ctypes.c_uint8 * self.nbytes).from_address(ptr))
            )
        else:
            self.pinned = None
            self.array = memoryview(bytearray(self.nbytes))

    def __del__(self) -> None:
        # Guard against partial init (e.g. if allocate raised).
        pinned = getattr(self, "pinned", None)
        if pinned is not None:
            mr, ptr, stream = pinned
            mr.deallocate(ptr, self.nbytes, stream)


class ByteRangeCache:
    """Exact-match ordered range-group host/pinned cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, tuple[tuple[int, int], ...]], _CacheEntry] = {}
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

    def get(
        self,
        path: str,
        ranges: Sequence[ByteRange],
    ) -> memoryview | None:
        """Return the already-packed cached buffer, or ``None`` on miss."""
        key = (path, tuple((r.offset, r.size) for r in ranges))
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry.array

    def allocate(self, request: ByteRangeRequest) -> _CacheEntry:
        """Allocate an uninitialized packed entry for direct pread writes."""
        with self._lock:
            pinned_mr = self._pinned_mr
            stream = self._stream
        return _CacheEntry(
            request.nbytes,
            pinned_mr=pinned_mr,
            stream=stream,
        )

    def put(
        self,
        request: ByteRangeRequest,
        data: memoryview,
    ) -> None:
        """Store a packed copy of ``data`` for ``request``."""
        if len(data) != request.nbytes:
            raise ValueError(
                f"data length {len(data)} does not match packed size="
                f"{request.nbytes} for {request.path!r}"
            )
        entry = self.allocate(request)
        entry.array[:] = byte_view(data)
        self.put_entry(request, entry)

    def put_entry(
        self,
        request: ByteRangeRequest,
        entry: _CacheEntry,
    ) -> None:
        """Install a populated cache-owned entry."""
        with self._lock:
            self._entries[request.as_key()] = entry

    def clear(self) -> None:
        """Drop all cached entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Return the number of cached ranges."""
        with self._lock:
            return len(self._entries)

    def __contains__(
        self,
        key: tuple[str, tuple[tuple[int, int], ...]],
    ) -> bool:
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


def record_byte_ranges(
    path: str,
    ranges: Sequence[ByteRange],
) -> None:
    """Record one ordered packed range group when recording is enabled."""
    if not _recording_enabled:
        return
    request = ByteRangeRequest(path=path, ranges=tuple(ranges))
    with _recorder_lock:
        if _recording_enabled:
            _recorded.append(request)


def get_recorded_byte_ranges() -> list[dict[str, Any]]:
    """Return recorded packed range groups as JSON-serializable dicts."""
    with _recorder_lock:
        return [r.to_dict() for r in _recorded]


def dump_recorded_byte_ranges(path: str | Path) -> None:
    """Write recorded byte ranges to ``path`` as JSON."""
    Path(path).write_text(
        json.dumps(get_recorded_byte_ranges(), indent=2) + "\n",
        encoding="utf-8",
    )


def load_byte_range_requests(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list of ``{path, ranges: [{offset, size}, ...]}`` groups."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"expected a JSON list in {path}, got {type(data).__name__}")
    return data


def _coerce_range_requests(
    ranges: Sequence[Mapping[str, Any]] | str | Path,
) -> list[ByteRangeRequest]:
    if isinstance(ranges, str | Path):
        ranges = load_byte_range_requests(ranges)
    requests = []
    for item in ranges:
        raw_ranges = item.get("ranges")
        if not isinstance(raw_ranges, list):
            raise TypeError(
                "each cache request must contain a 'ranges' list; "
                "re-record byte ranges with the packed-cache implementation"
            )
        requests.append(
            ByteRangeRequest(
                path=str(item["path"]),
                ranges=tuple(
                    ByteRange(offset=int(r["offset"]), size=int(r["size"]))
                    for r in raw_ranges
                ),
            )
        )
    return requests


def _batched_by_reads(
    requests: Sequence[ByteRangeRequest],
    batch_size: int,
) -> list[list[ByteRangeRequest]]:
    """Group requests so each batch issues at most ``batch_size`` range reads."""
    batches: list[list[ByteRangeRequest]] = []
    current: list[ByteRangeRequest] = []
    reads = 0
    for request in requests:
        request_reads = len(request.ranges)
        if current and reads + request_reads > batch_size:
            batches.append(current)
            current = []
            reads = 0
        current.append(request)
        reads += request_reads
    if current:
        batches.append(current)
    return batches


def _populate_batch(
    cache: ByteRangeCache,
    handle: Any,
    batch: Sequence[ByteRangeRequest],
) -> None:
    """Read one batch of groups directly into cache-owned packed buffers."""
    pending: list[tuple[ByteRangeRequest, _CacheEntry]] = []
    futures = []
    for request in batch:
        entry = cache.allocate(request)
        offset = 0
        for byte_range in request.ranges:
            futures.append(
                handle.pread(
                    entry.array[offset : offset + byte_range.size],
                    size=byte_range.size,
                    file_offset=byte_range.offset,
                )
            )
            offset += byte_range.size
        pending.append((request, entry))
    for future in futures:
        future.get()
    for request, entry in pending:
        cache.put_entry(request, entry)


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
    Manually fetch packed byte-range groups into the process-global cache.

    Each request gets one cache-owned contiguous allocation. Its range reads
    write directly into their final packed offsets, so a later cache hit can
    pass that allocation directly to H2D without a host-to-host gather.

    Parameters
    ----------
    ranges
        Either a list of ``{"path", "ranges": [{"offset", "size"}, ...]}``
        dicts or a path to JSON produced by
        :func:`dump_recorded_byte_ranges`.
    pinned_mr
        Optional pinned memory resource. When provided (with ``stream``),
        cache entries are allocated from this pool.
    stream
        CUDA stream paired with ``pinned_mr`` for allocate/deallocate.
    batch_size
        Approximate maximum number of range reads submitted before waiting.
    progress
        Print per-batch progress.

    Returns
    -------
    int
        Number of newly populated packed groups.
    """
    cache = get_byte_range_cache()
    if pinned_mr is not None or stream is not None:
        cache.configure(pinned_mr=pinned_mr, stream=stream)

    requests = _coerce_range_requests(ranges)
    # Preserve first-seen order while deduplicating exact ordered groups.
    unique: dict[tuple[str, tuple[tuple[int, int], ...]], ByteRangeRequest] = {}
    for req in requests:
        unique.setdefault(req.as_key(), req)

    by_path: dict[str, list[ByteRangeRequest]] = {}
    for req in unique.values():
        if req.as_key() in cache:
            continue
        by_path.setdefault(req.path, []).append(req)

    total_groups = sum(len(v) for v in by_path.values())
    total_ranges = sum(
        len(req.ranges) for path_reqs in by_path.values() for req in path_reqs
    )
    populated = 0
    populated_ranges = 0
    for path, path_reqs in by_path.items():
        handle = _open_handle(path)
        try:
            for batch in _batched_by_reads(path_reqs, batch_size):
                _populate_batch(cache, handle, batch)
                populated += len(batch)
                populated_ranges += sum(len(req.ranges) for req in batch)
                if progress:
                    print(
                        "byte-range cache: populated "
                        f"{populated}/{total_groups} groups "
                        f"({populated_ranges}/{total_ranges} ranges)",
                        flush=True,
                    )
        finally:
            close = getattr(handle, "close", None)
            if callable(close):
                close()
    return populated
