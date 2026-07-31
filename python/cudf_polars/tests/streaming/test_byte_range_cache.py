# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import ctypes
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

import pylibcudf as plc

from cudf_polars.streaming import byte_range_cache as brc
from cudf_polars.streaming.byte_range_cache import (
    ByteRange,
    ByteRangeRequest,
    clear_recorded_byte_ranges,
    disable_byte_range_recording,
    dump_recorded_byte_ranges,
    enable_byte_range_recording,
    get_byte_range_cache,
    get_recorded_byte_ranges,
    load_byte_range_requests,
    record_byte_ranges,
)
from cudf_polars.streaming.prefetch import pread_ranges

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_byte_range_cache_state():
    disable_byte_range_recording()
    clear_recorded_byte_ranges()
    get_byte_range_cache().clear()
    get_byte_range_cache().configure(pinned_mr=None, stream=None)
    yield
    disable_byte_range_recording()
    clear_recorded_byte_ranges()
    get_byte_range_cache().clear()
    get_byte_range_cache().configure(pinned_mr=None, stream=None)


def test_recorder_accumulates_and_dumps_json(tmp_path: Path):
    enable_byte_range_recording()
    record_byte_ranges(
        "s3://bucket/a.parquet",
        (ByteRange(10, 20), ByteRange(40, 5)),
    )
    record_byte_ranges("/data/b.parquet", (ByteRange(0, 8),))

    recorded = get_recorded_byte_ranges()
    assert recorded == [
        {
            "path": "s3://bucket/a.parquet",
            "ranges": [
                {"offset": 10, "size": 20},
                {"offset": 40, "size": 5},
            ],
        },
        {
            "path": "/data/b.parquet",
            "ranges": [{"offset": 0, "size": 8}],
        },
    ]

    out = tmp_path / "ranges.json"
    dump_recorded_byte_ranges(out)
    assert load_byte_range_requests(out) == recorded


def test_record_byte_range_noop_when_disabled():
    record_byte_ranges("x", (ByteRange(0, 1),))
    assert get_recorded_byte_ranges() == []


def test_cache_put_get_round_trip():
    cache = get_byte_range_cache()
    ranges = (ByteRange(4, 3), ByteRange(20, 5))
    request = ByteRangeRequest("/tmp/f.parquet", ranges)
    data = memoryview(b"abcdefgh")
    cache.put(request, data)
    got = cache.get("/tmp/f.parquet", ranges)
    assert got is not None
    assert bytes(got) == b"abcdefgh"
    assert cache.get("/tmp/f.parquet", tuple(reversed(ranges))) is None


def test_cache_put_accepts_ctypes_format_view():
    # ctypes buffers report format "<B", which memoryview refuses to slice-assign.
    src = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
    cache = get_byte_range_cache()
    ranges = (ByteRange(0, 4),)
    cache.put(ByteRangeRequest("/tmp/f.parquet", ranges), memoryview(src))
    got = cache.get("/tmp/f.parquet", ranges)
    assert got is not None
    assert bytes(got) == b"\x01\x02\x03\x04"


def test_cache_put_rejects_size_mismatch():
    cache = get_byte_range_cache()
    request = ByteRangeRequest("/tmp/f.parquet", (ByteRange(0, 2),))
    with pytest.raises(ValueError, match="does not match packed size"):
        cache.put(request, memoryview(b"abc"))


class _StubIOFuture:
    def get(self) -> None:
        return None


class _StubHandle:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def pread(self, buf, *, size: int, file_offset: int) -> _StubIOFuture:
        self.calls.append(
            {"size": size, "file_offset": file_offset, "nbytes": len(buf)}
        )
        # Fill with a recognizable pattern so packed buffer contents are checkable.
        buf.cast("B")[:] = bytes((file_offset + i) % 256 for i in range(size))
        return _StubIOFuture()


class _FakePinnedBuffer:
    def __init__(self, mr, nbytes, stream, context, loop) -> None:
        self.nbytes = nbytes
        # ctypes-backed like the real PinnedBuffer, so the view reports format "<B".
        self.array = memoryview((ctypes.c_uint8 * nbytes)())


def test_pread_ranges_cache_hit_skips_handle(monkeypatch):
    monkeypatch.setattr(
        "cudf_polars.streaming.prefetch.PinnedBuffer",
        _FakePinnedBuffer,
    )
    path = "s3://bucket/hit.parquet"
    ranges = [
        plc.io.text.ByteRangeInfo(100, 4),
        plc.io.text.ByteRangeInfo(200, 4),
    ]
    cache = get_byte_range_cache()
    cache_ranges = (ByteRange(100, 4), ByteRange(200, 4))
    cache.put(ByteRangeRequest(path, cache_ranges), memoryview(b"AAAABBBB"))

    handle: Any = _StubHandle()
    enable_byte_range_recording()
    host, futures, buf = pread_ranges(
        handle,
        path,
        ranges,
        pinned_mr=MagicMock(),
        stream=MagicMock(),
        context=MagicMock(),
        loop=asyncio.new_event_loop(),
    )

    assert handle.calls == []
    assert futures == []
    assert buf is None
    assert host is not None
    assert bytes(host[:8]) == b"AAAABBBB"
    assert get_recorded_byte_ranges() == [
        {
            "path": path,
            "ranges": [
                {"offset": 100, "size": 4},
                {"offset": 200, "size": 4},
            ],
        },
    ]


def test_pread_ranges_cache_miss_calls_pread(monkeypatch):
    monkeypatch.setattr(
        "cudf_polars.streaming.prefetch.PinnedBuffer",
        _FakePinnedBuffer,
    )
    path = "s3://bucket/miss.parquet"
    ranges = [plc.io.text.ByteRangeInfo(10, 3)]
    handle: Any = _StubHandle()

    host, futures, buf = pread_ranges(
        handle,
        path,
        ranges,
        pinned_mr=MagicMock(),
        stream=MagicMock(),
        context=MagicMock(),
        loop=asyncio.new_event_loop(),
    )

    assert len(handle.calls) == 1
    assert handle.calls[0] == {"size": 3, "file_offset": 10, "nbytes": 3}
    assert len(futures) == 1
    assert buf is not None
    assert host is not None
    assert bytes(host[:3]) == bytes([(10 + i) % 256 for i in range(3)])


def test_pread_ranges_partial_group_miss_reads_entire_group(monkeypatch):
    monkeypatch.setattr(
        "cudf_polars.streaming.prefetch.PinnedBuffer",
        _FakePinnedBuffer,
    )
    path = "/data/mixed.parquet"
    ranges = [
        plc.io.text.ByteRangeInfo(0, 2),
        plc.io.text.ByteRangeInfo(50, 2),
    ]
    # A cached singleton is not a hit for the ordered two-range group.
    singleton = ByteRangeRequest(path, (ByteRange(0, 2),))
    get_byte_range_cache().put(singleton, memoryview(b"ZZ"))
    handle: Any = _StubHandle()

    host, futures, _buf = pread_ranges(
        handle,
        path,
        ranges,
        pinned_mr=MagicMock(),
        stream=MagicMock(),
        context=MagicMock(),
        loop=asyncio.new_event_loop(),
    )

    assert len(handle.calls) == 2
    assert [call["file_offset"] for call in handle.calls] == [0, 50]
    assert len(futures) == 2
    assert host is not None
    assert bytes(host[:2]) == b"\x00\x01"


def test_populate_byte_range_cache_from_list(monkeypatch):
    opened: list[str] = []

    class Handle:
        def pread(self, buf, *, size: int, file_offset: int) -> _StubIOFuture:
            buf[:] = b"x" * size
            return _StubIOFuture()

        def close(self) -> None:
            return None

    def fake_open(path: str):
        opened.append(path)
        return Handle()

    monkeypatch.setattr(brc, "_open_handle", fake_open)

    n = brc.populate_byte_range_cache(
        [
            {
                "path": "/a.parquet",
                "ranges": [
                    {"offset": 0, "size": 3},
                    {"offset": 10, "size": 2},
                ],
            },
            {
                "path": "/a.parquet",
                "ranges": [
                    {"offset": 0, "size": 3},
                    {"offset": 10, "size": 2},
                ],
            },  # duplicate group
            {
                "path": "/b.parquet",
                "ranges": [{"offset": 1, "size": 2}],
            },
        ]
    )
    assert n == 2
    assert opened == ["/a.parquet", "/b.parquet"]
    cache = get_byte_range_cache()
    a = cache.get(
        "/a.parquet",
        (ByteRange(0, 3), ByteRange(10, 2)),
    )
    b = cache.get("/b.parquet", (ByteRange(1, 2),))
    assert a is not None
    assert b is not None
    assert bytes(a) == b"xxxxx"
    assert bytes(b) == b"xx"
