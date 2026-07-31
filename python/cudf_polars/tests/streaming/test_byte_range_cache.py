# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

import pylibcudf as plc

from cudf_polars.streaming import byte_range_cache as brc
from cudf_polars.streaming.byte_range_cache import (
    clear_recorded_byte_ranges,
    disable_byte_range_recording,
    dump_recorded_byte_ranges,
    enable_byte_range_recording,
    get_byte_range_cache,
    get_recorded_byte_ranges,
    load_byte_range_requests,
    record_byte_range,
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
    record_byte_range("s3://bucket/a.parquet", 10, 20)
    record_byte_range("/data/b.parquet", 0, 8)

    recorded = get_recorded_byte_ranges()
    assert recorded == [
        {"path": "s3://bucket/a.parquet", "offset": 10, "size": 20},
        {"path": "/data/b.parquet", "offset": 0, "size": 8},
    ]

    out = tmp_path / "ranges.json"
    dump_recorded_byte_ranges(out)
    assert load_byte_range_requests(out) == recorded


def test_record_byte_range_noop_when_disabled():
    record_byte_range("x", 0, 1)
    assert get_recorded_byte_ranges() == []


def test_cache_put_get_round_trip():
    cache = get_byte_range_cache()
    data = memoryview(b"abcdefgh")
    cache.put("/tmp/f.parquet", 4, 8, data)
    got = cache.get("/tmp/f.parquet", 4, 8)
    assert got is not None
    assert bytes(got) == b"abcdefgh"
    assert cache.get("/tmp/f.parquet", 4, 7) is None


def test_cache_put_rejects_size_mismatch():
    cache = get_byte_range_cache()
    with pytest.raises(ValueError, match="does not match size"):
        cache.put("/tmp/f.parquet", 0, 2, memoryview(b"abc"))


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
        buf[:] = bytes((file_offset + i) % 256 for i in range(size))
        return _StubIOFuture()


class _FakePinnedBuffer:
    def __init__(self, mr, nbytes, stream, context, loop) -> None:
        self.nbytes = nbytes
        self.array = memoryview(bytearray(nbytes))


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
    cache.put(path, 100, 4, memoryview(b"AAAA"))
    cache.put(path, 200, 4, memoryview(b"BBBB"))

    handle = _StubHandle()
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
    assert buf is not None
    assert host is not None
    assert bytes(host[:8]) == b"AAAABBBB"
    assert get_recorded_byte_ranges() == [
        {"path": path, "offset": 100, "size": 4},
        {"path": path, "offset": 200, "size": 4},
    ]


def test_pread_ranges_cache_miss_calls_pread(monkeypatch):
    monkeypatch.setattr(
        "cudf_polars.streaming.prefetch.PinnedBuffer",
        _FakePinnedBuffer,
    )
    path = "s3://bucket/miss.parquet"
    ranges = [plc.io.text.ByteRangeInfo(10, 3)]
    handle = _StubHandle()

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


def test_pread_ranges_mixed_hit_and_miss(monkeypatch):
    monkeypatch.setattr(
        "cudf_polars.streaming.prefetch.PinnedBuffer",
        _FakePinnedBuffer,
    )
    path = "/data/mixed.parquet"
    ranges = [
        plc.io.text.ByteRangeInfo(0, 2),
        plc.io.text.ByteRangeInfo(50, 2),
    ]
    get_byte_range_cache().put(path, 0, 2, memoryview(b"ZZ"))
    handle = _StubHandle()

    host, futures, _buf = pread_ranges(
        handle,
        path,
        ranges,
        pinned_mr=MagicMock(),
        stream=MagicMock(),
        context=MagicMock(),
        loop=asyncio.new_event_loop(),
    )

    assert len(handle.calls) == 1
    assert handle.calls[0]["file_offset"] == 50
    assert len(futures) == 1
    assert host is not None
    assert bytes(host[:2]) == b"ZZ"


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
            {"path": "/a.parquet", "offset": 0, "size": 3},
            {"path": "/a.parquet", "offset": 0, "size": 3},  # duplicate
            {"path": "/b.parquet", "offset": 1, "size": 2},
        ]
    )
    assert n == 2
    assert opened == ["/a.parquet", "/b.parquet"]
    cache = get_byte_range_cache()
    assert bytes(cache.get("/a.parquet", 0, 3)) == b"xxx"
    assert bytes(cache.get("/b.parquet", 1, 2)) == b"xx"
