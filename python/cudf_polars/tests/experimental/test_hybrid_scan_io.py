# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from cudf_polars.experimental.rapidsmpf.hybrid_scan import _coalesce_ranges


@dataclass(frozen=True)
class _Range:
    offset: int
    size: int


def test_coalesce_ranges_merges_adjacent_ranges() -> None:
    ranges = [_Range(0, 10), _Range(10, 10), _Range(30, 5)]
    coalesced, slices = _coalesce_ranges(ranges)

    assert [(block.offset, block.size) for block in coalesced] == [(0, 20), (30, 5)]
    assert [(item.block_index, item.block_offset, item.size) for item in slices] == [
        (0, 0, 10),
        (0, 10, 10),
        (1, 0, 5),
    ]


def test_coalesce_ranges_supports_merge_gap() -> None:
    ranges = [_Range(0, 10), _Range(12, 8)]
    coalesced, slices = _coalesce_ranges(ranges, max_gap=2)

    assert [(block.offset, block.size) for block in coalesced] == [(0, 20)]
    assert [(item.block_index, item.block_offset, item.size) for item in slices] == [
        (0, 0, 10),
        (0, 12, 8),
    ]
