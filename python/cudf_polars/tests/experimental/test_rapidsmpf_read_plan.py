# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl.ir import IRExecutionContext, Scan
from cudf_polars.experimental.base import (
    IOPartitionFlavor,
    IOPartitionPlan,
    PartitionInfo,
)
from cudf_polars.experimental.io import SplitScan
from cudf_polars.experimental.rapidsmpf.frontend import core as frontend_core
from cudf_polars.experimental.rapidsmpf.io import (
    RankReadPlan,
    ReadSpec,
    build_local_read_specs,
    build_rank_read_plan,
)
from cudf_polars.utils.config import ParquetOptions


def _make_scan(paths: list[str]) -> Scan:
    return Scan(
        {"a": DataType(pl.Int64())},
        "parquet",
        {},
        None,
        paths,
        None,
        0,
        -1,
        None,
        None,
        None,
        ParquetOptions(),
    )


def test_build_local_read_specs_split_files_across_rank_boundary() -> None:
    scan = _make_scan(["a.parquet", "b.parquet", "c.parquet"])
    comm = SimpleNamespace(rank=1, nranks=2)

    specs = build_local_read_specs(
        scan,
        IOPartitionPlan(2, IOPartitionFlavor.SPLIT_FILES),
        comm,
    )

    assert [(spec.paths, spec.split_index, spec.total_splits) for spec in specs] == [
        (("b.parquet",), 1, 2),
        (("c.parquet",), 0, 2),
        (("c.parquet",), 1, 2),
    ]

    read_plan = RankReadPlan({scan: specs})
    assert read_plan.parquet_metadata_keys() == (
        ("b.parquet",),
        ("c.parquet",),
    )
    assert read_plan.parquet_paths() == ("b.parquet", "c.parquet")


def test_build_rank_read_plan_uses_partition_info_io_plan() -> None:
    scan = _make_scan(["a.parquet", "b.parquet", "c.parquet", "d.parquet"])
    comm = SimpleNamespace(rank=1, nranks=2)

    read_plan = build_rank_read_plan(
        scan,
        {
            scan: PartitionInfo(
                count=2,
                io_plan=IOPartitionPlan(2, IOPartitionFlavor.FUSED_FILES),
            )
        },
        comm,
    )

    assert [spec.paths for spec in read_plan.by_scan[scan]] == [
        ("c.parquet", "d.parquet")
    ]


def test_read_spec_materialize_returns_scan_types() -> None:
    scan = _make_scan(["a.parquet", "b.parquet"])

    fused = ReadSpec(scan, ("a.parquet", "b.parquet")).materialize(ParquetOptions())
    assert isinstance(fused, Scan)
    assert fused.paths == ["a.parquet", "b.parquet"]

    split = ReadSpec(
        scan,
        ("a.parquet",),
        split_index=1,
        total_splits=3,
    ).materialize(ParquetOptions())
    assert isinstance(split, SplitScan)
    assert split.base_scan.paths == ["a.parquet"]
    assert split.split_index == 1
    assert split.total_splits == 3


def test_collect_parquet_metadata_from_rank_read_plan() -> None:
    scan = _make_scan(["a.parquet", "b.parquet", "c.parquet"])
    read_plan = RankReadPlan(
        {
            scan: (
                ReadSpec(scan, ("a.parquet",), split_index=0, total_splits=2),
                ReadSpec(scan, ("a.parquet",), split_index=1, total_splits=2),
                ReadSpec(scan, ("b.parquet", "c.parquet")),
            )
        }
    )
    footer_calls: list[str] = []
    metadata_calls: list[tuple[bytes, ...]] = []

    def read_footer_bytes(path: str) -> bytes:
        footer_calls.append(path)
        return f"footer:{path}".encode()

    def build_metadata_from_footer_bytes(
        footer_bytes: list[bytes],
    ) -> dict[str, tuple[bytes, ...]]:
        payload = tuple(footer_bytes)
        metadata_calls.append(payload)
        return {"footers": payload}

    ir_context = IRExecutionContext(
        parquet_footer_bytes={"b.parquet": b"cached-b"},
        parquet_metadata={("b.parquet", "c.parquet"): "cached"},  # type: ignore[dict-item]
    )
    with ThreadPoolExecutor(max_workers=2) as py_executor:
        updated = frontend_core.collect_parquet_metadata(
            ir_context,
            read_plan,
            py_executor,
            read_footer_bytes=read_footer_bytes,
            build_metadata_from_footer_bytes=build_metadata_from_footer_bytes,
        )

    assert sorted(footer_calls) == ["a.parquet", "c.parquet"]
    assert metadata_calls == [(b"footer:a.parquet",)]
    assert updated.parquet_footer_bytes == {
        "a.parquet": b"footer:a.parquet",
        "b.parquet": b"cached-b",
        "c.parquet": b"footer:c.parquet",
    }
    assert updated.parquet_metadata == {
        ("a.parquet",): {"footers": (b"footer:a.parquet",)},
        ("b.parquet", "c.parquet"): "cached",
    }
