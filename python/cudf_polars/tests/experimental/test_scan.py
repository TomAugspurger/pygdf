# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import date

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )


@pytest.mark.parametrize(
    "fmt, scan_fn",
    [
        ("csv", pl.scan_csv),
        ("ndjson", pl.scan_ndjson),
        ("parquet", pl.scan_parquet),
    ],
)
def test_parallel_scan(tmp_path, df, fmt, scan_fn, engine):
    make_partitioned_source(df, tmp_path, fmt, n_files=3)
    q = scan_fn(tmp_path)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {"target_partition_size": 1_000},
            "engine_options": {"parquet_options": {"use_rapidsmpf_native": True}},
        }
    ],
    indirect=True,
)
def test_scan_parquet_use_rapidsmpf_native(tmp_path, df, engine):
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    assert_gpu_result_equal(pl.scan_parquet(tmp_path), engine=engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_scan.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"target_partition_size": 1_000}}],
    indirect=True,
)
def test_split_scan_aligns_to_row_group_boundaries(tmp_path, df, engine):
    make_partitioned_source(df, tmp_path, "parquet", n_files=1, row_group_size=10)
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("mask", [None, pl.col("x") < 1_000])
@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"target_partition_size": 1_000}}],
    indirect=True,
)
def test_split_scan_predicate(tmp_path, df, mask, engine):
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    q = pl.scan_parquet(tmp_path)
    if mask is not None:
        q = q.filter(mask)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n_files", [2, 3])
@pytest.mark.parametrize(
    "blocksize,engine",
    [
        (1_000, {"executor_options": {"target_partition_size": 1_000}}),
        (10_000, {"executor_options": {"target_partition_size": 10_000}}),
        (1_000_000, {"executor_options": {"target_partition_size": 1_000_000}}),
    ],
    indirect=["engine"],
)
def test_target_partition_size(tmp_path, df, blocksize, n_files, engine):
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=engine)

    # Check partitioning (throwaway engine — no cluster/runtime needed)
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": blocksize},
    )
    qir = Translator(q._ldf.visit(), _engine).translate_ir()
    ir, info, _ = lower_ir_graph(qir, ConfigOptions.from_polars_engine(_engine))
    count = info[ir].count
    if blocksize <= 12_000:
        assert count > n_files
    else:
        assert count < n_files


# ---------------------------------------------------------------------------
# Hybrid scan reader tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def df_mixed():
    """DataFrame with date, numeric, and string columns (like TPC-H lineitem)."""
    n = 3_000
    return pl.DataFrame(
        {
            "quantity": list(range(n)),
            "price": [float(i * 10) for i in range(n)],
            "discount": [float(i % 10) / 100.0 for i in range(n)],
            "flag": ["A", "N", "R"] * (n // 3),
            "status": ["O", "F"] * (n // 2),
            "ship_date": [
                date(2020, 1, 1) + __import__("datetime").timedelta(days=i % 365)
                for i in range(n)
            ],
        }
    )


@pytest.mark.parametrize(
    "engine",
    [{"engine_options": {"parquet_options": {"reader": "hybrid-scan"}}}],
    indirect=True,
)
def test_hybrid_scan_predicate_column_order(tmp_path, df_mixed, engine):
    """Filter on a non-leading column must not reorder the output."""
    make_partitioned_source(df_mixed, tmp_path, "parquet", n_files=1)
    q = (
        pl.scan_parquet(tmp_path)
        .filter(pl.col("ship_date") <= pl.lit(date(2020, 6, 1)))
        .select("quantity", "price", "discount", "flag", "status", "ship_date")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "engine",
    [{"engine_options": {"parquet_options": {"reader": "hybrid-scan"}}}],
    indirect=True,
)
def test_hybrid_scan_predicate_groupby(tmp_path, df_mixed, engine):
    """GroupBy after a predicate-pushdown scan must produce correct results."""
    make_partitioned_source(df_mixed, tmp_path, "parquet", n_files=1)
    q = (
        pl.scan_parquet(tmp_path)
        .filter(pl.col("ship_date") <= pl.lit(date(2020, 6, 1)))
        .group_by("flag", "status")
        .agg(
            pl.col("quantity").sum().alias("sum_qty"),
            pl.col("price").sum().alias("sum_price"),
            pl.col("discount").mean().alias("avg_disc"),
        )
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False, check_exact=False)


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {"target_partition_size": 1_000},
            "engine_options": {"parquet_options": {"reader": "hybrid-scan"}},
        }
    ],
    indirect=True,
)
def test_hybrid_scan_split_predicate_groupby(tmp_path, df_mixed, engine):
    """Split-scan with predicate and groupby must not duplicate data."""
    make_partitioned_source(
        df_mixed, tmp_path, "parquet", n_files=1, row_group_size=100
    )
    q = (
        pl.scan_parquet(tmp_path)
        .filter(pl.col("ship_date") <= pl.lit(date(2020, 6, 1)))
        .group_by("flag")
        .agg(pl.col("price").sum().alias("total_price"))
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False, check_exact=False)


@pytest.mark.parametrize(
    "engine",
    [{"engine_options": {"parquet_options": {"reader": "hybrid-scan"}}}],
    indirect=True,
)
def test_hybrid_scan_non_ast_predicate(tmp_path, df_mixed, engine):
    """Predicates not convertible to parquet filters must still be applied."""
    make_partitioned_source(df_mixed, tmp_path, "parquet", n_files=1)
    q = (
        pl.scan_parquet(tmp_path)
        .filter(pl.col("flag").str.contains("A"))
        .group_by("flag")
        .agg(pl.col("price").sum().alias("total_price"))
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False, check_exact=False)


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {"target_partition_size": 100},
            "engine_options": {"parquet_options": {"reader": "hybrid-scan"}},
        }
    ],
    indirect=True,
)
def test_hybrid_scan_split_more_than_row_groups(tmp_path, df_mixed, engine):
    """SplitScan fallback must use cached metadata, not re-read from storage."""
    make_partitioned_source(
        df_mixed, tmp_path, "parquet", n_files=1, row_group_size=1500
    )
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {"target_partition_size": 100},
            "engine_options": {"parquet_options": {"reader": "hybrid-scan"}},
        }
    ],
    indirect=True,
)
def test_hybrid_scan_split_all_row_groups_filtered(tmp_path, engine):
    """Splits whose row groups are all eliminated by stats must return empty frames."""
    df = pl.DataFrame(
        {
            "x": list(range(300)),
            "y": [float(i) for i in range(300)],
        }
    )
    make_partitioned_source(df, tmp_path, "parquet", n_files=1, row_group_size=100)
    q = pl.scan_parquet(tmp_path).filter(pl.col("x") < 100)
    assert_gpu_result_equal(q, engine=engine)
