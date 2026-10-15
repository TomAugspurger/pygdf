# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Dask execution mode."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

import polars as pl

from rapidsmpf.bootstrap import is_running_with_rrun

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import DaskContext

# try/except instead of pytest.importorskip so mypy can use distributed for type checking
try:
    import distributed
except ImportError:
    pytest.skip("distributed not installed", allow_module_level=True)

from cudf_polars.engine.dask import DaskEngine, _shutdown_dask

if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = [
    pytest.mark.skipif(
        is_running_with_rrun(),
        reason="DaskEngine must not be created from within an rrun cluster",
    ),
]


@pytest.fixture(scope="module")
def dask_client() -> Iterator[distributed.Client]:
    # Use for DaskEngine constructor tests to avoid re-creating the cluster
    # Otherwise, use session-scoped DaskEngine in conftest
    with (
        distributed.LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            silence_logs=logging.WARNING,
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        yield client


def test_shutdown_dask_runs_all_steps_in_order() -> None:
    """A failed cleanup step must not prevent later resources from closing."""
    events: list[str] = []

    def run(func: Any) -> None:
        events.append(func.func.__name__)
        if func.func.__name__ == "_close_quent_worker":
            raise RuntimeError("worker Quent close failed")

    client = SimpleNamespace(run=run)
    runtime = SimpleNamespace(close=lambda: events.append("controller"))
    owned_client = SimpleNamespace(close=lambda: events.append("client"))
    owned_cluster = SimpleNamespace(close=lambda: events.append("cluster"))
    with pytest.raises(ExceptionGroup, match="Dask engine shutdown failed"):
        _shutdown_dask(
            cast("Any", client),
            "test",
            cast("Any", runtime),
            None,
            cast("Any", owned_client),
            cast("Any", owned_cluster),
        )

    assert events == [
        "_close_quent_worker",
        "controller",
        "_teardown_worker",
        "client",
        "cluster",
    ]


# ---------------------------------------------------------------------------
# GPU tests — reuse the session-scoped Dask cluster from conftest
# ---------------------------------------------------------------------------


def test_from_options(dask_client: distributed.Client) -> None:
    """DaskEngine.from_options with default StreamingOptions creates a valid engine."""
    opts = StreamingOptions(fallback_mode="silent")
    with DaskEngine.from_options(opts, dask_client=dask_client) as engine:
        assert engine.nranks >= 1


def test_yields_engine(dask_engine: DaskEngine) -> None:
    """DaskEngine is a GPUEngine with at least one rank."""
    assert isinstance(dask_engine, pl.GPUEngine)
    assert dask_engine.nranks >= 1


def test_executor_options_forwarded(dask_engine: DaskEngine) -> None:
    """Reserved executor_options keys are injected into the engine config."""
    opts = dask_engine.config["executor_options"]
    assert opts["cluster"] == "dask"
    assert isinstance(opts["dask_context"], DaskContext)


def test_gather_cluster_info(dask_engine: DaskEngine) -> None:
    """gather_cluster_info returns one ClusterInfo per rank with expected fields."""
    infos = dask_engine.gather_cluster_info()
    assert len(infos) == dask_engine.nranks
    for info in infos:
        assert isinstance(info.pid, int)
        assert isinstance(info.hostname, str)
    # Each worker runs in its own process.
    assert len({info.pid for info in infos}) == dask_engine.nranks


def test_worker_host_memory_limit(dask_engine: DaskEngine) -> None:
    """Memory limit is respected."""
    scheduler_info = dask_engine._dask_ctx.client.scheduler_info(n_workers=-1)
    worker = next(iter(scheduler_info["workers"].values()))
    assert worker["memory_limit"] == distributed.system.MEMORY_LIMIT


def test_from_options_creates_engine(dask_client: distributed.Client) -> None:
    """DaskEngine.from_options produces a working engine and runs a query."""
    opts = StreamingOptions(max_rows_per_partition=10, fallback_mode="silent")
    with DaskEngine.from_options(opts, dask_client=dask_client) as eng:
        assert isinstance(eng, pl.GPUEngine)
        assert eng.nranks >= 1
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        assert_gpu_result_equal(lf, engine=eng, check_row_order=False)


def test_run(dask_engine: DaskEngine) -> None:
    result = dask_engine._run(os.getpid)
    assert len(set(result)) == dask_engine.nranks


def test_make_engine_reuses_workers(dask_engine: DaskEngine) -> None:
    """Two engines from the same cluster share the same Dask workers."""
    cluster = dask_engine.cluster
    workers_before = sorted(cluster._client.scheduler_info(n_workers=-1)["workers"])
    pids_before = sorted(dask_engine._run(os.getpid))

    e2 = cluster.make_engine(executor_options={"max_rows_per_partition": 7})
    try:
        workers_after = sorted(cluster._client.scheduler_info(n_workers=-1)["workers"])
        pids_after = sorted(e2._run(os.getpid))
        assert workers_before == workers_after
        assert pids_before == pids_after
    finally:
        e2.shutdown()


def test_make_engine_with_options(dask_engine: DaskEngine) -> None:
    """Engine created via ``make_engine`` carries the requested options."""
    cluster = dask_engine.cluster
    e2 = cluster.make_engine(executor_options={"max_rows_per_partition": 42})
    try:
        opts = e2.config["executor_options"]
        assert opts["max_rows_per_partition"] == 42
        assert opts["cluster"] == "dask"
        assert isinstance(opts["dask_context"], DaskContext)
    finally:
        e2.shutdown()


def test_make_engine_and_collect(dask_engine: DaskEngine) -> None:
    """Engine from ``make_engine`` drives a real query."""
    cluster = dask_engine.cluster
    e2 = cluster.make_engine(executor_options={"max_rows_per_partition": 3})
    try:
        assert_gpu_result_equal(
            pl.LazyFrame({"a": [1, 2, 3, 4, 5]}),
            engine=e2,
            check_row_order=False,
        )
    finally:
        e2.shutdown()


def test_make_engine_after_cluster_shutdown_raises() -> None:
    """``make_engine`` raises after the cluster is shut down."""
    from cudf_polars.engine.dask import DaskCluster

    cluster = DaskCluster()
    cluster.shutdown()
    cluster.shutdown()  # idempotent
    with pytest.raises(RuntimeError, match="shut-down"):
        cluster.make_engine()
