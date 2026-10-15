# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ray execution mode."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

import polars as pl

from rapidsmpf.bootstrap import is_running_with_rrun

from cudf_polars.utils.config import RayContext

ray = pytest.importorskip("ray")
from cudf_polars.engine.ray import RayEngine  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        is_running_with_rrun(),
        reason="RayEngine must not be created from within an rrun cluster",
    ),
]


# ---------------------------------------------------------------------------
# Context-manager smoke tests (no GPU required)
# ---------------------------------------------------------------------------


def test_reserved_executor_keys() -> None:
    """executor_options rejects reserved keys."""
    for key in ("cluster", "spmd_context", "ray_context"):
        with pytest.raises(TypeError, match="reserved"):
            RayEngine(executor_options={key: "anything"})


def test_reserved_engine_options_keys() -> None:
    """engine_options rejects reserved keys."""
    for key in ("memory_resource", "executor"):
        with pytest.raises(TypeError, match="reserved"):
            RayEngine(engine_options={key: "anything"})


def test_raises_inside_rrun() -> None:
    """RayEngine must not be created from within an rrun cluster."""
    with (
        patch(
            "rapidsmpf.bootstrap.is_running_with_rrun",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="rrun"),
    ):
        RayEngine()


def test_num_ranks_requires_allow_gpu_sharing(ray_num_ranks: int) -> None:
    """num_ranks requires engine_options['allow_gpu_sharing']=True."""
    with pytest.raises(ValueError, match="allow_gpu_sharing"):
        RayEngine(num_ranks=ray_num_ranks)
    with pytest.raises(ValueError, match="allow_gpu_sharing"):
        RayEngine(num_ranks=ray_num_ranks, engine_options={"allow_gpu_sharing": False})


def test_num_ranks_must_be_positive() -> None:
    """num_ranks must be at least 1."""
    with pytest.raises(ValueError, match="num_ranks"):
        RayEngine(num_ranks=0, engine_options={"allow_gpu_sharing": True})


# ---------------------------------------------------------------------------
# GPU tests — reuse the session-scoped Ray cluster from conftest
# ---------------------------------------------------------------------------


def test_yields_engine(ray_engine: RayEngine) -> None:
    """RayEngine is a GPUEngine with at least one rank."""
    assert isinstance(ray_engine, pl.GPUEngine)
    assert ray_engine.nranks >= 1


def test_executor_options_forwarded(ray_engine: RayEngine) -> None:
    """Reserved executor_options keys are injected into the engine config."""
    opts = ray_engine.config["executor_options"]
    assert opts["cluster"] == "ray"
    assert isinstance(opts["ray_context"], RayContext)
    assert ray_engine.rank_actors == opts["ray_context"].rank_actors
    assert len(ray_engine.rank_actors) == ray_engine.nranks


def test_gather_cluster_info(ray_engine: RayEngine) -> None:
    """gather_cluster_info returns one ClusterInfo per rank with expected fields."""
    infos = ray_engine.gather_cluster_info()
    assert len(infos) == ray_engine.nranks
    for info in infos:
        assert isinstance(info.hostname, str)
        assert isinstance(info.pid, int)
    # Each actor runs in its own process.
    assert len({info.pid for info in infos}) == ray_engine.nranks


def test_run(ray_engine: RayEngine) -> None:
    result = ray_engine._run(os.getpid)
    assert len(set(result)) == ray_engine.nranks


def test_num_ranks_oversubscribes(ray_engine: RayEngine, ray_num_ranks: int) -> None:
    """num_ranks creates the requested number of actors sharing GPU 0."""
    assert ray_engine.nranks == ray_num_ranks
    assert len(ray_engine.rank_actors) == ray_num_ranks
    result = pl.LazyFrame({"a": [1, 2, 3, 4]}).collect(engine=ray_engine)
    assert sorted(result["a"].to_list()) == [1, 2, 3, 4]


def test_make_engine_reuses_actors(ray_engine: RayEngine) -> None:
    """Two engines from the same cluster share the same Ray actors."""
    cluster = ray_engine.cluster
    assert cluster._rank_actors is not None
    actors_before = list(cluster._rank_actors)
    pids_before = ray_engine._run(os.getpid)

    e2 = cluster.make_engine(
        executor_options={"max_rows_per_partition": 7},
        engine_options={"allow_gpu_sharing": True},
    )
    try:
        assert cluster._rank_actors is not None
        actors_after = list(cluster._rank_actors)
        pids_after = e2._run(os.getpid)
        assert all(a is b for a, b in zip(actors_before, actors_after, strict=True))
        assert pids_before == pids_after
    finally:
        e2.shutdown()


def test_make_engine_with_options(ray_engine: RayEngine) -> None:
    """Engine created via ``make_engine`` carries the requested options."""
    cluster = ray_engine.cluster
    e2 = cluster.make_engine(
        executor_options={"max_rows_per_partition": 42},
        engine_options={"allow_gpu_sharing": True},
    )
    try:
        opts = e2.config["executor_options"]
        assert opts["max_rows_per_partition"] == 42
        assert opts["cluster"] == "ray"
        assert isinstance(opts["ray_context"], RayContext)
        assert opts["ray_context"].rank_actors == cluster._rank_actors
    finally:
        e2.shutdown()


def test_make_engine_and_collect(ray_engine: RayEngine) -> None:
    """Engine from ``make_engine`` drives a real query."""
    cluster = ray_engine.cluster
    e2 = cluster.make_engine(
        executor_options={"max_rows_per_partition": 3},
        engine_options={"allow_gpu_sharing": True},
    )
    try:
        result = pl.LazyFrame({"a": [1, 2, 3, 4, 5]}).collect(engine=e2)
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4, 5]
    finally:
        e2.shutdown()


def test_make_engine_after_cluster_shutdown_raises(
    ray_init_options: dict[str, Any],
    ray_num_ranks: int,
) -> None:
    """``make_engine`` raises after the cluster is shut down."""
    from cudf_polars.engine.ray import RayCluster

    cluster = RayCluster(
        num_ranks=ray_num_ranks,
        engine_options={"allow_gpu_sharing": True},
        ray_init_options=ray_init_options,
    )
    cluster.shutdown()
    cluster.shutdown()  # idempotent
    with pytest.raises(RuntimeError, match="shut-down"):
        cluster.make_engine(engine_options={"allow_gpu_sharing": True})


def test_shutdown_skips_when_ray_not_initialized(
    ray_init_options: dict[str, Any],
) -> None:
    """``shutdown`` short-circuits if ``ray.is_initialized()`` is ``False``."""
    engine = RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        # The no-auto-init guard is independent of rank count.
        num_ranks=1,
        ray_init_options=ray_init_options,
    )
    try:
        with patch("ray.is_initialized", return_value=False):
            engine.shutdown()  # must not raise, must not auto-init
        # The exit stack is closed; cluster is cleared.
        assert engine._cluster is None
    finally:
        # Defensive: if the guard somehow didn't fire, make sure the
        # actors are released so the next test's fixture isn't blocked.
        if engine._cluster is not None:
            engine.shutdown()


def test_gathers_are_in_rank_order(ray_init_options: dict[str, Any]) -> None:
    # Two ranks are sufficient to prove that the methods order by rank rather
    # than actor-list position. The explicit reversal makes this deterministic,
    # so additional engine constructions only repeat the same assertion.
    engine = RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=2,
        ray_init_options=ray_init_options,
    )
    try:
        # Each actor's own view: (rank, info) in actor order.
        actors = engine.rank_actors
        reported = ray.get([actor.get_info.remote() for actor in actors])
        assert sorted(rank for rank, _ in reported) == [0, 1]

        # Actor positions that put the ranks in ascending order.
        ascending = sorted(range(len(actors)), key=lambda i: reported[i][0])

        # What the gathers must return: the infos in rank order.
        expected = [reported[i][1].pid for i in ascending]

        # Hold the actors in descending rank order, so position and rank are
        # guaranteed to disagree. Without this, a position-ordered gather can
        # agree with rank order by luck.
        assert engine._cluster is not None
        engine._cluster._rank_actors = [actors[i] for i in reversed(ascending)]

        assert [info.pid for info in engine.gather_cluster_info()] == expected

        # `_run` reaches the same actor for a given rank, so its pids line up
        # with the cluster info once both are in rank order.
        assert engine._run(os.getpid) == expected
    finally:
        engine.shutdown()
