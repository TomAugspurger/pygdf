# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from rapidsmpf.bootstrap import get_nranks, is_running_with_rrun

from cudf_polars.testing.engine_utils import (
    ALL_ENGINE_FIXTURE_PARAMS,
    STREAMING_ENGINE_FIXTURE_PARAMS,
    EngineFixtureParam,
    create_streaming_options,
    merge_streaming_options,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_140, POLARS_VERSION_LT_141


@pytest.fixture
def xfail_decimal_sum_precision_polars_140(request: pytest.FixtureRequest) -> None:
    """xfail decimal ``sum`` tests on polars 1.40."""
    request.applymarker(
        pytest.mark.xfail(
            condition=(not POLARS_VERSION_LT_140) and POLARS_VERSION_LT_141,
            reason="polars 1.40 reports narrow precision for decimal sum "
            "(Decimal(9,2) vs (38,2)), fixed in 1.41. "
            "See https://github.com/pola-rs/polars/issues/27269",
        )
    )


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.engine.dask import DaskCluster, DaskEngine
    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.engine.ray import RayCluster, RayEngine
    from cudf_polars.engine.spmd import SPMDCluster, SPMDEngine


@pytest.fixture(scope="session")
def ray_num_ranks() -> int:
    """
    Number of ranks for multi-rank streaming engines that share one GPU
    (currently ``RayEngine``). Single-GPU dev hosts and CI runners require
    ``allow_gpu_sharing=True`` to oversubscribe one device across actors.
    """
    return 2


@pytest.fixture(scope="session")
def ray_init_options(ray_num_ranks: int) -> dict[str, Any]:
    """
    Keyword arguments forwarded to ``ray.init`` for the test Ray cluster.

    When using this fixture, a ``RayEngine`` must be constructed with:
    - ``num_ranks`` set
    - ``engine_options={"allow_gpu_sharing": True}``

    This is required because the cluster is configured with ``num_gpus=0``,
    so Ray does not autodetect or track GPU resources.
    """
    return {
        "num_cpus": ray_num_ranks,
        "num_gpus": 0,
        "include_dashboard": False,
        "object_store_memory": 256 * 1024 * 1024,  # 256 MB
    }


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"], scope="session")
def with_nulls(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(autouse=True)
def _skip_unless_spmd(request: pytest.FixtureRequest) -> None:
    """Skip tests in SPMD multi-rank mode unless marked with ``pytest.mark.spmd``."""
    if (
        is_running_with_rrun()
        and get_nranks() > 1
        and not request.node.get_closest_marker("spmd")
    ):
        pytest.skip("skip: SPMD nranks > 1 (mark with pytest.mark.spmd to run)")


@pytest.fixture(scope="session", autouse=True)
def _configure_dask() -> Generator[None, None, None]:
    """Apply test-suite-wide Dask configuration."""
    # workaround https://github.com/dask/distributed/issues/9374
    try:
        import dask
    except ImportError:
        yield
        return

    with dask.config.set({"distributed.client.scheduler-info-interval": "60s"}):
        yield


@pytest.fixture(scope="session")
def _engine_param(request: pytest.FixtureRequest) -> EngineFixtureParam:
    """Decoded engine variant selected by pytest parametrization.

    :func:`pytest_generate_tests` inspects all tests and then filters those
    with ``_engine_param`` in their fixture list according to the public
    engine fixture being used. For example, if a given test requests the
    :func:`spmd_engine` fixture then its underlying ``_engine_param`` is
    rebound to only loop over spmd engines for that test.
    """
    return EngineFixtureParam(full_name=request.param)


@pytest.fixture(scope="session")
def spmd_cluster() -> Generator[SPMDCluster, None, None]:
    """
    Session-scoped :class:`~cudf_polars.engine.spmd.SPMDCluster`.

    Bootstraps the UCX communicator and rapidsmpf resources once per test
    session. Individual tests obtain :class:`SPMDEngine` instances from this
    cluster via :meth:`~cudf_polars.engine.spmd.SPMDCluster.make_engine`,
    which is fast because the expensive physical setup is already done.
    """
    from cudf_polars.engine.spmd import SPMDCluster

    with SPMDCluster() as cluster:
        yield cluster


@pytest.fixture(scope="session")
def dask_cluster() -> Generator[DaskCluster, None, None]:
    """
    Session-scoped :class:`~cudf_polars.engine.dask.DaskCluster`.

    Bootstraps the Dask client and UCX communicators once per test session.
    Individual tests obtain :class:`DaskEngine` instances via
    :meth:`~cudf_polars.engine.dask.DaskCluster.make_engine`.
    """
    pytest.importorskip("distributed")
    from cudf_polars.engine.dask import DaskCluster

    with DaskCluster(engine_options={"allow_gpu_sharing": True}) as cluster:
        yield cluster


@pytest.fixture(scope="session")
def ray_cluster(
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
) -> Generator[RayCluster, None, None]:
    """
    Session-scoped :class:`~cudf_polars.engine.ray.RayCluster`.

    Bootstraps Ray and UCX communicators once per test session.
    Individual tests obtain :class:`RayEngine` instances via
    :meth:`~cudf_polars.engine.ray.RayCluster.make_engine`.
    """
    pytest.importorskip("ray")
    from cudf_polars.engine.ray import RayCluster

    with RayCluster(
        num_ranks=ray_num_ranks,
        engine_options={"allow_gpu_sharing": True},
        ray_init_options=ray_init_options,
    ) as cluster:
        yield cluster


@pytest.fixture(scope="session")
def _unconfigured_engine(
    _engine_param: EngineFixtureParam,
    spmd_cluster: SPMDCluster,
    request: pytest.FixtureRequest,
) -> Generator[
    tuple[
        pl.GPUEngine | SPMDCluster | DaskCluster | RayCluster, StreamingOptions | None
    ],
    None,
    None,
]:
    """
    Fixture generating a cluster/engine and baseline options for each test.

    For all streaming backends this now yields the session-scoped cluster so
    per-test fixtures can call :meth:`make_engine` cheaply without paying the
    bootstrap cost again.  The in-memory backend still yields a plain engine.

    Dask and Ray clusters are resolved lazily via ``request.getfixturevalue``
    so that SPMD-only tests are not skipped when those packages are absent.

    Tests should not use this fixture directly; use the public fixtures.
    """
    if _engine_param.engine_name == "in-memory":
        yield pl.GPUEngine(executor="in-memory", raise_on_fail=True), None
    else:
        if _engine_param.engine_name in ("dask", "ray") and is_running_with_rrun():
            pytest.skip(
                f"{_engine_param.engine_name} engine cannot be constructed "
                "inside an rrun cluster"
            )
        match _engine_param.engine_name:
            case "spmd" | "spmd-small":
                yield (
                    spmd_cluster,
                    create_streaming_options(_engine_param.blocksize_mode),
                )
            case "dask":  # pragma: no cover
                yield (
                    request.getfixturevalue("dask_cluster"),
                    create_streaming_options(_engine_param.blocksize_mode),
                )
            case "ray":  # pragma: no cover
                yield (
                    request.getfixturevalue("ray_cluster"),
                    create_streaming_options(_engine_param.blocksize_mode),
                )
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown streaming engine: {_engine_param.engine_name!r}"
                )


def _engine_from_unconfigured(
    thing: pl.GPUEngine | SPMDCluster | DaskCluster | RayCluster,
    options: StreamingOptions,
) -> StreamingEngine:
    """
    Produce a configured engine from the ``_unconfigured_engine`` fixture value.

    For :class:`~cudf_polars.engine.spmd.SPMDCluster`,
    :class:`~cudf_polars.engine.dask.DaskCluster`, and
    :class:`~cudf_polars.engine.ray.RayCluster` this calls
    :meth:`make_engine` with the resolved options. For SPMD, if rapidsmpf
    options (e.g. ``statistics=True``) differ from the cluster's defaults, a
    fresh :class:`~cudf_polars.engine.spmd.SPMDEngine` with its own cluster
    is created so the requested ``Context`` settings take effect.
    """
    from cudf_polars.engine.spmd import SPMDCluster as _SPMDCluster

    if isinstance(thing, _SPMDCluster):
        # If the options request non-default rapidsmpf settings the cluster
        # must be rebuilt with them, since they affect the rapidsmpf Context.
        from cudf_polars.engine.options import StreamingOptions as _StreamingOptions
        from cudf_polars.engine.spmd import SPMDEngine

        default_rmpf = _StreamingOptions().to_rapidsmpf_options()
        requested_rmpf = options.to_rapidsmpf_options()
        if str(requested_rmpf) != str(default_rmpf):
            # Rapidsmpf options differ → need a dedicated cluster.
            return SPMDEngine.from_options(options)
        return thing.make_engine(
            executor_options=options.to_executor_options(),
            engine_options=options.to_engine_options(),
        )
    # Dask / Ray cluster: import lazily so missing optional packages don't
    # break SPMD-only tests.
    from cudf_polars.engine.options import StreamingOptions as _StreamingOptions

    default_rmpf = _StreamingOptions().to_rapidsmpf_options()
    requested_rmpf = options.to_rapidsmpf_options()
    rmpf_differ = str(requested_rmpf) != str(default_rmpf)

    try:
        from cudf_polars.engine.dask import DaskCluster as _DaskCluster, DaskEngine

        if isinstance(thing, _DaskCluster):
            if rmpf_differ:
                return DaskEngine.from_options(options)
            return thing.make_engine(
                executor_options=options.to_executor_options(),
                engine_options=options.to_engine_options(),
            )
    except ImportError:
        pass
    try:
        from cudf_polars.engine.ray import RayCluster as _RayCluster, RayEngine

        if isinstance(thing, _RayCluster):
            if rmpf_differ:
                return RayEngine.from_options(options)
            return thing.make_engine(
                executor_options=options.to_executor_options(),
                engine_options=options.to_engine_options(),
            )
    except ImportError:
        pass
    raise RuntimeError(f"Unexpected unconfigured engine type: {type(thing)}")


@pytest.fixture
def spmd_engine(
    _unconfigured_engine: tuple[SPMDCluster, StreamingOptions],
) -> Generator[SPMDEngine, None, None]:
    """Yield a per-test :class:`SPMDEngine` from the shared cluster."""
    from cudf_polars.engine.spmd import SPMDEngine as _SPMDEngine

    thing, options = _unconfigured_engine
    engine = _engine_from_unconfigured(thing, options)
    assert isinstance(engine, _SPMDEngine)
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.fixture
def spmd_engine_factory(
    _unconfigured_engine: tuple[SPMDCluster, StreamingOptions],
) -> Generator[Callable[[StreamingOptions], SPMDEngine], None, None]:
    """
    Yield a function that, when called, produces a :class:`SPMDEngine`.

    Use this in place of :func:`streaming_engine_factory` for tests that
    must run on SPMD only.
    """
    thing, base = _unconfigured_engine
    owned: list[SPMDEngine] = []

    def factory(options: StreamingOptions) -> SPMDEngine:
        from cudf_polars.engine.spmd import SPMDEngine as _SPMDEngine

        engine = _engine_from_unconfigured(
            thing, merge_streaming_options(base, options)
        )
        assert isinstance(engine, _SPMDEngine)
        if engine._owns_cluster:
            owned.append(engine)
        return engine

    try:
        yield factory
    finally:
        for eng in owned:
            eng.shutdown()


@pytest.fixture
def ray_engine(
    _unconfigured_engine: tuple[RayCluster, StreamingOptions],
) -> Generator[RayEngine, None, None]:
    """Yield a per-test :class:`RayEngine` from the shared cluster."""
    from cudf_polars.engine.ray import RayEngine as _RayEngine

    thing, options = _unconfigured_engine
    engine = _engine_from_unconfigured(thing, options)
    assert isinstance(engine, _RayEngine)
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.fixture
def dask_engine(
    _unconfigured_engine: tuple[DaskCluster, StreamingOptions],
) -> Generator[DaskEngine, None, None]:
    """Yield a per-test :class:`DaskEngine` from the shared cluster."""
    from cudf_polars.engine.dask import DaskEngine as _DaskEngine

    thing, options = _unconfigured_engine
    engine = _engine_from_unconfigured(thing, options)
    assert isinstance(engine, _DaskEngine)
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.fixture
def streaming_engine_factory(
    _unconfigured_engine: tuple[
        StreamingEngine | SPMDCluster | DaskCluster | RayCluster, StreamingOptions
    ],
) -> Generator[Callable[[StreamingOptions], StreamingEngine], None, None]:
    """
    Yield a function that, when called, produces a :class:`StreamingEngine`.

    Engines that own their own cluster (created when rapidsmpf options differ
    from the session cluster) are tracked and shut down after the test.
    """
    thing, base = _unconfigured_engine
    owned: list[StreamingEngine] = []

    def factory(options: StreamingOptions) -> StreamingEngine:
        engine = _engine_from_unconfigured(
            thing, merge_streaming_options(base, options)
        )
        if getattr(engine, "_owns_cluster", False):
            owned.append(engine)
        return engine

    try:
        yield factory
    finally:
        for eng in owned:
            eng.shutdown()


@pytest.fixture
def streaming_engine(
    _unconfigured_engine: tuple[
        StreamingEngine | SPMDCluster | DaskCluster | RayCluster, StreamingOptions
    ],
) -> Generator[StreamingEngine, None, None]:
    """
    Yield a per-test :class:`StreamingEngine`.

    Inherits the parametrization of ``_unconfigured_engine``, so tests using
    this fixture run once per ``(backend, blocksize_mode)`` combination.
    """
    thing, options = _unconfigured_engine
    engine = _engine_from_unconfigured(thing, options)
    from cudf_polars.engine.spmd import SPMDCluster as _SPMDCluster

    is_cluster_engine = isinstance(thing, _SPMDCluster)
    if not is_cluster_engine:
        try:
            from cudf_polars.engine.dask import DaskCluster as _DaskCluster
            from cudf_polars.engine.ray import RayCluster as _RayCluster

            is_cluster_engine = isinstance(thing, (_DaskCluster, _RayCluster))
        except ImportError:
            pass
    if is_cluster_engine:
        try:
            yield engine
        finally:
            engine.shutdown()
    else:
        yield engine


@pytest.fixture
def engine(
    _unconfigured_engine: tuple[
        pl.GPUEngine | SPMDCluster | DaskCluster | RayCluster, StreamingOptions | None
    ],
) -> Generator[pl.GPUEngine, None, None]:
    """
    Yield a :class:`polars.GPUEngine` for each engine variant under test.

    Every variant is configured with ``raise_on_fail=True``, so an unsupported
    GPU path raises instead of silently falling back to the CPU engine.

    Notes
    -----
    For tests that require a :class:`StreamingEngine` only, use the
    :func:`streaming_engine` fixture instead.
    """
    thing, options = _unconfigured_engine
    if options is None:
        yield thing  # type: ignore[misc]
        return
    eng = _engine_from_unconfigured(thing, options)
    from cudf_polars.engine.spmd import SPMDCluster as _SPMDCluster

    is_cluster_engine = isinstance(thing, _SPMDCluster)
    if not is_cluster_engine:
        try:
            from cudf_polars.engine.dask import DaskCluster as _DaskCluster
            from cudf_polars.engine.ray import RayCluster as _RayCluster

            is_cluster_engine = isinstance(thing, (_DaskCluster, _RayCluster))
        except ImportError:
            pass
    if is_cluster_engine:
        try:
            yield eng
        finally:
            eng.shutdown()
    else:
        yield eng


@pytest.fixture
def in_memory_engine() -> pl.GPUEngine:
    """
    Return an in-memory :class:`polars.GPUEngine` with ``raise_on_fail=True``.

    Use this for tests that intentionally exercise in-memory-only behavior
    instead of the parametrized engine matrix, but generally prefer the ``engine``
    fixture instead.
    """
    return pl.GPUEngine(executor="in-memory", raise_on_fail=True)


@pytest.fixture
def timeout_seconds() -> int:
    """
    Conservative timeout for APIs that accept a timeout parameter.
    """
    return 30


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "skip_on_streaming_engine(reason, *, engine=None): skip the test for "
        'streaming ``engine`` variants (e.g. ``"spmd"``, ``"spmd-small"``, '
        '``"dask"``, ``"ray"``) while still allowing the in-memory variant to run.',
    )
    config.addinivalue_line(
        "markers",
        "engine_params(params): run an engine-fixture test only for the "
        "listed engine parameter ids.",
    )

    # Ray's internal subprocess management leaks `/dev/null` file handles, and
    # distributed's shutdown leaves unclosed sockets. Under Python 3.14 +
    # pytest 9, these surface as unraisable `ResourceWarning`s and, combined
    # with `filterwarnings = ["error", ...]` in pyproject.toml, fail
    # otherwise-unrelated tests when the GC finalizer happens to fire during
    # them. With `pytest-xdist --dist=worksteal`, the leak can land in any
    # test that shares a worker with a ray/dask test, so the suppression must
    # apply globally rather than per-module.
    config.addinivalue_line("filterwarnings", "ignore::ResourceWarning")
    # https://github.com/open-telemetry/opentelemetry-python/issues/5231 (used by Ray)
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Parametrize the shared engine fixture without cartesian products."""
    fixtures = set(metafunc.fixturenames)
    if "_engine_param" not in fixtures:
        return

    if "spmd_engine" in fixtures or "spmd_engine_factory" in fixtures:
        engines = ["spmd"]
    elif "ray_engine" in fixtures:
        engines = ["ray"]
    elif "dask_engine" in fixtures:
        engines = ["dask"]
    elif "streaming_engine" in fixtures or "streaming_engine_factory" in fixtures:
        engines = STREAMING_ENGINE_FIXTURE_PARAMS
    elif "engine" in fixtures:
        engines = ALL_ENGINE_FIXTURE_PARAMS
    else:
        raise AssertionError("Unknown engine fixture")

    marker = metafunc.definition.get_closest_marker("engine_params")
    if marker is not None:
        # Selectors must not opt unavailable backends back into the fixture.
        engines = [engine for engine in engines if engine in marker.args[0]]

    if not engines:
        # Empty parameter sets are collection errors in this suite.
        metafunc.parametrize(
            "_engine_param",
            [pytest.param(None, marks=pytest.mark.skip(reason="No available engines"))],
            indirect=True,
            ids=["unavailable"],
            scope="session",
        )
        return

    metafunc.parametrize(
        "_engine_param",
        engines,
        indirect=True,
        ids=engines,
        scope="session",
    )


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
):
    """Apply ``skip_on_streaming_engine`` markers to streaming engine items."""
    for item in items:
        marker = item.get_closest_marker("skip_on_streaming_engine")
        if marker is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        engine_param = callspec.params.get("_engine_param")
        if engine_param is None or engine_param == "in-memory":
            continue
        engine_filter = marker.kwargs.get("engine")
        if engine_filter is not None:
            if isinstance(engine_filter, str):
                engine_filter = (engine_filter,)
            # Strip the ``-small`` suffix so ``"spmd-small"`` matches
            # ``engine=("spmd",)``.
            engine_name = engine_param.removesuffix("-small")
            if engine_name not in engine_filter:
                continue
        reason = (
            marker.args[0]
            if marker.args
            else marker.kwargs.get("reason", "unsupported on streaming engine")
        )
        item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="module")
def parquet_stats_executor() -> concurrent.futures.ThreadPoolExecutor:  # type: ignore[misc]
    """A thread pool to use for cudf-polars status collection."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        yield executor
