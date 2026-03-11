# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracing and monitoring IR execution."""

from __future__ import annotations

import contextlib
import enum
import functools
import os
import time
from typing import TYPE_CHECKING, Any, Concatenate, Literal, ParamSpec

import nvtx
import pynvml

import rmm
import rmm.statistics

from cudf_polars.utils.config import _bool_converter, get_device_handle

try:
    import structlog
except ImportError:
    _HAS_STRUCTLOG = False
else:
    _HAS_STRUCTLOG = True


LOG_TRACES = _HAS_STRUCTLOG and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES", "0")
)
LOG_MEMORY = LOG_TRACES and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES_MEMORY", "1")
)
LOG_DATAFRAMES = LOG_TRACES and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES_DATAFRAMES", "1")
)

CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"

nvtx_annotate_cudf_polars = functools.partial(
    nvtx.annotate, domain=CUDF_POLARS_NVTX_DOMAIN
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import cudf_polars.containers
    from cudf_polars.dsl import ir


class Scope(str, enum.Enum):
    """Scope values for structured logging."""

    PLAN = "plan"
    ACTOR = "actor"
    EVALUATE_IR_NODE = "evaluate_ir_node"
    TABLE_CHUNK = "table_chunk"


@functools.cache
def _getpid() -> int:  # pragma: no cover
    # Gets called for each IR.do_evaluate node, so we'll cache it.
    return os.getpid()


def make_snapshot(
    node_type: type[ir.IR],
    frames: Sequence[cudf_polars.containers.DataFrame],
    extra: dict[str, Any] | None = None,
    *,
    pid: int,
    device_handle: Any | None = None,
    phase: Literal["input", "output"] = "input",
) -> dict:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
    """
    Collect statistics about the evaluation of an IR node.

    Parameters
    ----------
    node_type
        The type of the IR node.
    frames
        The list of DataFrames to capture information for. For ``phase="input"``,
        this is typically the dataframes passed to ``IR.do_evaluate``. For
        ``phase="output"``, this is typically the DataFrame returned from
        ``IR.do_evaluate``.
    extra
        Extra information to log.
    pid
        The ID of the current process. Used for NVML memory usage.
    device_handle
        The pynvml device handle. Used for NVML memory usage.
    phase
        The phase of the evaluation. Either "input" or "output".
    """
    ir_name = node_type.__name__

    d: dict[str, Any] = {
        "type": ir_name,
    }

    if LOG_DATAFRAMES:
        d.update(
            {
                f"count_frames_{phase}": len(frames),
                f"frames_{phase}": [
                    {
                        "shape": frame.table.shape(),
                        "size": sum(
                            col.device_buffer_size() for col in frame.table.columns()
                        ),
                    }
                    for frame in frames
                ],
            }
        )
        d[f"total_bytes_{phase}"] = sum(x["size"] for x in d[f"frames_{phase}"])

    if LOG_MEMORY:
        stats = rmm.statistics.get_statistics()
        if stats:
            d.update(
                {
                    f"rmm_current_bytes_{phase}": stats.current_bytes,
                    f"rmm_current_count_{phase}": stats.current_count,
                    f"rmm_peak_bytes_{phase}": stats.peak_bytes,
                    f"rmm_peak_count_{phase}": stats.peak_count,
                    f"rmm_total_bytes_{phase}": stats.total_bytes,
                    f"rmm_total_count_{phase}": stats.total_count,
                }
            )

        if device_handle is not None:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
            for proc in processes:
                if proc.pid == pid:
                    d[f"nvml_current_bytes_{phase}"] = proc.usedGpuMemory
                    break
    if extra:
        d.update(extra)

    return d


P = ParamSpec("P")
CALLBACK: Callable[[dict[str, Any]], None] | None = None

# Registry of do_evaluate functions left unwrapped when LOG_TRACES was False at
# import time. Used by tracing_enabled() to monkeypatch them after import.
_trace_registry: list = []


def _make_do_evaluate_wrapper(
    func: Callable[Concatenate[type[ir.IR], P], cudf_polars.containers.DataFrame],
) -> Callable[Concatenate[type[ir.IR], P], cudf_polars.containers.DataFrame]:
    if TYPE_CHECKING:
        # Avoid circular import: ir is loaded when decorator is applied or when
        # tracing_enabled() runs.
        import cudf_polars.containers as _containers
        from cudf_polars.dsl import ir as _ir

    @functools.wraps(func)
    def wrapper(
        cls: type[_ir.IR],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> _containers.DataFrame:
        # do this just once
        pynvml.nvmlInit()
        maybe_handle = get_device_handle()
        pid = _getpid()
        log = structlog.get_logger()

        # By convention, all non-dataframe arguments (non-child) come first.
        # Anything remaining is a dataframe, except for 'context' kwarg.
        frames: list[_containers.DataFrame] = (
            list(args) + [v for k, v in kwargs.items() if k != "context"]
        )[cls._n_non_child_args :]  # type: ignore[assignment]

        before_start = time.monotonic_ns()
        before = make_snapshot(
            cls, frames, phase="input", device_handle=maybe_handle, pid=pid
        )
        before_end = time.monotonic_ns()

        # The decorator preserves the exact signature of the original do_evaluate method.
        # Each IR.do_evaluate method is a classmethod that takes the IR class as first
        # argument, followed by the method-specific arguments, and returns a DataFrame.

        start = time.monotonic_ns()
        result = func(cls, *args, **kwargs)
        stop = time.monotonic_ns()

        after_start = time.monotonic_ns()
        after = make_snapshot(
            cls,
            [result],
            phase="output",
            extra={"start": start, "stop": stop},
            device_handle=maybe_handle,
            pid=pid,
        )
        after_end = time.monotonic_ns()
        record = (
            before
            | after
            | {
                "scope": Scope.EVALUATE_IR_NODE.value,
                "overhead_duration": (before_end - before_start)
                + (after_end - after_start),
            }
        )
        if CALLBACK is not None:
            CALLBACK(record)

        log.info("Execute IR", **record)

        return result

    wrapper.__wrapped__ = func
    return wrapper


def log_do_evaluate(
    func: Callable[Concatenate[type[ir.IR], P], cudf_polars.containers.DataFrame],
) -> Callable[Concatenate[type[ir.IR], P], cudf_polars.containers.DataFrame]:
    """
    Decorator for an ``IR.do_evaluate`` method that logs information before and after evaluation.

    Parameters
    ----------
    func
        The ``IR.do_evaluate`` method to wrap.
    """
    if not LOG_TRACES:
        _trace_registry.append(func)
        return func
    else:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
        return _make_do_evaluate_wrapper(func)


def _all_ir_subclasses(cls: type) -> list[type]:
    """Recursively collect cls and all its subclasses."""
    return [cls] + [
        sub for direct in cls.__subclasses__() for sub in _all_ir_subclasses(direct)
    ]


def _patch_do_evaluate_methods(
    *,
    clear_registry: bool,
) -> list[tuple[type, Any]]:
    """
    Patch do_evaluate methods for classes whose impl is in _trace_registry.

    Returns a list of (cls, impl) for each patched class so callers can unpatch.

    If clear_registry is True, _trace_registry is cleared after copying (for
    tracing_enabled). If False, the registry is left intact (for tracing_enabled).
    """
    registry = list(_trace_registry)
    if clear_registry:
        _trace_registry.clear()
    if not registry:
        return []

    from cudf_polars.dsl import ir as ir_module

    with contextlib.suppress(ImportError):
        import cudf_polars.experimental.shuffle  # noqa: F401

    patched: list[tuple[type, Any]] = []
    for cls in _all_ir_subclasses(ir_module.IR):
        do_evaluate = getattr(cls, "do_evaluate", None)
        if do_evaluate is None:
            continue
        impl = getattr(do_evaluate, "__func__", do_evaluate)
        if impl in registry:
            cls.do_evaluate = classmethod(_make_do_evaluate_wrapper(impl))  # type: ignore[attr-defined]
            patched.append((cls, impl))
    return patched


def _unpatch_do_evaluate_methods(patched: list[tuple[type, Any]]) -> None:
    """Restore do_evaluate to the original unwrapped implementation."""
    for cls, impl in patched:
        cls.do_evaluate = classmethod(impl)  # type: ignore[attr-defined]


@contextlib.contextmanager
def tracing_enabled(
    *,
    memory: bool = True,
    dataframes: bool = True,
) -> Generator[None, None, None]:
    """
    Context manager: enable tracing for the duration of the block, then restore.

    On entry, current LOG_TRACES / LOG_MEMORY / LOG_DATAFRAMES are saved and
    tracing is enabled (globals set and do_evaluate methods patched if needed).
    On exit, the previous state is restored (globals and any patches reverted).

    Useful for tests that want tracing only for a specific block without
    affecting the rest of the process or using a subprocess.

    Parameters
    ----------
    memory
        Whether to log RMM/NVML memory while tracing is enabled (default True).
    dataframes
        Whether to log dataframe shapes/sizes while enabled (default True).
    """
    global LOG_TRACES, LOG_MEMORY, LOG_DATAFRAMES  # noqa: PLW0603
    if not _HAS_STRUCTLOG:
        yield
        return

    saved_traces = LOG_TRACES
    saved_memory = LOG_MEMORY
    saved_dataframes = LOG_DATAFRAMES

    LOG_TRACES = True
    LOG_MEMORY = memory
    LOG_DATAFRAMES = dataframes
    patched = _patch_do_evaluate_methods(clear_registry=False)

    try:
        yield
    finally:
        _unpatch_do_evaluate_methods(patched)
        LOG_TRACES = saved_traces
        LOG_MEMORY = saved_memory
        LOG_DATAFRAMES = saved_dataframes


@contextlib.contextmanager
def bound_contextvars(**kwargs: Any) -> Generator[None, None, None]:
    """Wrapper around structlog.contextvars.bound_contextvars."""
    if _HAS_STRUCTLOG:
        with structlog.contextvars.bound_contextvars(**kwargs):
            yield
    else:
        yield


def log(message: str, **kwargs: Any) -> None:
    """Wrapper around structlog.get_logger().info."""
    if _HAS_STRUCTLOG:
        log = structlog.get_logger()
        log.info(message, **kwargs)
