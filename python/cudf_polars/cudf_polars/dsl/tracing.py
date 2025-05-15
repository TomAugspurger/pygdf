# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed tracing for cudf-polars."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import opentelemetry.trace
import opentelemetry.trace.propagation.tracecontext

F = TypeVar("F", bound=Callable[..., Any])


def trace(func: F) -> F:
    """Trace the execution of some IR.do_evaluate call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracer = opentelemetry.trace.get_tracer("cudf_polars")
        with tracer.start_as_current_span(name=func.__qualname__):
            return func(*args, **kwargs)

    # error: Incompatible return value type (got "_Wrapped[[VarArg(Any), KwArg(Any)], Any, [VarArg(Any), KwArg(Any)], Any]", expected "F")  [return-value]
    return wrapper  # type: ignore[return-value]
