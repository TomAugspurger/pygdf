# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Distributed tracing for cudf-polars."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import opentelemetry.trace
import opentelemetry.sdk.trace.id_generator
import opentelemetry.trace.propagation.tracecontext


F = TypeVar("F", bound=Callable[..., Any])


def trace(func: F) -> F:
    """Trace the execution of some IR.do_evaluate call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracer = opentelemetry.trace.get_tracer("cudf_polars")

        parent_id = kwargs.pop("parent_id", None)

        if parent_id is not None:
            span_id = opentelemetry.sdk.trace.id_generator.RandomIdGenerator().generate_trace_id()
            span_context = opentelemetry.trace.SpanContext(
                trace_id=parent_id,
                span_id=span_id,
                is_remote=True,
                # trace_flags=opentelemetry.trace.TraceFlags.SAMPLED,
            )
            ctx = opentelemetry.trace.set_span_in_context(
                opentelemetry.trace.NonRecordingSpan(span_context)
            )
        else:
            ctx = None

        with tracer.start_as_current_span(name=func.__qualname__, context=ctx):
            return func(*args, **kwargs)

    # error: Incompatible return value type (got "_Wrapped[[VarArg(Any), KwArg(Any)], Any, [VarArg(Any), KwArg(Any)], Any]", expected "F")  [return-value]
    return wrapper  # type: ignore[return-value]
