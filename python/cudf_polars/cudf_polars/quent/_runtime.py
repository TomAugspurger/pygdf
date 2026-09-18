# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime state for schema-generated cudf-polars Quent bindings."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid

    from cudf_polars import _quent as quent_bindings

try:
    from cudf_polars import _quent
except ImportError:  # pragma: no cover - depends on optional extension
    _quent = None  # type: ignore[assignment]


class QuentSession:
    """Own one collector-backed generated context and its active FSM handles."""

    def __init__(self, collector_address: str) -> None:
        if _quent is None:
            raise ImportError(
                "Quent tracing requires the cudf-polars Quent extension. "
                "Build python/cudf_polars/quent/bridge with maturin."
            )
        self._declarations_lock = threading.Lock()
        self._declared: set[tuple[str, uuid.UUID]] = set()
        self._engines: dict[uuid.UUID, quent_bindings.EngineInitHandle] = {}
        self._workers: dict[uuid.UUID, quent_bindings.WorkerInitHandle] = {}
        self._queries: dict[uuid.UUID, quent_bindings.QueryExecutingHandle] = {}
        self._evaluations: dict[uuid.UUID, quent_bindings.EvaluateRunningHandle] = {}
        self._actors: dict[uuid.UUID, quent_bindings.ActorRunningHandle] = {}
        self._context = _quent.Context(
            _quent.ExporterOptions.collector(collector_address)
        )
        self._closed = False

    @property
    def context(self) -> quent_bindings.Context:
        """Return the generated instrumentation context."""
        return self._context

    def declare_once(self, entity_name: str, identifier: uuid.UUID) -> bool:
        """Claim one declaration for an entity in this session."""
        key = (entity_name, identifier)
        with self._declarations_lock:
            if key in self._declared:
                return False
            self._declared.add(key)
            return True

    def close(self) -> None:
        """Close active handles and wait for collector delivery."""
        if self._closed:
            return
        self._engines.clear()
        self._workers.clear()
        self._queries.clear()
        self._evaluations.clear()
        self._actors.clear()
        self._context.close()
        self._closed = True


__all__ = ["QuentSession"]
