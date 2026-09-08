# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime adapter for schema-generated cudf-polars Quent bindings."""

from __future__ import annotations

import json
import threading
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import uuid
    from typing import TypeVar

    from cudf_polars import _quent as quent_bindings

    HandleT = TypeVar("HandleT")

try:
    from cudf_polars import _quent
except ImportError:  # pragma: no cover - depends on optional extension
    _quent = None  # type: ignore[assignment]


class EntityKind(Enum):
    """Entity types declared by the cudf-polars Quent schema."""

    ENGINE = auto()
    QUERY_GROUP = auto()
    WORKER = auto()
    PLAN = auto()
    OPERATOR = auto()
    PORT = auto()
    THREAD_POOL = auto()
    PROCESSOR = auto()
    MEMORY = auto()
    DATA_CHANNEL = auto()
    QUERY = auto()
    EVALUATE = auto()
    ACTOR = auto()


class QuentSession:
    """Own generated observers, handles, and an in-memory callback exporter."""

    def __init__(self) -> None:
        if _quent is None:
            raise ImportError(
                "Quent tracing requires the cudf-polars Quent extension. "
                "Build python/cudf_polars/quent/bridge with maturin."
            )
        assert _quent is not None
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._handles: dict[tuple[EntityKind, uuid.UUID], Any] = {}
        self._context = _quent.Context(
            _quent.ExporterOptions.callback(self._record_event)
        )
        self._closed = False

    def _record_event(self, payload: str) -> None:
        event = json.loads(payload)
        with self._lock:
            self._events.append({"timestamp": event["timestamp"], "event": event})

    @staticmethod
    def to_uuid(value: uuid.UUID) -> Any:
        """Convert a Python UUID to the generated binding's UUID type."""
        assert _quent is not None
        return _quent.Uuid(str(value))

    def _get_handle(
        self,
        kind: EntityKind,
        identifier: uuid.UUID,
        observer: quent_bindings.Observer[HandleT],
    ) -> HandleT:
        """Return one generated handle for an explicitly selected observer."""
        key = (kind, identifier)
        try:
            return self._handles[key]
        except KeyError:
            handle = observer.create(self.to_uuid(identifier))
            self._handles[key] = handle
            return handle

    def engine(self, identifier: uuid.UUID) -> quent_bindings.EngineHandle:
        return self._get_handle(
            EntityKind.ENGINE, identifier, self._context.engine_observer()
        )

    def query_group(self, identifier: uuid.UUID) -> quent_bindings.QueryGroupHandle:
        return self._get_handle(
            EntityKind.QUERY_GROUP, identifier, self._context.query_group_observer()
        )

    def worker(self, identifier: uuid.UUID) -> quent_bindings.WorkerHandle:
        return self._get_handle(
            EntityKind.WORKER, identifier, self._context.worker_observer()
        )

    def plan(self, identifier: uuid.UUID) -> quent_bindings.PlanHandle:
        return self._get_handle(
            EntityKind.PLAN, identifier, self._context.plan_observer()
        )

    def operator(self, identifier: uuid.UUID) -> quent_bindings.OperatorHandle:
        return self._get_handle(
            EntityKind.OPERATOR, identifier, self._context.operator_observer()
        )

    def port(self, identifier: uuid.UUID) -> quent_bindings.PortHandle:
        return self._get_handle(
            EntityKind.PORT, identifier, self._context.port_observer()
        )

    def thread_pool(self, identifier: uuid.UUID) -> quent_bindings.ThreadPoolHandle:
        return self._get_handle(
            EntityKind.THREAD_POOL, identifier, self._context.thread_pool_observer()
        )

    def processor(self, identifier: uuid.UUID) -> quent_bindings.ProcessorHandle:
        return self._get_handle(
            EntityKind.PROCESSOR, identifier, self._context.processor_observer()
        )

    def memory(self, identifier: uuid.UUID) -> quent_bindings.MemoryHandle:
        return self._get_handle(
            EntityKind.MEMORY, identifier, self._context.memory_observer()
        )

    def data_channel(self, identifier: uuid.UUID) -> quent_bindings.DataChannelHandle:
        return self._get_handle(
            EntityKind.DATA_CHANNEL,
            identifier,
            self._context.data_channel_observer(),
        )

    def query(self, identifier: uuid.UUID) -> quent_bindings.QueryHandle:
        return self._get_handle(
            EntityKind.QUERY, identifier, self._context.query_observer()
        )

    def evaluate(self, identifier: uuid.UUID) -> quent_bindings.EvaluateHandle:
        return self._get_handle(
            EntityKind.EVALUATE, identifier, self._context.evaluate_observer()
        )

    def actor(self, identifier: uuid.UUID) -> quent_bindings.ActorHandle:
        return self._get_handle(
            EntityKind.ACTOR, identifier, self._context.actor_observer()
        )

    def drain(self) -> list[dict[str, Any]]:
        """Close the exporter, wait for delivery, and return buffered events."""
        if not self._closed:
            self._handles.clear()
            self._context.close()
            self._closed = True
        with self._lock:
            events = sorted(self._events, key=lambda item: item["timestamp"])
            self._events.clear()
        return events


__all__ = ["EntityKind", "QuentSession"]
