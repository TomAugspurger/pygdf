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
    from collections.abc import Mapping

    from cudf_polars import _quent as quent_bindings

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
    DEVICE_MEMORY = auto()
    STORAGE = auto()
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
        self._engines: dict[uuid.UUID, quent_bindings.EngineInitHandle] = {}
        self._workers: dict[uuid.UUID, quent_bindings.WorkerInitHandle] = {}
        self._queries: dict[uuid.UUID, quent_bindings.QueryExecutingHandle] = {}
        self._evaluations: dict[uuid.UUID, quent_bindings.EvaluateRunningHandle] = {}
        self._actors: dict[uuid.UUID, quent_bindings.ActorRunningHandle] = {}
        self._declared: set[tuple[EntityKind, uuid.UUID]] = set()
        self._context = _quent.Context(
            _quent.ExporterOptions.callback(self._record_event)
        )
        self._closed = False

    def _record_event(self, payload: str) -> None:
        event = json.loads(payload)
        with self._lock:
            self._events.append({"timestamp": event["timestamp"], "event": event})

    def declare_once(self, kind: EntityKind, identifier: uuid.UUID) -> bool:
        """
        Claim the right to emit an entity's once-only declaration event.

        The generated bindings raise if an entity instance is declared twice,
        so callers reachable more than once per session must gate their emit
        on this. Ownership lives here because the session, not any individual
        caller, is what the generated bindings track once-events against.
        """
        key = (kind, identifier)
        with self._lock:
            if key in self._declared:
                return False
            self._declared.add(key)
            return True

    def _get_handle(
        self,
        kind: EntityKind,
        identifier: uuid.UUID,
        observer: Any,
    ) -> Any:
        """Return one generated handle for an explicitly selected observer."""
        key = (kind, identifier)
        try:
            return self._handles[key]
        except KeyError:
            handle = observer.handle(identifier)
            self._handles[key] = handle
            return handle

    def init_engine(
        self,
        identifier: uuid.UUID,
        *,
        instance_name: str,
        implementation: Mapping[str, object],
    ) -> None:
        """Start an Engine FSM and retain its typed init-state handle."""
        self._engines[identifier] = (
            self._context.engine_observer()
            .handle(identifier)
            .init(
                instance_name=instance_name,
                implementation=implementation,
            )
        )

    def exit_engine(self, identifier: uuid.UUID) -> None:
        """Transition an Engine FSM from init to exit."""
        self._engines.pop(identifier).exit()

    def init_worker(
        self,
        identifier: uuid.UUID,
        *,
        instance_name: str,
        engine: uuid.UUID,
        parent_engine_id: str,
    ) -> None:
        """Start a Worker FSM and retain its typed init-state handle."""
        self._workers[identifier] = (
            self._context.worker_observer()
            .handle(identifier)
            .init(
                instance_name=instance_name,
                engine=engine,
                parent_engine_id=parent_engine_id,
            )
        )

    def exit_worker(self, identifier: uuid.UUID) -> None:
        """Transition a Worker FSM from init to exit."""
        self._workers.pop(identifier).exit()

    def start_query(
        self,
        identifier: uuid.UUID,
        *,
        instance_name: str,
        query_group: uuid.UUID,
    ) -> None:
        """Advance a Query FSM through its immediate setup states."""
        initialized = (
            self._context.query_observer()
            .handle(identifier)
            .initialized(
                instance_name=instance_name,
                query_group=query_group,
            )
        )
        planning = initialized.planning()
        self._queries[identifier] = planning.executing()

    def complete_query(self, identifier: uuid.UUID) -> None:
        """Transition an executing Query FSM to completed."""
        self._queries.pop(identifier).completed()

    def fail_query(self, identifier: uuid.UUID, *, error: str) -> None:
        """Transition an executing Query FSM to failed."""
        self._queries.pop(identifier).failed(error=error)

    def start_evaluate(
        self,
        identifier: uuid.UUID,
        *,
        instance_name: str,
        actor: uuid.UUID,
        io: bool,
        input_bytes: int,
        processor: Mapping[str, object],
        channel: Mapping[str, object] | None,
    ) -> None:
        """Advance an Evaluate FSM through queued into running."""
        queued = (
            self._context.evaluate_observer()
            .handle(identifier)
            .queued(
                instance_name=instance_name,
                actor=actor,
            )
        )
        self._evaluations[identifier] = queued.running(
            io=io,
            input_bytes=input_bytes,
            processor=processor,
            channel=channel,
        )

    def complete_evaluate(self, identifier: uuid.UUID, *, output_bytes: int) -> None:
        """Transition a running Evaluate FSM to completed."""
        self._evaluations.pop(identifier).completed(output_bytes=output_bytes)

    def fail_evaluate(self, identifier: uuid.UUID, *, error: str) -> None:
        """Transition a running Evaluate FSM to failed."""
        self._evaluations.pop(identifier).failed(error=error)

    def start_actor(
        self,
        identifier: uuid.UUID,
        *,
        operator: uuid.UUID,
        worker: uuid.UUID,
    ) -> None:
        """Advance an Actor FSM through started into running."""
        started = (
            self._context.actor_observer()
            .handle(identifier)
            .started(
                operator=operator,
                worker=worker,
            )
        )
        self._actors[identifier] = started.running()

    def complete_actor(
        self, identifier: uuid.UUID, *, values: Mapping[str, object]
    ) -> None:
        """Transition a running Actor FSM to completed."""
        self._actors.pop(identifier).completed(values=values)

    def fail_actor(
        self,
        identifier: uuid.UUID,
        *,
        error: str,
        values: Mapping[str, object],
    ) -> None:
        """Transition a running Actor FSM to failed."""
        self._actors.pop(identifier).failed(error=error, values=values)

    def query_group(self, identifier: uuid.UUID) -> quent_bindings.QueryGroupHandle:
        return self._get_handle(
            EntityKind.QUERY_GROUP, identifier, self._context.query_group_observer()
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

    def device_memory(self, identifier: uuid.UUID) -> quent_bindings.DeviceMemoryHandle:
        return self._get_handle(
            EntityKind.DEVICE_MEMORY,
            identifier,
            self._context.device_memory_observer(),
        )

    def storage(self, identifier: uuid.UUID) -> quent_bindings.StorageHandle:
        return self._get_handle(
            EntityKind.STORAGE, identifier, self._context.storage_observer()
        )

    def data_channel(self, identifier: uuid.UUID) -> quent_bindings.DataChannelHandle:
        return self._get_handle(
            EntityKind.DATA_CHANNEL,
            identifier,
            self._context.data_channel_observer(),
        )

    def drain(self) -> list[dict[str, Any]]:
        """Close the exporter, wait for delivery, and return buffered events."""
        if not self._closed:
            self._handles.clear()
            self._engines.clear()
            self._workers.clear()
            self._queries.clear()
            self._evaluations.clear()
            self._actors.clear()
            self._context.close()
            self._closed = True
        with self._lock:
            events = sorted(self._events, key=lambda item: item["timestamp"])
            self._events.clear()
        return events


__all__ = ["EntityKind", "QuentSession"]
