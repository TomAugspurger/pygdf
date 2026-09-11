# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudf-polars orchestration for schema-generated Quent handles."""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from typing import TYPE_CHECKING

from cudf_polars.quent._plan import build_parent_operators_map, build_plan
from cudf_polars.quent._runtime import EntityKind, QuentSession
from cudf_polars.quent._types import (
    Backend,
    DataChannel,
    DataChannelType,
    DeviceMemory,
    Engine,
    Implementation,
    Processor,
    Query,
    QueryGroup,
    Storage,
    ThreadPool,
)
from cudf_polars.utils.config import get_total_device_memory

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._types import Evaluate, Operator, Plan, Port, Worker
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

__all__ = [
    "LocalQuentContext",
    "ProcessorRegistry",
    "QuentContext",
    "QuentIRExecutionContext",
    "QuentSession",
    "WorkerResources",
]


class ProcessorRegistry:
    """Map Python executor threads to generated Processor handles."""

    def __init__(self) -> None:
        self._processors: dict[int, Processor] = {}
        self._lock = threading.Lock()

    def get_or_declare_processor(
        self, session: QuentSession, thread_ident: int, pool_id: uuid.UUID
    ) -> Processor:
        """Get or declare the Processor associated with a host thread."""
        with self._lock:
            if thread_ident in self._processors:
                return self._processors[thread_ident]
            processor = Processor(thread_pool_id=pool_id)
            self._processors[thread_ident] = processor

        session.processor(processor.id).declared(
            instance_name=f"Thread {processor.id.hex[:8]}",
            thread_pool=session.to_uuid(pool_id),
        )
        return processor


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuentContext:
    """Serializable identities shared by all ranks of a streaming engine."""

    engine: Engine = dataclasses.field(default_factory=Engine)
    query_group: QueryGroup = dataclasses.field(default_factory=QueryGroup)
    query: Query = dataclasses.field(default_factory=Query)

    def _serialize(self) -> bytes:
        payload = {
            "engine": {
                "id": int(self.engine.id),
                "implementation": dataclasses.asdict(self.engine.implementation),
            },
            "query_group": {
                "id": int(self.query_group.id),
                "instance_name": self.query_group.instance_name,
            },
            "query": {
                "id": int(self.query.id),
                "instance_name": self.query.instance_name,
            },
        }
        return json.dumps(payload).encode()

    @classmethod
    def _deserialize(cls, data: bytes) -> Self:
        payload = json.loads(data)
        return cls(
            engine=Engine(
                id=uuid.UUID(int=int(payload["engine"]["id"])),
                implementation=Implementation(
                    name=payload["engine"]["implementation"]["name"],
                    version=payload["engine"]["implementation"]["version"],
                    backend=Backend(payload["engine"]["implementation"]["backend"]),
                ),
            ),
            query_group=QueryGroup(
                id=uuid.UUID(int=int(payload["query_group"]["id"])),
                instance_name=payload["query_group"]["instance_name"],
            ),
            query=Query(
                id=uuid.UUID(int=int(payload["query"]["id"])),
                instance_name=payload["query"]["instance_name"],
            ),
        )

    def query_for(self, query_id: uuid.UUID) -> Query:
        """Create a per-collect Query while preserving the configured name."""
        return Query(id=query_id, instance_name=self.query.instance_name)

    def _emit_engine_init_events(
        self, session: QuentSession, *, backend: Backend | None = None
    ) -> None:
        implementation = self.engine.implementation
        session.engine(self.engine.id).init(
            instance_name=f"cudf-polars-{str(self.engine.id)[:8]}",
            implementation={
                "name": implementation.name,
                "version": implementation.version,
                "backend": str(backend or implementation.backend),
                "custom_attributes": {
                    "backend": str(backend or implementation.backend)
                },
            },
        )

    def _emit_engine_exit_events(self, session: QuentSession) -> None:
        session.engine(self.engine.id).exit()

    def _emit_query_group_events(self, session: QuentSession) -> None:
        if not session.declare_once(EntityKind.QUERY_GROUP, self.query_group.id):
            return
        session.query_group(self.query_group.id).declared(
            instance_name=self.query_group.instance_name,
            engine=session.to_uuid(self.engine.id),
        )

    def _emit_query_events(self, session: QuentSession, query: Query) -> None:
        handle = session.query(query.id)
        handle.initialized(
            instance_name=query.instance_name or query.id.hex[:8],
            query_group=session.to_uuid(self.query_group.id),
        )
        handle.planning()
        handle.executing()

    def _emit_query_completed_event(self, session: QuentSession, query: Query) -> None:
        session.query(query.id).completed()

    def _emit_query_failed_event(
        self, session: QuentSession, query: Query, error: BaseException
    ) -> None:
        session.query(query.id).failed(error=str(error))

    def _emit_plan_declarations(
        self,
        session: QuentSession,
        plan: Plan,
        operators: list[Operator],
        ports: list[Port],
    ) -> None:
        session.plan(plan.id).declared(
            instance_name=plan.instance_name,
            query=session.to_uuid(plan.query.id),
            parent_plan=(
                session.to_uuid(plan.parent_plan.id)
                if plan.parent_plan is not None
                else None
            ),
            worker=session.to_uuid(plan.worker.id) if plan.worker is not None else None,
            edges=[
                {
                    "source": session.to_uuid(edge.source.id),
                    "target": session.to_uuid(edge.target.id),
                }
                for edge in plan.edges
            ],
        )
        for operator in operators:
            session.operator(operator.id).declared(
                plan=session.to_uuid(operator.plan.id),
                parent_operators=[
                    session.to_uuid(parent.id) for parent in operator.parent_operators
                ],
                instance_name=f"{operator.type_name}-{operator.id.hex[:8]}",
                type_name=operator.type_name,
                attributes=operator.attributes,
            )
        for port in ports:
            session.port(port.id).declared(
                operator=session.to_uuid(port.operator.id),
                instance_name=port.instance_name,
            )

    def _emit_physical_plan_events(
        self,
        session: QuentSession,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        plan_id: uuid.UUID,
        worker: Worker,
        *,
        parent_plan: Plan,
        node_map: dict[str, list[str]],
        logical_op_by_id: dict[str, Operator],
    ) -> dict[str, Operator]:
        parent_operators = build_parent_operators_map(node_map, logical_op_by_id)
        plan, operators, ports, operator_by_id = build_plan(
            ir,
            config_options,
            query=None,
            plan_id=plan_id,
            worker=worker,
            instance_name="physical",
            parent_plan=parent_plan,
            parent_operators_by_node_id=parent_operators,
        )
        self._emit_plan_declarations(session, plan, operators, ports)
        return operator_by_id

    def _emit_evaluate_begin_events(
        self,
        ir_type: type[IR],
        evaluate: Evaluate,
        execution_context: QuentIRExecutionContext,
        input_frames_bytes: int,
    ) -> None:
        processor = execution_context.get_or_declare_processor(threading.get_ident())
        handle = execution_context.logger.evaluate(evaluate.id)
        handle.queued(
            instance_name=evaluate.instance_name,
            operator=execution_context.logger.to_uuid(evaluate.operator.id),
            worker=execution_context.logger.to_uuid(evaluate.worker.id),
        )
        handle.running(
            io=ir_type.is_io_node,
            input_bytes=input_frames_bytes,
            processor={
                "target": execution_context.logger.to_uuid(processor.id),
                "data": {},
            },
            channel=(
                {
                    "target": execution_context.logger.to_uuid(
                        execution_context.worker_resources.disk_to_device_channel.id
                    ),
                    "data": {"bytes": input_frames_bytes},
                }
                if ir_type.is_io_node
                else None
            ),
        )

    def _emit_evaluate_end_event(
        self,
        evaluate: Evaluate,
        execution_context: QuentIRExecutionContext,
        result: DataFrame | None,
        error: BaseException | None,
    ) -> None:
        handle = execution_context.logger.evaluate(evaluate.id)
        if error is not None:
            handle.failed(error=str(error))
        else:
            assert result is not None
            handle.completed(output_bytes=result._size_bytes)


@dataclasses.dataclass(kw_only=True)
class WorkerResources:
    """Per-worker resource identities and generated declarations."""

    thread_pool: ThreadPool
    processor_registry: ProcessorRegistry
    device_memory: DeviceMemory
    filesystem: Storage
    disk_to_device_channel: DataChannel
    link_channels: dict[int, DataChannel]

    @classmethod
    def build(
        cls,
        instance_suffix: str,
        engine_id: uuid.UUID,
        worker_id: uuid.UUID,
        rank: int,
        nranks: int,
    ) -> Self:
        del engine_id
        device_memory = DeviceMemory(
            instance_name=f"{instance_suffix} device memory",
            worker_id=worker_id,
            capacity_bytes=get_total_device_memory() or 0,
        )
        filesystem = Storage(
            instance_name=f"{instance_suffix} filesystem",
            worker_id=worker_id,
        )
        disk_to_device = DataChannel(
            instance_name=f"{instance_suffix} disk -> device",
            channel_type=DataChannelType.DISK_TO_DEVICE,
            worker_id=worker_id,
            source=filesystem,
            target=device_memory,
        )
        links = {
            target_rank: DataChannel(
                instance_name=f"rank-{rank} -> rank-{target_rank}",
                channel_type=DataChannelType.INTER_RANK,
                worker_id=worker_id,
                source=device_memory,
                target=device_memory,
            )
            for target_rank in range(nranks)
            if target_rank != rank
        }
        return cls(
            thread_pool=ThreadPool(worker_id=worker_id),
            processor_registry=ProcessorRegistry(),
            device_memory=device_memory,
            filesystem=filesystem,
            disk_to_device_channel=disk_to_device,
            link_channels=links,
        )

    def declare(self, session: QuentSession) -> None:
        session.device_memory(self.device_memory.id).declared(
            instance_name=self.device_memory.instance_name,
            worker=session.to_uuid(self.device_memory.worker_id),
            limits={"bytes": self.device_memory.capacity_bytes},
        )
        session.storage(self.filesystem.id).declared(
            instance_name=self.filesystem.instance_name,
            worker=session.to_uuid(self.filesystem.worker_id),
        )
        session.thread_pool(self.thread_pool.id).declared(
            instance_name=f"Thread Pool {self.thread_pool.id.hex[:8]}",
            worker=session.to_uuid(self.thread_pool.worker_id),
        )
        for channel in (self.disk_to_device_channel, *self.link_channels.values()):
            session.data_channel(channel.id).declared(
                instance_name=channel.instance_name,
                channel_type=str(channel.channel_type),
                worker=session.to_uuid(channel.worker_id),
                source=session.to_uuid(channel.source.id),
                target=session.to_uuid(channel.target.id),
            )

    def finalize(self, session: QuentSession) -> None:
        """Finish worker resources; generated plain resources need no exit event."""


@dataclasses.dataclass(kw_only=True)
class LocalQuentContext:
    """Rank-local generated Quent state."""

    context: QuentContext
    query: Query
    worker: Worker
    logger: QuentSession
    worker_resources: WorkerResources

    def get_or_declare_processor(self, thread_ident: int) -> Processor:
        return self.worker_resources.processor_registry.get_or_declare_processor(
            self.logger, thread_ident, self.worker_resources.thread_pool.id
        )


@dataclasses.dataclass(kw_only=True)
class QuentIRExecutionContext(LocalQuentContext):
    """Rank-local state with an Operator bound to the current IR node."""

    quent_operator: Operator

    @classmethod
    def from_execution_context(
        cls, execution_context: LocalQuentContext, quent_operator: Operator
    ) -> Self:
        return cls(
            quent_operator=quent_operator,
            context=execution_context.context,
            query=execution_context.query,
            worker=execution_context.worker,
            logger=execution_context.logger,
            worker_resources=execution_context.worker_resources,
        )
