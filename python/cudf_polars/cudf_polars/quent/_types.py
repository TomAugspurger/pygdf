# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import dataclasses
import enum
import sys
import time
import uuid
from typing import Any, Literal

from cudf_polars import __version__

QUENT_SCOPE = "QUENT"


class EventName(enum.Enum):
    """Quent event names."""

    ENGINE = "Engine"
    WORKER = "Worker"
    QUERY_GROUP = "QueryGroup"
    QUERY = "Query"
    PLAN = "Plan"
    OPERATOR = "Operator"
    PORT = "Port"
    TASK = "Task"
    MEMORY = "Memory"
    CHANNEL = "Channel"
    THREAD_POOL = "ThreadPool"
    PROCESSOR = "Processor"


if sys.version_info >= (3, 14):  # pragma: no cover; requires Python 3.14+
    new_quent_id = uuid.uuid7
else:  # pragma: no cover; requires Python 3.13 or earlier
    new_quent_id = uuid.uuid4


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """Quent event envelope: id + timestamp + data payload."""

    id: uuid.UUID
    timestamp: int
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Implementation:
    """Engine implementation metadata."""

    name: str = "cudf-polars"
    version: str = __version__
    custom_attributes: list[Any] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "name": self.name,
            "version": self.version,
            "custom_attributes": self.custom_attributes,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class StatisticsAttribute:
    """Typed key/value pair for Quent statistics custom attributes."""

    key: str
    value_type: Literal["U64", "F64", "String"]
    value: int | float | str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": {self.value_type: self.value}}


@dataclasses.dataclass(frozen=True, slots=True)
class Statistics:
    """Operator statistics payload."""

    input_bytes: int
    output_bytes: int
    output_rows: int
    custom_attributes: list[StatisticsAttribute] = dataclasses.field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to Quent's custom attributes format."""
        base_attributes: list[StatisticsAttribute] = [
            StatisticsAttribute(
                key="input_bytes", value_type="U64", value=self.input_bytes
            ),
            StatisticsAttribute(
                key="output_bytes", value_type="U64", value=self.output_bytes
            ),
            StatisticsAttribute(
                key="output_rows", value_type="U64", value=self.output_rows
            ),
        ]
        return {
            "custom_attributes": [
                *(attribute.to_dict() for attribute in base_attributes),
                *(attribute.to_dict() for attribute in self.custom_attributes),
            ]
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Operator:
    """
    A Quent Operator.

    Parameters
    ----------
    parent_operators: list[Operator]
        The operators that are the parents of this operator.
        Note that these are *not* related to the children from cudf-polars' IR.
        Instead, this expresses some kind of lowering relationship (i.e. this node
        was lowered from the given operators).

    Examples
    --------
    {"id":"019dd571-1062-77c2-9803-62c7c1941381","timestamp":1777402450018384340,"data":{"Operator":{"Declaration":{"plan_id":"019dd571-1062-77c2-9803-642b6c301d29","parent_operator_ids":[],"instance_name":"Scan-NodeIndex(0)","type_name":"Scan","custom_attributes":[]}}}}
    """

    id: uuid.UUID
    plan: Plan
    parent_operators: list[Operator]
    instance_name: str
    type_name: str
    custom_attributes: list[Any] = dataclasses.field(default_factory=list)

    @property
    def plan_id(self) -> uuid.UUID:
        """Compatibility accessor for the operator's plan UUID."""
        return self.plan.id

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "id": str(self.id),
            "plan_id": str(self.plan.id),
            "parent_operator_ids": [
                str(operator.id) for operator in self.parent_operators
            ],
            "instance_name": self.instance_name,
            "type_name": self.type_name,
            "custom_attributes": self.custom_attributes,
        }

    def declare(self, timestamp: int | None = None) -> Event:
        """Declare a Quent Operator."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.OPERATOR.value: {"Declaration": self.to_dict()}},
        )

    def statistics(self, statistics: Statistics, timestamp: int | None = None) -> Event:
        """Emit post-execution operator statistics."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.OPERATOR.value: {"Statistics": statistics.to_dict()}},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Engine:
    """A Quent Engine."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    implementation: Implementation = dataclasses.field(default_factory=Implementation)

    def _init(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent engine init event.

        Examples
        --------
        {"id":"019dd571-105a-7c53-a15b-713cbdd7666b","timestamp":1777402450018164995,"data":{"Engine":{"Init":{"implementation":{"name":"Simulator","version":"0.0.0-PoC","custom_attributes":[]},"instance_name":"holodeck-9dfbdcf7"}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.ENGINE.value: {
                    "Init": {
                        "implementation": self.implementation.to_dict(),
                        "instance_name": f"cudf-polars-{str(self.id)[:8]}",
                    }
                }
            },
        )

    def _exit(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent engine exit event.

        Examples
        --------
        {"id":"019dd571-105a-7c53-a15b-713cbdd7666b","timestamp":1777402451406253343,"data":{"Engine":{"Exit":null}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.ENGINE.value: {"Exit": None}},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Worker:
    """A Quent Worker."""

    id: uuid.UUID
    engine: Engine
    instance_name: str

    @property
    def engine_id(self) -> uuid.UUID:
        """Compatibility accessor for the worker's parent engine UUID."""
        return self.engine.id

    def _init(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent worker init event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-6179ddb14b3d","timestamp":1777402450018191773,"data":{"Worker":{"Init":{"parent_engine_id":"019dd571-105a-7c53-a15b-713cbdd7666b","instance_name":"drone-0"}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.WORKER.value: {
                    "Init": {
                        "parent_engine_id": str(self.engine_id),
                        "instance_name": self.instance_name,
                    }
                }
            },
        )

    def _exit(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent worker exit event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-618b9db790c2","timestamp":1777402451406250693,"data":{"Worker":{"Exit":null}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.WORKER.value: {"Exit": None}},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """
    A Quent Plan.

    Examples
    --------
    {"id":"019dd571-1062-77c2-9803-642b6c301d29","timestamp":1777402450018374018,"data":{"Plan":{"Declaration":{"parent":{"query_id":"019dd571-1062-77c2-9803-62bd37658144","plan_id":null},"instance_name":"logical","edges":[{"source":"019dd571-1062-77c2-9803-62e0c8278c73","target":"019dd571-1062-77c2-9803-62f9fc6f8dd2"},{"source":"019dd571-1062-77c2-9803-6321681d89a7","target":"019dd571-1062-77c2-9803-6337bca9d715"},{"source":"019dd571-1062-77c2-9803-63542123336f","target":"019dd571-1062-77c2-9803-636aad7178d1"},{"source":"019dd571-1062-77c2-9803-637d87343372","target":"019dd571-1062-77c2-9803-6389c3b5868a"},{"source":"019dd571-1062-77c2-9803-63a1b8236d47","target":"019dd571-1062-77c2-9803-63bebe4c1970"},{"source":"019dd571-1062-77c2-9803-63d888c0ba46","target":"019dd571-1062-77c2-9803-63e1b00c3ae0"},{"source":"019dd571-1062-77c2-9803-6400931a8db7","target":"019dd571-1062-77c2-9803-6412f5addd70"}],"worker_id":null}}}}
    """

    id: uuid.UUID
    query: Query | None
    parent_plan: Plan | None
    instance_name: str  # TODO: Literal? logical / physical
    edges: list[Edge]
    worker: Worker | None

    @property
    def query_id(self) -> uuid.UUID | None:
        """Compatibility accessor for the parent query UUID."""
        return self.query.id if self.query is not None else None

    @property
    def parent_plan_id(self) -> uuid.UUID | None:
        """Compatibility accessor for the parent plan UUID."""
        return self.parent_plan.id if self.parent_plan is not None else None

    @property
    def worker_id(self) -> uuid.UUID | None:
        """Compatibility accessor for the worker UUID."""
        return self.worker.id if self.worker is not None else None

    def declare(self, timestamp: int | None = None) -> Event:
        """Declare a Quent Plan."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.PLAN.value: {
                    "Declaration": {
                        "parent": {
                            "query_id": str(self.query.id)
                            if self.query is not None
                            else None,
                            "plan_id": str(self.parent_plan.id)
                            if self.parent_plan is not None
                            else None,
                        },
                        "instance_name": self.instance_name,
                        "edges": [
                            {
                                "source": str(edge.source.id),
                                "target": str(edge.target.id),
                            }
                            for edge in self.edges
                        ],
                        "worker_id": str(self.worker.id)
                        if self.worker is not None
                        else None,
                    }
                }
            },
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Edge:
    """Plan edge connecting a source port to a target port."""

    source: Port
    target: Port


@dataclasses.dataclass(frozen=True, slots=True)
class Port:
    """Plan port."""

    id: uuid.UUID
    operator: Operator
    instance_name: str

    def declare(self, timestamp: int | None = None) -> Event:
        """
        Declare a Quent Port.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62e0c8278c73","timestamp":1777402450018384708,"data":{"Port":{"Declaration":{"operator_id":"019dd571-1062-77c2-9803-62c7c1941381","instance_name":"out"}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.PORT.value: {
                    "Declaration": {
                        "operator_id": str(self.operator.id),
                        "instance_name": self.instance_name,
                    }
                }
            },
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Query:
    """A Quent Query with lifecycle state transitions."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str | None = None

    def _init(self, query_group: QueryGroup, timestamp: int | None = None) -> Event:
        """
        Build a Quent Query Init event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62bd37658144","timestamp":1777402450018294782,"data":{"Query":{"seq":0,"state":{"Init":{"instance_name":"Q0","query_group_id":"019dd571-1062-77c2-9803-62a66b6e0c5f"}}}}}
        """
        name = self.instance_name or self.id.hex[:8]
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.QUERY.value: {
                    "seq": 0,
                    "state": {
                        "Init": {
                            "instance_name": name,
                            "query_group_id": str(query_group.id),
                        }
                    },
                }
            },
        )

    def _planning(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent Query Planning event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62bd37658144","timestamp":1777402450018327459,"data":{"Query":{"seq":1,"state":{"Planning":{}}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.QUERY.value: {"seq": 1, "state": {"Planning": {}}}},
        )

    def _executing(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent Query Executing event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62bd37658144","timestamp":1777402450018327459,"data":{"Query":{"seq":2,"state":{"Executing":{}}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.QUERY.value: {"seq": 2, "state": {"Executing": {}}}},
        )

    def _exit(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent Query Exit event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62bd37658144","timestamp":1777402450365535083,"data":{"Query":{"seq":3,"state":"Exit"}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.QUERY.value: {"seq": 3, "state": "Exit"}},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class QueryGroup:
    """Build a Quent Query Group."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str | None = None

    def _declare(self, engine: Engine, timestamp: int | None = None) -> Event:
        """Declare a Quent QueryGroup."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.QUERY_GROUP.value: {
                    "Declaration": {
                        "instance_name": self.instance_name,
                        "engine_id": str(engine.id),
                    }
                }
            },
        )


# Resource Types


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Memory:
    """A Quent Memory resource."""

    # TODO: I think we'll want multiple of these (host / pinned / device memory)
    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str
    resource_type_name: str  # TODO: see if this is static.
    parent_group_id: uuid.UUID  # TODO: refactor this to be an engine

    def initializing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Memory Initializing event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.MEMORY.value: {
                    "seq": 0,
                    "state": {
                        "MemoryInitializing": {
                            "instance_name": self.instance_name,
                            "parent_group_id": str(self.parent_group_id),
                            "resource_type_name": self.resource_type_name,
                        }
                    },
                }
            },
        )

    def operating(self, capacity_bytes: int, timestamp: int | None = None) -> Event:
        """Build a Quent Memory Operating event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.MEMORY.value: {
                    "seq": 1,
                    "state": {"MemoryOperating": {"capacity_bytes": capacity_bytes}},
                }
            },
        )

    def finalizing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Memory Finalizing event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.MEMORY.value: {"seq": 2, "state": {"MemoryFinalizing": None}}
            },
        )

    def exit(self, timestamp: int | None = None) -> Event:
        """Build a Quent Memory Exit event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={EventName.MEMORY.value: {"seq": 3, "state": "Exit"}},
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Channel:
    """
    A Quent Channel resource.

    A Channel is a unidirectional data-transfer resource between two entities.
    Examples include disk-to-device I/O channels and inter-rank network links.

    Parameters
    ----------
    source
        The entity this channel receives from (e.g. a Filesystem Memory).
    target
        The entity this channel sends to (e.g. a Device Memory).
    parent_group_id
        The resource group this channel belongs to (e.g. a Worker or Network ID).
    """

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str
    resource_type_name: str
    parent_group_id: uuid.UUID
    source: Memory
    target: Memory

    def initializing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Channel Initializing event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.CHANNEL.value: {
                    "seq": 0,
                    "state": {
                        "ChannelInitializing": {
                            "instance_name": self.instance_name,
                            "parent_group_id": str(self.parent_group_id),
                            "resource_type_name": self.resource_type_name,
                            "source_id": str(self.source.id),
                            "target_id": str(self.target.id),
                        }
                    },
                }
            },
        )

    def operating(
        self, capacity_bytes: int | None = None, timestamp: int | None = None
    ) -> Event:
        """Build a Quent Channel Operating event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.CHANNEL.value: {
                    "seq": 1,
                    "state": {"ChannelOperating": {"capacity_bytes": capacity_bytes}},
                }
            },
        )

    def finalizing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Channel Finalizing event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.CHANNEL.value: {
                    "seq": 2,
                    "state": {"ChannelFinalizing": None},
                }
            },
        )

    def exit(self, timestamp: int | None = None) -> Event:
        """Build a Quent Channel Exit event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={EventName.CHANNEL.value: {"seq": 3, "state": "Exit"}},
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Network:
    """
    A Quent Network resource group.

    Groups inter-rank Link channels under a single entity.
    """

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    engine_id: uuid.UUID

    def declare(self, timestamp: int | None = None) -> Event:
        """Build a Network declaration event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                "Network": {
                    "Declaration": {
                        "instance_name": "Network",
                        "parent_group_id": str(self.engine_id),
                    }
                }
            },
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ThreadPool:
    # {"id":"019dfdf2-26e2-7a42-af43-ea3e27459b8c","timestamp":1778081998562528225,"data":{"ThreadPool":{"Declaration":{"instance_name":"Thread Pool","parent_group_id":"019dfdf2-26e2-7a42-af43-e9d3ad80cb69"}}}}
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    worker_id: uuid.UUID

    def declare(self, timestamp: int | None = None) -> Event:
        instance_name = f"Thread Pool {self.id.hex[:8]}"
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.THREAD_POOL.value: {
                    "Declaration": {
                        "instance_name": instance_name,
                        "parent_group_id": str(self.worker_id),
                    }
                },
            },
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Processor:
    # {"id":"019dfdf2-26e2-7a42-af43-ea51ff967fa1","timestamp":1778081998562538443,"data":{"Processor":{"seq":0,"state":{"ProcessorInitializing":{"instance_name":"Thread 1","parent_group_id":"019dfdf2-26e2-7a42-af43-ea3e27459b8c","resource_type_name":"processor"}}}}}
    # {"id":"019dfdf2-26e2-7a42-af43-ea51ff967fa1","timestamp":1778081998562538609,"data":{"Processor":{"seq":1,"state":{"ProcessorOperating":null}}}}
    # {"id":"019dfdf2-26e2-7a42-af43-ea51ff967fa1","timestamp":1778081999758405882,"data":{"Processor":{"seq":2,"state":{"ProcessorFinalizing":null}}}}
    # {"id":"019dfdf2-26e2-7a42-af43-ea51ff967fa1","timestamp":1778081999758405986,"data":{"Processor":{"seq":3,"state":"Exit"}}}
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    pool_id: uuid.UUID

    def initializing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Processor Initializing event."""
        instance_name = f"Thread {self.id.hex[:8]}"
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.PROCESSOR.value: {
                    "seq": 0,
                    "state": {
                        "ProcessorInitializing": {
                            "instance_name": instance_name,
                            "parent_group_id": str(self.pool_id),
                            "resource_type_name": "processor",
                        }
                    },
                }
            },
        )

    def operating(self, timestamp: int | None = None) -> Event:
        """Build a Quent Processor Operating event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.PROCESSOR.value: {
                    "seq": 1,
                    "state": {"ProcessorOperating": None},
                }
            },
        )

    def finalizing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Processor Finalizing event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.PROCESSOR.value: {
                    "seq": 2,
                    "state": {"ProcessorFinalizing": None},
                }
            },
        )

    def exit(self, timestamp: int | None = None) -> Event:
        """Build a Quent Processor Exit event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={EventName.PROCESSOR.value: {"seq": 3, "state": "Exit"}},
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Task:
    """A Quent Task."""

    # e.g.
    # {"id":"019dfdf2-26e2-7a42-af43-ef508b85b019","timestamp":1778081998562999387,"data":{"Task":{"seq":0,"state":{"Queueing":{"instance_name":"task-0","operator_id":"019dfdf2-26e2-7a42-af43-edc157afe24a"}}}}}
    # {"id":"019dfdf2-26e2-7a42-af43-ef508b85b019","timestamp":1778081998563099700,"data":{"Task":{"seq":1,"state":{"Allocating":{"use_thread":{"resource_id":"019dfdf2-26e2-7a42-af43-ea4ed5de211d","capacity":null}}}}}}Q
    # {"id":"019dfdf2-26e2-7a42-af43-ef508b85b019","timestamp":1778081998563159401,"data":{"Task":{"seq":2,"state":{"Loading":{"use_thread":{"resource_id":"019dfdf2-26e2-7a42-af43-ea4ed5de211d","capacity":null},"use_fs_to_mem":{"resource_id":"019dfdf2-26e2-7a42-af43-ea146f707e99","capacity":{"capacity_bytes":648019968}},"use_memory":{"resource_id":"019dfdf2-26e2-7a42-af43-ea071579c1a5","capacity":{"capacity_bytes":648019968}}}}}}}
    # {"id":"019dfdf2-26e2-7a42-af43-ef508b85b019","timestamp":1778081998563241752,"data":{"Task":{"seq":3,"state":{"Computing":{"use_thread":{"resource_id":"019dfdf2-26e2-7a42-af43-ea4ed5de211d","capacity":null},"use_memory":{"resource_id":"019dfdf2-26e2-7a42-af43-ea071579c1a5","capacity":{"capacity_bytes":1944059904}}}}}}}
    # {"id":"019dfdf2-26e2-7a42-af43-ef508b85b019","timestamp":1778081998563245059,"data":{"Task":{"seq":4,"state":"Exit"}}}

    # or:
    # {"id":"019dfdf2-26e3-74b3-b38b-fca60d27a1d1","timestamp":1778081998563029999,"data":{"Task":{"seq":0,"state":{"Queueing":{"instance_name":"task-64","operator_id":"019dfdf2-26e2-7a42-af43-edc157afe24a"}}}}}
    # {"id":"019dfdf2-26e3-74b3-b38b-fca60d27a1d1","timestamp":1778081998563122381,"data":{"Task":{"seq":1,"state":{"Allocating":{"use_thread":{"resource_id":"019dfdf2-26e2-7a42-af43-ea51ff967fa1","capacity":null}}}}}}
    # {"id":"019dfdf2-26e3-74b3-b38b-fca60d27a1d1","timestamp":1778081998563177943,"data":{"Task":{"seq":2,"state":{"Computing":{"use_thread":{"resource_id":"019dfdf2-26e2-7a42-af43-ea51ff967fa1","capacity":null},"use_memory":{"resource_id":"019dfdf2-26e2-7a42-af43-ea071579c1a5","capacity":{"capacity_bytes":0}}}}}}}
    # {"id":"019dfdf2-26e3-74b3-b38b-fca60d27a1d1","timestamp":1778081998563181641,"data":{"Task":{"seq":3,"state":"Exit"}}}

    # So the states are task-specific (spilling, etc.). hmmm.
    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str | None = None
    operator_id: uuid.UUID

    def queueing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Task Queueing event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": 0,
                    "state": {
                        "Queueing": {
                            "instance_name": self.instance_name or self.id.hex[:8],
                            "operator_id": str(self.operator_id),
                        }
                    },
                }
            },
        )

    def allocating(
        self,
        resource_id: uuid.UUID,
        capacity: int | None = None,
        timestamp: int | None = None,
    ) -> Event:
        """Build a Quent Task Allocating event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": 1,
                    "state": {
                        "Allocating": {
                            "use_thread": {
                                "resource_id": str(resource_id),
                                "capacity": capacity,
                            }
                        }
                    },
                }
            },
        )

    def loading(
        self,
        use_thread: Processor | None = None,
        use_channel: Channel | None = None,
        channel_capacity_bytes: int = 0,
        use_memory: Memory | None = None,
        memory_capacity_bytes: int = 0,
        timestamp: int | None = None,
    ) -> Event:
        """Build a Quent Task Loading event."""
        loading_data: dict[str, dict[str, Any]] = {}
        if use_thread is not None:
            loading_data["use_thread"] = {
                "resource_id": str(use_thread.id),
                "capacity": None,
            }
        if use_channel is not None:
            loading_data["use_fs_to_mem"] = {
                "resource_id": str(use_channel.id),
                "capacity": {"capacity_bytes": channel_capacity_bytes},
            }
        if use_memory is not None:
            loading_data["use_memory"] = {
                "resource_id": str(use_memory.id),
                "capacity": {"capacity_bytes": memory_capacity_bytes},
            }
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": 2,
                    "state": {"Loading": loading_data},
                }
            },
        )

    def computing(
        self,
        use_thread: Processor | None = None,
        use_memory: Memory | None = None,
        memory_capacity_bytes: int = 0,
        timestamp: int | None = None,
    ) -> Event:
        """Build a Quent Task Computing event."""
        # TODO: capacity?
        computing_data: dict[str, dict[str, Any]] = {}
        if use_thread is not None:
            computing_data["use_thread"] = {
                "resource_id": str(use_thread.id),
                "capacity": None,
            }
        if use_memory is not None:
            computing_data["use_memory"] = {
                "resource_id": str(use_memory.id),
                "capacity": {"capacity_bytes": memory_capacity_bytes},
            }
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": 3,
                    "state": {"Computing": computing_data},
                }
            },
        )

    def sending(
        self,
        use_thread: Processor | None = None,
        use_link: Channel | None = None,
        link_capacity_bytes: int = 0,
        timestamp: int | None = None,
    ) -> Event:
        """Build a Quent Task Sending event."""
        sending_data: dict[str, dict[str, Any]] = {}
        if use_thread is not None:
            sending_data["use_thread"] = {
                "resource_id": str(use_thread.id),
                "capacity": None,
            }
        if use_link is not None:
            sending_data["use_link"] = {
                "resource_id": str(use_link.id),
                "capacity": {"capacity_bytes": link_capacity_bytes},
            }
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": 4,
                    "state": {"Sending": sending_data},
                }
            },
        )

    def exit(self, timestamp: int | None = None) -> Event:
        """Build a Quent Task Exit event."""
        return Event(
            id=self.id,
            timestamp=timestamp or time.time_ns(),
            data={EventName.TASK.value: {"seq": 5, "state": "Exit"}},
        )
