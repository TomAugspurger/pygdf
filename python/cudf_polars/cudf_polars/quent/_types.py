# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudf-polars objects that carry identities for generated Quent handles."""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any

from cudf_polars import __version__

if TYPE_CHECKING:
    import uuid

    from cudf_polars._quent import DynamicAttributeValue


def _new_quent_id() -> uuid.UUID:
    """Create an identity with Quent's generated UUID helper."""
    from cudf_polars import _quent

    return _quent.now_v7()


class Backend(enum.StrEnum):
    """Supported cudf-polars distributed execution backends."""

    UNKNOWN = "unknown"
    SPMD = "spmd"
    RAY = "ray"
    DASK = "dask"


class DataChannelType(enum.StrEnum):
    """Kinds of concrete worker data paths."""

    DISK_TO_DEVICE = "disk-to-device"
    INTER_RANK = "inter-rank"


@dataclasses.dataclass(frozen=True, slots=True)
class Implementation:
    """Runtime implementation metadata."""

    name: str = "cudf-polars"
    version: str = __version__
    backend: Backend = Backend.UNKNOWN


@dataclasses.dataclass(frozen=True, slots=True)
class Engine:
    """A cudf-polars streaming engine identity."""

    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)
    implementation: Implementation = dataclasses.field(default_factory=Implementation)


@dataclasses.dataclass(frozen=True, slots=True)
class QueryGroup:
    """A logical group of query executions."""

    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)
    instance_name: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Query:
    """One collect or sink execution."""

    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)
    instance_name: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Worker:
    """A rank that executes a physical plan."""

    id: uuid.UUID
    engine: Engine
    instance_name: str

    @property
    def engine_id(self) -> uuid.UUID:
        return self.engine.id


@dataclasses.dataclass(frozen=True, slots=True)
class Edge:
    """A directed connection between two plan ports."""

    source: Port
    target: Port


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """A logical or physical IR graph."""

    id: uuid.UUID
    query: Query
    parent_plan: Plan | None
    instance_name: str
    edges: list[Edge]
    worker: Worker | None


@dataclasses.dataclass(frozen=True, slots=True)
class Operator:
    """A node in a logical or physical plan."""

    id: uuid.UUID
    plan: Plan
    parent_operators: list[Operator]
    type_name: str
    attributes: dict[str, DynamicAttributeValue] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Port:
    """A named input or output of an operator."""

    id: uuid.UUID
    operator: Operator
    instance_name: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ThreadPool:
    """A worker-local host thread pool."""

    worker_id: uuid.UUID
    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Processor:
    """A host thread resource."""

    thread_pool_id: uuid.UUID
    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeviceMemory:
    """A bounded worker-local device-memory resource."""

    instance_name: str
    worker_id: uuid.UUID
    capacity_bytes: int
    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Storage:
    """A worker-local storage resource with unknown capacity."""

    instance_name: str
    worker_id: uuid.UUID
    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DataChannel:
    """An I/O or inter-rank data path."""

    instance_name: str
    channel_type: DataChannelType
    worker_id: uuid.UUID
    source: DeviceMemory | Storage
    target: DeviceMemory | Storage
    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Evaluate:
    """A synchronous host-side IR evaluation."""

    instance_name: str
    id: uuid.UUID = dataclasses.field(default_factory=_new_quent_id)


def dynamic_attributes(
    values: dict[str, Any],
) -> dict[str, DynamicAttributeValue]:
    """Convert arbitrary plan metadata to generated dynamic scalar values."""
    import json

    result: dict[str, DynamicAttributeValue] = {}
    for key, value in values.items():
        if value is None or isinstance(value, bool | int | float | str):
            result[key] = value
        else:
            result[key] = json.dumps(value, sort_keys=True, default=str)
    return result


__all__ = [
    "Backend",
    "DataChannel",
    "DataChannelType",
    "DeviceMemory",
    "Edge",
    "Engine",
    "Evaluate",
    "Implementation",
    "Operator",
    "Plan",
    "Port",
    "Processor",
    "Query",
    "QueryGroup",
    "Storage",
    "ThreadPool",
    "Worker",
    "dynamic_attributes",
]
