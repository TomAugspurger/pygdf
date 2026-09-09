# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-generated Quent telemetry."""

from __future__ import annotations

import dataclasses
import uuid

import pytest

pytest.importorskip("cudf_polars._quent")

from cudf_polars.quent._context import ProcessorRegistry, QuentContext
from cudf_polars.quent._runtime import EntityKind, QuentSession
from cudf_polars.quent._types import (
    Edge,
    Operator,
    Plan,
    Port,
    Query,
    ThreadPool,
    Worker,
    dynamic_attributes,
)


def _events(session: QuentSession) -> list[dict]:
    return [item["event"] for item in session.drain()]


def test_context_lifecycle_uses_generated_handles(
    quent_context: QuentContext,
) -> None:
    session = QuentSession()
    query = quent_context.query_for(uuid.uuid4())

    quent_context._emit_engine_init_events(session)
    quent_context._emit_query_group_events(session)
    quent_context._emit_query_events(session, query)
    quent_context._emit_query_exit_events(session, query)
    quent_context._emit_engine_exit_events(session)

    events = _events(session)
    assert [next(iter(event["data"])) for event in events] == [
        "Engine",
        "QueryGroup",
        "Query",
        "Query",
        "Query",
        "Query",
        "Engine",
    ]
    query_events = [event for event in events if "Query" in event["data"]]
    assert [
        (
            next(iter(event["data"]["Query"]))
            if isinstance(event["data"]["Query"], dict)
            else event["data"]["Query"]
        )
        for event in query_events
    ] == ["Initialized", "Planning", "Executing", "Exited"]
    assert str(quent_context.engine.id) == events[0]["id"]
    assert str(query.id) == query_events[0]["id"]


def test_query_group_is_declared_once_across_derived_contexts(
    quent_context: QuentContext,
) -> None:
    # Benchmarks re-derive the context per iteration while keeping one session,
    # so the dedupe has to key off the session rather than the context object.
    session = QuentSession()
    quent_context._emit_query_group_events(session)
    for iteration in range(2):
        derived = dataclasses.replace(
            quent_context, query=Query(instance_name=f"Iteration {iteration}")
        )
        derived._emit_query_group_events(session)

    events = _events(session)
    assert [next(iter(event["data"])) for event in events] == ["QueryGroup"]


def test_once_event_is_checked_by_generated_handle() -> None:
    session = QuentSession()
    identifier = uuid.uuid4()
    handle = session.engine(identifier)
    implementation = {
        "name": "cudf-polars",
        "version": "test",
        "backend": "spmd",
        "custom_attributes": {"backend": "spmd"},
    }
    handle.init(instance_name="engine", implementation=implementation)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="already emitted"):
        handle.init(instance_name="engine", implementation=implementation)  # type: ignore[arg-type]
    session.drain()


def test_entity_kinds_match_the_declarative_schema() -> None:
    assert {kind.name for kind in EntityKind} == {
        "ENGINE",
        "QUERY_GROUP",
        "WORKER",
        "PLAN",
        "OPERATOR",
        "PORT",
        "THREAD_POOL",
        "PROCESSOR",
        "MEMORY",
        "DATA_CHANNEL",
        "QUERY",
        "EVALUATE",
        "ACTOR",
    }


def test_context_serialization_preserves_shared_identities(
    quent_context: QuentContext,
) -> None:
    result = QuentContext._deserialize(quent_context._serialize())
    assert result.engine == quent_context.engine
    assert result.query_group == quent_context.query_group
    assert result.query == quent_context.query


def test_plan_declarations_reference_generated_entities(
    quent_context: QuentContext,
) -> None:
    session = QuentSession()
    worker = Worker(uuid.uuid4(), quent_context.engine, "rank-0")
    query = quent_context.query_for(uuid.uuid4())
    plan = Plan(uuid.uuid4(), query, None, "logical", [], worker)
    operator = Operator(uuid.uuid4(), plan, [], "Scan", {"node_id": "0"})
    output = Port(uuid.uuid4(), operator, "out")
    consumer = Operator(uuid.uuid4(), plan, [], "Filter", {"node_id": "1"})
    input_ = Port(uuid.uuid4(), consumer, "in")
    plan.edges.append(Edge(output, input_))

    quent_context._emit_plan_declarations(
        session, plan, [operator, consumer], [output, input_]
    )
    events = _events(session)
    plan_event = events[0]["data"]["Plan"]["Declared"]
    assert plan_event["query"]["target"] == str(query.id)
    assert plan_event["edges"] == [
        {
            "source": {"target": str(output.id), "data": None},
            "target": {"target": str(input_.id), "data": None},
        }
    ]
    assert events[1]["data"]["Operator"]["Declared"]["attributes"] == [
        {"key": "node_id", "value": {"String": "0"}}
    ]


def test_processor_registry_declares_each_thread_once() -> None:
    session = QuentSession()
    registry = ProcessorRegistry()
    pool = ThreadPool(worker_id=uuid.uuid4())

    first = registry.get_or_declare_processor(session, 10, pool.id)
    assert registry.get_or_declare_processor(session, 10, pool.id) is first
    second = registry.get_or_declare_processor(session, 11, pool.id)
    assert second != first

    events = _events(session)
    assert len(events) == 2
    assert all("Processor" in event["data"] for event in events)


def test_dynamic_attributes_preserve_scalars_and_encode_structures() -> None:
    assert dynamic_attributes(
        {"name": "scan", "count": 3, "nested": {"column": "x"}}
    ) == {"name": "scan", "count": 3, "nested": '{"column": "x"}'}
