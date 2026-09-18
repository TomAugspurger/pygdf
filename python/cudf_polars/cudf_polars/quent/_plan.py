# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

from __future__ import annotations

import functools
import json
import uuid
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.explain import SerializablePlan

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._runtime import QuentSession
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

_JOIN_TYPES = frozenset({"Join", "ConditionalJoin"})


def _dynamic_attributes(
    values: dict[str, Any],
) -> dict[str, bool | int | float | str | None]:
    """Convert arbitrary plan metadata to generated dynamic scalar values."""
    return {
        key: (
            value
            if value is None or isinstance(value, bool | int | float | str)
            else json.dumps(value, sort_keys=True, default=str)
        )
        for key, value in values.items()
    }


def emit_plan(
    session: QuentSession,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    query_id: uuid.UUID,
    plan_id: uuid.UUID,
    worker_id: uuid.UUID | None,
    *,
    instance_name: str = "logical",
    parent_plan_id: uuid.UUID | None = None,
    parent_operators_by_node_id: dict[str, list[uuid.UUID]] | None = None,
    emit: bool = True,
) -> dict[str, uuid.UUID]:
    """Build and emit one plan using deterministic entity UUIDs."""
    serializable_plan = SerializablePlan.from_ir(ir, config_options=config_options)
    parent_ops = parent_operators_by_node_id or {}
    operator_by_ir_id: dict[str, uuid.UUID] = {}
    port_lookup: dict[tuple[uuid.UUID, str], uuid.UUID] = {}
    for node_id in sorted(serializable_plan.nodes.keys(), key=int):
        serializable_node = serializable_plan.nodes[node_id]
        operator_id = uuid.uuid5(plan_id, f"operator:{node_id}")
        operator_by_ir_id[node_id] = operator_id
        for port_name in port_names_for_node(
            len(serializable_node.children), serializable_node.type
        ):
            port_lookup[(operator_id, port_name)] = uuid.uuid5(
                operator_id, f"port:{port_name}"
            )
    if not emit:
        return operator_by_ir_id

    edges: list[dict[str, uuid.UUID]] = []
    for node_id in sorted(serializable_plan.nodes.keys(), key=int):
        serializable_node = serializable_plan.nodes[node_id]
        operator_id = operator_by_ir_id[node_id]
        input_port_names = port_names_for_node(
            len(serializable_node.children), serializable_node.type
        )[1:]
        for i, child_id in enumerate(serializable_node.children):
            child_operator_id = operator_by_ir_id[child_id]
            edges.append(
                {
                    "source": port_lookup[(child_operator_id, "out")],
                    "target": port_lookup[(operator_id, input_port_names[i])],
                }
            )

    context = session.context
    context.plan_observer().handle(plan_id).declared(
        instance_name=instance_name,
        query=query_id,
        parent_plan=parent_plan_id,
        worker=worker_id,
        edges=edges,
    )
    for node_id in sorted(serializable_plan.nodes.keys(), key=int):
        serializable_node = serializable_plan.nodes[node_id]
        operator_id = operator_by_ir_id[node_id]
        context.operator_observer().handle(operator_id).declared(
            plan=plan_id,
            parent_operators=parent_ops.get(node_id, []),
            instance_name=f"{serializable_node.type}-{operator_id.hex[:8]}",
            type_name=serializable_node.type,
            attributes=_dynamic_attributes(
                {"node_id": node_id, **serializable_node.properties}
            ),
        )
        for port_name in port_names_for_node(
            len(serializable_node.children), serializable_node.type
        ):
            context.port_observer().handle(
                port_lookup[(operator_id, port_name)]
            ).declared(operator=operator_id, instance_name=port_name)
    return operator_by_ir_id


@functools.cache
def port_names_for_node(n_children: int, node_type: str) -> tuple[str, ...]:
    """Determine port names for an IR node based on its children count and type."""
    if n_children == 0:
        return ("out",)
    elif n_children == 1:
        return (
            "out",
            "in",
        )
    elif n_children == 2 and node_type in _JOIN_TYPES:
        return (
            "out",
            "left",
            "right",
        )
    else:
        return ("out", *tuple(f"in_{i}" for i in range(n_children)))


def build_parent_operators_map(
    node_map: dict[str, list[str]],
    logical_op_by_id: dict[str, uuid.UUID],
) -> dict[str, list[uuid.UUID]]:
    """
    Map physical node IDs to their logical-plan parent operators.

    Parameters
    ----------
    node_map
        Mapping from physical (post-lowering) stable IDs to the
        logical (pre-lowering) stable IDs they were derived from.
    logical_op_by_id
        Mapping from logical stable ID to its operator UUID.

    Returns
    -------
    Mapping from physical stable ID to parent operator UUIDs, with an empty
    list for entries with no parents.
    """
    return {
        physical_sid: [
            logical_op_by_id[sid] for sid in logical_sids if sid in logical_op_by_id
        ]
        for physical_sid, logical_sids in node_map.items()
    }


def build_quent_operator_map(
    ir: IR,
    physical_op_by_id: dict[str, uuid.UUID],
) -> dict[IR, uuid.UUID]:
    """Build a map from IR nodes to their physical-plan operator UUIDs."""
    result: dict[IR, uuid.UUID] = {}
    for node in traversal([ir]):
        stable_id = str(node.get_stable_id())
        if stable_id in physical_op_by_id:
            result[node] = physical_op_by_id[stable_id]
    return result
