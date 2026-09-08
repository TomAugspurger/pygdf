# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for schema-generated Quent telemetry."""

from __future__ import annotations

import pytest

import cudf_polars.quent


@pytest.fixture
def quent_context() -> cudf_polars.quent.QuentContext:
    return cudf_polars.quent.QuentContext(
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )
