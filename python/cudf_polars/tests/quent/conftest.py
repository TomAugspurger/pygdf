# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for schema-generated Quent telemetry."""

from __future__ import annotations

import pytest

import cudf_polars.quent


@pytest.fixture
def quent_context() -> cudf_polars.quent.QuentConfig:
    return cudf_polars.quent.QuentConfig(
        query_group_name="test_query_group",
        query_name="test_query",
    )
