# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cudf_polars.dsl.ir import Empty


def test_empty() -> None:
    # This is hard to test via tha polars API with non-distributed scheduler,
    # so we hit it directly.
    empty = Empty({})
    result = Empty.do_evaluate(empty.schema)
    assert result.column_map == {}
