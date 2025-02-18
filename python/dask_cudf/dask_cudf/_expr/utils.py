# Copyright (c) 2025, NVIDIA CORPORATION.

from typing import Any


def _convert_to_list(column: Any) -> list | None:
    if column is None or isinstance(column, list):
        return column
    elif isinstance(column, tuple):
        column = list(column)
    elif hasattr(column, "tolist"):
        column = column.tolist()
    else:
        # we'll assume it's a scalar
        column = [column]
    return column
