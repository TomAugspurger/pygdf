# Copyright (c) 2025, NVIDIA CORPORATION.

from typing import TYPE_CHECKING, Any

import pandas.api.types

if TYPE_CHECKING:
    from dask_cudf._expr.collection import CudfFrameBase


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


def _raise_if_object_series(x: "CudfFrameBase", funcname: Any) -> None:
    """
    Utility function to raise an error if an object column does not support
    a certain operation like `mean`.
    """
    if x.ndim == 1 and hasattr(x, "dtype"):
        if x.dtype == object:
            raise ValueError(f"`{funcname}` not supported with object series")
        elif pandas.api.types.is_string_dtype(x):
            raise ValueError(f"`{funcname}` not supported with string series")
