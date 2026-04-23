# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parquet footer helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import kvikio
import numpy as np

import pylibcudf as plc

__all__ = [
    "build_parquet_metadata_from_footer_bytes",
    "read_parquet_footer_bytes",
]

_PARQUET_SUFFIX_SIZE = 8
_PARQUET_MAGIC = b"PAR1"


def read_parquet_footer_bytes(path: str) -> bytes:
    """Read the Parquet footer bytes for a single local file."""
    file_size = Path(path).stat().st_size
    if file_size < _PARQUET_SUFFIX_SIZE:
        raise ValueError(f"Invalid parquet file {path!r}: file is too small")

    suffix_buffer = np.empty(_PARQUET_SUFFIX_SIZE, dtype=np.uint8)
    with kvikio.CuFile(path, "r") as file:
        read_size = file.pread(
            suffix_buffer,
            size=_PARQUET_SUFFIX_SIZE,
            file_offset=file_size - _PARQUET_SUFFIX_SIZE,
        ).get()
        if read_size != _PARQUET_SUFFIX_SIZE:
            raise RuntimeError(
                f"Failed to read parquet suffix from {path!r}: read {read_size} bytes"
            )

        suffix = suffix_buffer.tobytes()
        if suffix[4:] != _PARQUET_MAGIC:
            raise ValueError(f"Invalid parquet file {path!r}: missing PAR1 suffix")

        footer_size = int.from_bytes(suffix[:4], byteorder="little")
        footer_offset = file_size - _PARQUET_SUFFIX_SIZE - footer_size
        if footer_offset < 0:
            raise ValueError(
                f"Invalid parquet file {path!r}: footer extends before start of file"
            )

        footer_buffer = np.empty(footer_size, dtype=np.uint8)
        read_size = file.pread(
            footer_buffer,
            size=footer_size,
            file_offset=footer_offset,
        ).get()
        if read_size != footer_size:
            raise RuntimeError(
                f"Failed to read parquet footer from {path!r}: read {read_size} bytes"
            )

    return footer_buffer.tobytes()


def build_parquet_metadata_from_footer_bytes(
    footer_bytes: list[bytes],
) -> plc.io.parquet_metadata.ParquetMetadata:
    """Construct ``ParquetMetadata`` from raw footer bytes."""
    buffers: list[io.BytesIO] = []
    for footer in footer_bytes:
        buf = io.BytesIO()
        buf.write(_PARQUET_MAGIC)
        buf.write(footer)
        buf.write(len(footer).to_bytes(4, "little"))
        buf.write(_PARQUET_MAGIC)
        buf.seek(0)
        buffers.append(buf)
    return plc.io.parquet_metadata.read_parquet_metadata(
        plc.io.SourceInfo(cast(Any, buffers))
    )
