#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

rapids-logger "pytest cudf-polars Quent"
pushd python/cudf_polars
python ../../ci/timeout_with_stack.py --enable-python 5400 \
  python -m pytest --cache-clear -p no:benchmark \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars-quent.xml" \
  tests/quent
popd
