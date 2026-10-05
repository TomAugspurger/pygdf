#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_python_common.sh test_python_other

rapids-logger "Install cudf-polars Quent wheel"
QUENT_WHEELHOUSE=$(rapids-download-from-github "cudf_polars_quent_wheel")
python -m pip install --no-deps "${QUENT_WHEELHOUSE}"/cudf_polars_quent-*.whl

rapids-logger "Test cudf-polars Quent integration"
exec ./ci/run_cudf_polars_quent_tests.sh
