#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-datetime-string

# shellcheck source=ci/build_python_common.sh
source ./ci/build_python_common.sh

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin cudf-polars-quent py build"

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

# cudf-polars-quent does not depend on libcudf or any other cudf package,
# so no cudf channels are prepended.
build_conda_package cudf-polars-quent "${RAPIDS_CONDA_BLD_OUTPUT_DIR}"

# The package has no CUDA dependency, so the artifact name has no CUDA suffix.
RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python cudf-polars-quent cudf --stable)"
export RAPIDS_PACKAGE_NAME
