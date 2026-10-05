#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking this script outside the repository root.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

rapids-logger "Create cudf-polars Quent build environment"
. /opt/conda/etc/profile.d/conda.sh

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key build_cudf_polars_quent \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n cudf_polars_quent

# Temporarily allow unbound variables for conda activation.
set +u
conda activate cudf_polars_quent
set -u

rapids-print-env

BRIDGE_DIR="${PWD}/python/cudf_polars/quent/bridge"
TRACKED_STUB="${BRIDGE_DIR}/cudf_polars_quent.pyi"
WHEEL_DIR="${PWD}/cudf-polars-quent-wheel"

mkdir -p "${WHEEL_DIR}"

pushd "${BRIDGE_DIR}"
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo clean -p cudf-polars-quent
python -m maturin build --locked --out "${WHEEL_DIR}"

shopt -s nullglob
generated_stubs=(target/*/build/cudf-polars-quent-*/out/cudf_polars_quent.pyi)
generated_wheels=("${WHEEL_DIR}"/cudf_polars_quent-*.whl)
shopt -u nullglob
if ((${#generated_stubs[@]} == 0)); then
  echo "No generated Quent stub found" >&2
  exit 1
fi
if ((${#generated_wheels[@]} != 1)); then
  echo "Expected one generated Quent wheel, found ${#generated_wheels[@]}" >&2
  exit 1
fi

generated_stub="${generated_stubs[0]}"
for candidate in "${generated_stubs[@]:1}"; do
  if [[ "${candidate}" -nt "${generated_stub}" ]]; then
    generated_stub="${candidate}"
  fi
done

cp "${generated_stub}" "${TRACKED_STUB}"
popd

git diff --exit-code -- "${TRACKED_STUB}"
