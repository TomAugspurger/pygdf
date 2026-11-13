// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/cudfpolars.rs"));

mod generated_python {
    include!(concat!(env!("OUT_DIR"), "/pyo3_bridge.rs"));

    pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        __quent_pyo3_bridge::_quent_generated(module)
    }
}

#[pyfunction]
fn write_browser_bundle(
    input: &str,
    output: &str,
) -> PyResult<(String, usize, usize, String, u32)> {
    let summary =
        cudf_polars_quent_browser::generate::generate_bundle(Path::new(input), Path::new(output))
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok((
        summary.engine_id.to_string(),
        summary.context_count,
        summary.event_count,
        summary.schema_digest,
        summary.format_version,
    ))
}

#[pymodule]
fn cudf_polars_quent(module: &Bound<'_, PyModule>) -> PyResult<()> {
    generated_python::register(module)?;
    module.add_function(wrap_pyfunction!(write_browser_bundle, module)?)?;
    Ok(())
}
