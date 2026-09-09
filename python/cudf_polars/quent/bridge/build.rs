// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    quent_build_info::emit_source();

    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model.yaml");
    println!("cargo:rerun-if-changed={}", model.display());

    let parsed = quent_yaml::parse_from_file(model)?;
    for warning in &parsed.warnings {
        println!("cargo:warning={warning}");
    }

    let generated = quent_instrumentation_build::generate(
        &parsed.schema,
        &quent_instrumentation_build::Options {
            serde: true,
            umbrella_event: true,
            analyzer_package: Some("cudf-polars-quent-analyzer".to_owned()),
            record_derives: &["Clone"],
            ..Default::default()
        },
    )?;
    for warning in generated.warnings {
        println!("cargo:warning={warning}");
    }

    let options = quent_schema_codegen_python::Options {
        module_name: "_quent".to_owned(),
        instrumentation_path: "crate".to_owned(),
        exporters: quent_schema_codegen_python::Exporters {
            ndjson: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let mut bindings = quent_schema_codegen_python::emit(&parsed.schema, &options)?;
    for file in &mut bindings {
        file.content = file.content.replace(
            "pub fn uuid(&self) -> PyUuid",
            "pub fn entity_uuid(&self) -> PyUuid",
        );
        file.content = file.content.replace(
            "#[pymodule(name = \"_quent\")]",
            "#[pyfunction]\n    fn model_qmi() -> PyResult<String> {\n        use quent_instrumentation::Model;\n        let info = quent_instrumentation::build_info::ArtifactInfo::new(\n            crate::CudfPolars::model_info(),\n        );\n        serde_json::to_string_pretty(&info).map_err(|error| {\n            pyo3::exceptions::PyRuntimeError::new_err(error.to_string())\n        })\n    }\n    #[pymodule(name = \"_quent\")]",
        );
        file.content = file.content.replace(
            "pub fn _quent(module: &Bound<'_, PyModule>) -> PyResult<()> {\n",
            "pub fn _quent(module: &Bound<'_, PyModule>) -> PyResult<()> {\n        module.add_function(wrap_pyfunction!(model_qmi, module)?)?;\n",
        );
        file.content = file.content.replace(
            "impl PyUuid {\n",
            "impl PyUuid {\n        #[new]\n        pub fn new(value: &str) -> PyResult<Self> {\n            use std::str::FromStr;\n            let inner = quent_instrumentation::Uuid::from_str(value)\n                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;\n            Ok(Self { inner })\n        }\n",
        );
        file.content = file.content.replace(
            "enum ExporterKind {\n        Noop,\n        Options(quent_io::ExporterOptions),\n    }",
            "enum ExporterKind {\n        Noop,\n        Options(quent_io::ExporterOptions),\n        Callback(Py<PyAny>),\n    }",
        );
        file.content = file.content.replace(
            "pub fn ndjson(output_dir: std::path::PathBuf) -> Self {\n            Self::filesystem(quent_io::FileSystemFormat::Ndjson, output_dir)\n        }",
            "pub fn ndjson(output_dir: std::path::PathBuf) -> Self {\n            Self::filesystem(quent_io::FileSystemFormat::Ndjson, output_dir)\n        }\n        #[staticmethod]\n        pub fn callback(callback: Py<PyAny>) -> Self {\n            Self { inner: ExporterKind::Callback(callback) }\n        }",
        );
        file.content = file.content.replace(
            "pub fn new(options: Option<PyRef<'_, PyExporterOptions>>) -> PyResult<Self> {\n            let result = match options.as_deref().map(|options| &options.inner) {",
            "pub fn new(py: Python<'_>, options: Option<PyRef<'_, PyExporterOptions>>) -> PyResult<Self> {\n            let result = match options.as_deref().map(|options| &options.inner) {",
        );
        file.content = file.content.replace(
            "Some(ExporterKind::Options(options)) => {\n                    <crate::Context<crate::CudfPolars>>::try_new(options.clone())\n                }",
            "Some(ExporterKind::Options(options)) => {\n                    <crate::Context<crate::CudfPolars>>::try_new(options.clone())\n                }\n                Some(ExporterKind::Callback(callback)) => {\n                    let callback = callback.clone_ref(py);\n                    <crate::Context<crate::CudfPolars>>::try_new(\n                        quent_instrumentation::EventCallback::<crate::CudfPolarsEvent>::new(\n                            move |event| {\n                                let payload = serde_json::to_string(&event)\n                                    .expect(\"generated Quent events must serialize\");\n                                Python::attach(|py| {\n                                    let _ = callback.call1(py, (payload,));\n                                });\n                            },\n                        ),\n                    )\n                }",
        );
        file.content = file.content.replace(
            "pub fn close(&mut self) {\n            self.inner.take();\n        }",
            "pub fn close(&mut self) {\n            if let Some(inner) = self.inner.take() {\n                Python::attach(|py| py.detach(|| drop(inner)));\n            }\n        }",
        );
    }
    quent_schema_codegen_python::write_generated_files(&bindings, &out_dir)?;
    let mut stubs = quent_schema_codegen_python::emit_stubs(&parsed.schema, &options)?;
    for file in &mut stubs {
        file.content = file.content.replace(
            "T = TypeVar(\"T\")\n",
            "T = TypeVar(\"T\")\n\ndef model_qmi() -> str: ...\n",
        );
        file.content = file.content.replace(
            "    def uuid(self) -> Uuid: ...",
            "    def entity_uuid(self) -> Uuid: ...",
        );
        file.content = file.content.replace(
            "class Uuid:\n",
            "class Uuid:\n    def __init__(self, value: str) -> None: ...\n",
        );
        file.content = file.content.replace(
            "    def ndjson(output_dir: str | PathLike[str]) -> ExporterOptions: ...\n",
            "    def ndjson(output_dir: str | PathLike[str]) -> ExporterOptions: ...\n    @staticmethod\n    def callback(callback: object) -> ExporterOptions: ...\n",
        );
    }
    quent_schema_codegen_python::write_generated_files(&stubs, &out_dir)?;
    Ok(())
}
