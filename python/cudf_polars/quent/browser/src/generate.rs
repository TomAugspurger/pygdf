// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, path::Path};

use cudf_polars_quent_store::{CudfPolars, CudfPolarsEvent, EngineEvent};
use quent_events::Event;
use quent_store::{
    context::ContextSet,
    event::{CombinedEventLoader, filesystem::Loader},
};
use uuid::Uuid;

use crate::bundle::{self, Capabilities, FORMAT_VERSION, Metadata};

const QUENT_REVISION: &str = "8551f90c648b25508061f941a6fc43046f739d7a";

/// Convert a completed collector directory or Quent ZIP to a browser bundle.
pub fn generate_bundle(
    input: &Path,
    output: &Path,
) -> Result<GenerationSummary, Box<dyn std::error::Error>> {
    let extracted;
    let root = if input.is_dir() {
        input
    } else {
        extracted = tempfile::tempdir()?;
        extract_zip(input, extracted.path())?;
        extracted.path()
    };
    let context_ids = discover_context_ids(root)?;
    if context_ids.is_empty() {
        return Err(format!("no UUID context directories found under {}", root.display()).into());
    }

    let mut events = Vec::<Event<CudfPolarsEvent>>::new();
    for context_id in &context_ids {
        events.extend(
            Loader::<CudfPolars>::new(root, ContextSet::one(*context_id))
                .combined_events()?
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    events.sort_by_key(|event| (event.timestamp, event.id));
    let engine_ids = events
        .iter()
        .filter_map(|event| {
            matches!(
                event.data,
                CudfPolarsEvent::Engine(EngineEvent::Init { .. })
            )
            .then_some(event.id)
        })
        .collect::<HashSet<_>>();
    if engine_ids.len() != 1 {
        return Err(format!(
            "expected exactly one engine in browser bundle, found {}",
            engine_ids.len()
        )
        .into());
    }
    let engine_id = *engine_ids.iter().next().unwrap();
    let event_count = events.len();
    let payload = postcard::to_allocvec(&events)?;
    let model_provenance = std::fs::read(root.join(context_ids[0].to_string()).join("model.qmi"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    let bytes = bundle::encode(
        &Metadata {
            format_version: FORMAT_VERSION,
            model_name: "cudf-polars".into(),
            schema_digest: bundle::schema_digest(),
            model_provenance,
            analyzer_revision: env!("CARGO_PKG_VERSION").into(),
            quent_revision: QUENT_REVISION.into(),
            codec: "postcard".into(),
            engine_id,
            context_ids: context_ids.clone(),
            capabilities: Capabilities {
                data_flow: false,
                nvtx: false,
            },
        },
        &payload,
    )?;
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, bytes)?;
    Ok(GenerationSummary {
        engine_id,
        context_count: context_ids.len(),
        event_count,
        schema_digest: bundle::schema_digest(),
        format_version: FORMAT_VERSION,
    })
}

#[derive(Clone, Debug)]
pub struct GenerationSummary {
    pub engine_id: Uuid,
    pub context_count: usize,
    pub event_count: usize,
    pub schema_digest: String,
    pub format_version: u32,
}

fn discover_context_ids(root: &Path) -> Result<Vec<Uuid>, std::io::Error> {
    let mut ids = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && let Some(name) = entry.file_name().to_str()
            && let Ok(id) = Uuid::parse_str(name)
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

fn extract_zip(input: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(input)?)?;
    for index in 0..archive.len() {
        let mut file = archive.by_index(index)?;
        let Some(relative) = file.enclosed_name() else {
            return Err(format!("ZIP contains unsafe path: {}", file.name()).into());
        };
        let path = output.join(relative);
        if file.is_dir() {
            std::fs::create_dir_all(path)?;
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::io::copy(&mut file, &mut std::fs::File::create(path)?)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudf_polars_quent_store::Implementation;

    const MODEL_INFO: &str = r#"{"model":{"name":"CudfPolars","package":"test","source":{"version":"test"},"analyzer_package":"cudf-polars-quent-analyzer"},"quent":{"version":"test"}}"#;

    #[test]
    fn converts_a_multi_event_context() {
        let temp = tempfile::tempdir().unwrap();
        let context_id = Uuid::now_v7();
        let engine_id = Uuid::now_v7();
        let context = temp.path().join(context_id.to_string());
        let engine_dir = context.join("Engine");
        std::fs::create_dir_all(&engine_dir).unwrap();
        let events = [
            Event::new(
                engine_id,
                1,
                EngineEvent::Init {
                    seq: 0,
                    instance_name: "test".into(),
                    implementation: Implementation {
                        name: "cudf-polars".into(),
                        version: "test".into(),
                        backend: "spmd".into(),
                        custom_attributes: Default::default(),
                    },
                },
            ),
            Event::new(engine_id, 2, EngineEvent::Exit { seq: 1 }),
        ];
        let ndjson = events
            .iter()
            .map(|event| serde_json::to_string(event).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(engine_dir.join(format!("{engine_id}.ndjson")), ndjson).unwrap();
        std::fs::write(context.join("model.qmi"), MODEL_INFO).unwrap();
        let output = temp.path().join("trace.quent-browser");

        let summary = generate_bundle(temp.path(), &output).unwrap();
        let bytes = std::fs::read(output).unwrap();
        let decoded = bundle::decode(&bytes).unwrap();
        let decoded_events: Vec<Event<CudfPolarsEvent>> =
            postcard::from_bytes(decoded.payload).unwrap();

        assert_eq!(summary.engine_id, engine_id);
        assert_eq!(summary.context_count, 1);
        assert_eq!(summary.event_count, 2);
        assert_eq!(decoded_events.len(), 2);
        assert_eq!(
            decoded.metadata.model_provenance.unwrap()["model"]["name"],
            "CudfPolars"
        );
    }

    #[test]
    fn rejects_context_without_an_engine() {
        let temp = tempfile::tempdir().unwrap();
        let context = temp.path().join(Uuid::now_v7().to_string());
        std::fs::create_dir_all(&context).unwrap();
        std::fs::write(context.join("model.qmi"), MODEL_INFO).unwrap();

        let error =
            generate_bundle(temp.path(), &temp.path().join("trace.quent-browser")).unwrap_err();
        assert!(error.to_string().contains("exactly one engine"));
    }
}
