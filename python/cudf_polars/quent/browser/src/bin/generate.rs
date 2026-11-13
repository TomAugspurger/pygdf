// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use cudf_polars_quent_browser::bundle::{self, Capabilities, FORMAT_VERSION, Metadata};
use cudf_polars_quent_store::{CudfPolars, CudfPolarsEvent, EngineEvent};
use quent_events::Event;
use quent_store::{
    context::ContextSet,
    event::{CombinedEventLoader, filesystem::Loader},
};
use uuid::Uuid;

const QUENT_REVISION: &str = "8551f90c648b25508061f941a6fc43046f739d7a";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .ok_or("usage: cudf-polars-quent-bundle <input-dir-or-zip> <output>")?,
    );
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: cudf-polars-quent-bundle <input-dir-or-zip> <output>")?,
    );
    if args.next().is_some() {
        return Err("usage: cudf-polars-quent-bundle <input-dir-or-zip> <output>".into());
    }

    let extracted;
    let root = if input.is_dir() {
        input.as_path()
    } else {
        extracted = tempfile::tempdir()?;
        extract_zip(&input, extracted.path())?;
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
            context_ids,
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
    std::fs::write(&output, bytes)?;
    println!(
        "wrote {} events for engine {} to {}",
        events.len(),
        engine_id,
        output.display()
    );
    Ok(())
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
