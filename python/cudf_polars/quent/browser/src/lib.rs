// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Browser-hosted API facade for cudf-polars Quent traces.

pub mod bundle;
#[cfg(feature = "generate")]
pub mod generate;

use cudf_polars_quent_analyzer::CudfPolarsUiAnalyzer;
use cudf_polars_quent_store::{CudfPolarsEvent, EngineEvent};
use quent_analyzer::{AnalyzerError, AnalyzerResult};
use quent_events::Event;
use quent_query_engine_analyzer::{EngineEntity, QueryEngineModel, QueryEntity, ui::UiAnalyzer};
use quent_query_engine_ui::{
    self as ui, EngineContexts, OperatorFilter, QueryFilter, ServerContract,
};
use quent_ui::{
    entities::request::EntityListRequest,
    timeline::{
        categorical::CategoricalTimelineRequest,
        request::{BulkTimelineRequest, SingleTimelineRequest},
    },
};
use serde::Serialize;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

const MODEL_NAME: &str = "cudf-polars";

#[wasm_bindgen]
pub struct BrowserServer {
    analyzer: CudfPolarsUiAnalyzer,
    engine_id: Uuid,
    context_ids: Vec<Uuid>,
}

#[wasm_bindgen]
impl BrowserServer {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Self, JsValue> {
        let bundle = bundle::decode(bytes)
            .map_err(|error| js_error(format!("unable to decode browser bundle: {error}")))?;
        if bundle.metadata.model_name != MODEL_NAME {
            return Err(js_error(format!(
                "browser bundle model mismatch: expected {MODEL_NAME}, got {}",
                bundle.metadata.model_name
            )));
        }
        let expected_digest = bundle::schema_digest();
        if bundle.metadata.schema_digest != expected_digest {
            return Err(js_error(format!(
                "browser bundle schema mismatch: expected {expected_digest}, got {}",
                bundle.metadata.schema_digest
            )));
        }
        if bundle.metadata.codec != "postcard" {
            return Err(js_error(format!(
                "unsupported cudf-polars event codec: {}",
                bundle.metadata.codec
            )));
        }
        let events: Vec<Event<CudfPolarsEvent>> = postcard::from_bytes(bundle.payload)
            .map_err(|error| js_error(format!("unable to decode cudf-polars events: {error}")))?;
        let engine_id = events
            .iter()
            .find_map(|event| {
                matches!(
                    event.data,
                    CudfPolarsEvent::Engine(EngineEvent::Init { .. })
                )
                .then_some(event.id)
            })
            .ok_or_else(|| js_error("cudf-polars browser bundle contains no engine"))?;
        if engine_id != bundle.metadata.engine_id {
            return Err(js_error(
                "browser bundle engine ID does not match its events",
            ));
        }
        let analyzer = CudfPolarsUiAnalyzer::try_new(engine_id, events.into_iter())
            .map_err(|error| api_error(error.into()))?;
        Ok(Self {
            analyzer,
            engine_id,
            context_ids: bundle.metadata.context_ids,
        })
    }

    #[wasm_bindgen(js_name = listEngines)]
    pub async fn list_engines_js(&self) -> Result<String, JsValue> {
        json_result(ServerContract::list_engines(self, true).await)
    }

    #[wasm_bindgen(js_name = engineContexts)]
    pub async fn engine_contexts_js(&self, engine_id: &str) -> Result<String, JsValue> {
        json_result(
            ServerContract::engine_contexts(self, parse_uuid(engine_id).map_err(api_error)?).await,
        )
    }

    #[wasm_bindgen(js_name = queries)]
    pub async fn queries_js(&self, engine_id: &str) -> Result<String, JsValue> {
        json_result(ServerContract::queries(self, parse_uuid(engine_id).map_err(api_error)?).await)
    }

    #[wasm_bindgen(js_name = query)]
    pub async fn query_js(&self, engine_id: &str, query_id: &str) -> Result<String, JsValue> {
        json_result(
            ServerContract::query(
                self,
                parse_uuid(engine_id).map_err(api_error)?,
                parse_uuid(query_id).map_err(api_error)?,
            )
            .await,
        )
    }

    #[wasm_bindgen(js_name = singleTimeline)]
    pub async fn single_timeline_js(
        &self,
        engine_id: &str,
        request: &str,
    ) -> Result<String, JsValue> {
        json_result(
            ServerContract::single_timeline(
                self,
                parse_uuid(engine_id).map_err(api_error)?,
                parse(request).map_err(api_error)?,
            )
            .await,
        )
    }

    #[wasm_bindgen(js_name = bulkTimelines)]
    pub async fn bulk_timelines_js(
        &self,
        engine_id: &str,
        request: &str,
    ) -> Result<String, JsValue> {
        json_result(
            ServerContract::bulk_timelines(
                self,
                parse_uuid(engine_id).map_err(api_error)?,
                parse(request).map_err(api_error)?,
            )
            .await,
        )
    }

    #[wasm_bindgen(js_name = dataFlowTimeline)]
    pub async fn data_flow_timeline_js(
        &self,
        engine_id: &str,
        request: &str,
    ) -> Result<String, JsValue> {
        json_result(
            ServerContract::data_flow_timeline(
                self,
                parse_uuid(engine_id).map_err(api_error)?,
                parse(request).map_err(api_error)?,
            )
            .await,
        )
    }

    #[wasm_bindgen(js_name = entities)]
    pub async fn entities_js(&self, engine_id: &str, request: &str) -> Result<String, JsValue> {
        json_result(
            ServerContract::entities(
                self,
                parse_uuid(engine_id).map_err(api_error)?,
                parse(request).map_err(api_error)?,
            )
            .await,
        )
    }
}

impl BrowserServer {
    fn analyzer(&self, engine_id: Uuid) -> Result<&CudfPolarsUiAnalyzer, ApiError> {
        (engine_id == self.engine_id)
            .then_some(&self.analyzer)
            .ok_or(ApiError::NotFound)
    }
}

#[async_trait::async_trait]
impl ServerContract for BrowserServer {
    type Error = ApiError;

    async fn list_engines(&self, with_metadata: bool) -> Result<ui::EngineListResponse, ApiError> {
        let items = if with_metadata {
            vec![self.analyzer.query_engine_model().engine()?.to_ui()?]
        } else {
            vec![ui::Engine::new(self.engine_id)]
        };
        Ok(ui::EngineListResponse {
            items,
            initial_group_by_attribute: CudfPolarsUiAnalyzer::engine_initial_group_by_attribute(),
        })
    }

    async fn engine(&self, engine_id: Uuid) -> Result<ui::Engine, ApiError> {
        Ok(self
            .analyzer(engine_id)?
            .query_engine_model()
            .engine()?
            .to_ui()?)
    }

    async fn engine_contexts(&self, engine_id: Uuid) -> Result<EngineContexts, ApiError> {
        self.analyzer(engine_id)?;
        Ok(EngineContexts {
            engine_id,
            context_ids: self.context_ids.clone(),
        })
    }

    async fn queries(&self, engine_id: Uuid) -> Result<ui::QueryListResponse, ApiError> {
        let analyzer = self.analyzer(engine_id)?;
        let items = analyzer
            .query_engine_model()
            .queries()
            .map(QueryEntity::to_ui)
            .collect::<AnalyzerResult<Vec<_>>>()?;
        Ok(ui::QueryListResponse {
            items,
            initial_group_by_attribute: analyzer.query_initial_group_by_attribute(),
        })
    }

    async fn query(&self, engine_id: Uuid, query_id: Uuid) -> Result<ui::QueryBundle, ApiError> {
        Ok(self.analyzer(engine_id)?.query_bundle(query_id)?)
    }

    async fn single_timeline(
        &self,
        engine_id: Uuid,
        request: SingleTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> Result<quent_ui::timeline::response::SingleTimelineResponse, ApiError> {
        Ok(self
            .analyzer(engine_id)?
            .single_resource_timeline(request)?)
    }

    async fn bulk_timelines(
        &self,
        engine_id: Uuid,
        request: BulkTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> Result<quent_ui::timeline::response::BulkTimelinesResponse, ApiError> {
        Ok(self.analyzer(engine_id)?.bulk_resource_timeline(request)?)
    }

    async fn data_flow_timeline(
        &self,
        engine_id: Uuid,
        request: CategoricalTimelineRequest<QueryFilter>,
    ) -> Result<ui::DataFlowTimelineBinned, ApiError> {
        Ok(self.analyzer(engine_id)?.data_flow_timeline(request)?)
    }

    async fn entities(
        &self,
        engine_id: Uuid,
        request: EntityListRequest<QueryFilter, OperatorFilter>,
    ) -> Result<ui::EntityListResponse, ApiError> {
        Ok(self.analyzer(engine_id)?.list_entities(request)?)
    }
}

pub enum ApiError {
    NotFound,
    Unsupported,
    Other(String),
}

impl From<AnalyzerError> for ApiError {
    fn from(error: AnalyzerError) -> Self {
        match error {
            AnalyzerError::Unsupported => Self::Unsupported,
            error => Self::Other(error.to_string()),
        }
    }
}

fn parse_uuid(value: &str) -> Result<Uuid, ApiError> {
    Uuid::parse_str(value).map_err(|error| ApiError::Other(error.to_string()))
}

fn parse<T: serde::de::DeserializeOwned>(body: &str) -> Result<T, ApiError> {
    serde_json::from_str(body).map_err(|error| ApiError::Other(error.to_string()))
}

fn json_result<T: Serialize>(result: Result<T, ApiError>) -> Result<String, JsValue> {
    serde_json::to_string(&result.map_err(api_error)?).map_err(|error| js_error(error.to_string()))
}

fn api_error(error: ApiError) -> JsValue {
    match error {
        ApiError::NotFound => js_error("QUENT_NOT_FOUND:not found"),
        ApiError::Unsupported => js_error("QUENT_UNSUPPORTED:operation is unsupported"),
        ApiError::Other(message) => js_error(message),
    }
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    JsValue::from_str(message.as_ref())
}
