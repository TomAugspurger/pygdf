// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{AnalyzerError, AnalyzerResult, Entity, Span};
use quent_dynamic_attributes::DynamicAttributes;
use quent_events::Event;
use quent_model::{FsmEvent, Ref};
use quent_query_engine_analyzer::plain::legacy::{
    InMemoryQueryEngineModel, InMemoryQueryEngineModelBuilder,
};
use quent_query_engine_analyzer::ui::{QuentViewer, UiAnalyzer, ViewerEventStream};
use quent_query_engine_analyzer::{
    EngineEntity, OperatorEntity, PlanEntity, PortEntity, QueryEngineModel, QueryEntity,
    QueryGroupEntity, WorkerEntity,
};
use quent_query_engine_model as query_engine;
use quent_query_engine_ui::{
    DataFlowTimelineBinned, OperatorFilter, QueryBundle, QueryEntities, QueryFilter,
};
use quent_simulator_ui::EntityRef as UiEntityRef;
use quent_store::event::ModelEventLoader;
use quent_ui::{
    ResourceGroupNode, ResourceTree,
    entities::{request::EntityListRequest, response::EntityListResponse},
    quantity::QuantitySpec,
    timeline::{
        categorical::CategoricalTimelineRequest,
        request::{BulkChunkedTimelineRequest, BulkTimelineRequest, SingleTimelineRequest},
        response::{BulkChunkedTimelinesResponse, BulkTimelinesResponse, SingleTimelineResponse},
    },
};
use uuid::Uuid;

mod generated {
    #![allow(dead_code, unused_imports)]
    include!(concat!(env!("OUT_DIR"), "/cudfpolars.rs"));
}

use generated::{
    CudfPolars, CudfPolarsEvent, EngineEvent, OperatorEvent, PlanEvent, PortEvent, QueryEvent,
    QueryGroupEvent, WorkerEvent,
};

/// Entry point discovered by `quent-open`.
pub struct Viewer;

impl QuentViewer for Viewer {
    type Analyzer = CudfPolarsUiAnalyzer;

    fn import_events(
        dir: &std::path::Path,
    ) -> quent_model::io::ImporterResult<ViewerEventStream<Self::Analyzer>> {
        let context_id = dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                quent_model::io::ImporterError::other(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("context path has no UUID file name: {}", dir.display()),
                ))
            })?
            .parse::<Uuid>()
            .map_err(quent_model::io::ImporterError::other)?;
        let root = dir.parent().ok_or_else(|| {
            quent_model::io::ImporterError::other(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("context path has no parent: {}", dir.display()),
            ))
        })?;
        let events = quent_store::event::filesystem::Store::<CudfPolars>::new(root)
            .load_model_events(context_id)
            .map_err(quent_model::io::ImporterError::other)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(quent_model::io::ImporterError::other)?;
        Ok(Box::new(events.into_iter()))
    }
}

/// Query-engine UI adapter for the schema-generated cudf-polars event model.
pub struct CudfPolarsUiAnalyzer {
    model: InMemoryQueryEngineModel,
}

impl UiAnalyzer for CudfPolarsUiAnalyzer {
    type Event = CudfPolarsEvent;
    type EntityRef = UiEntityRef;

    fn try_new(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<Self::Event>>,
    ) -> AnalyzerResult<Self> {
        let mut builder = InMemoryQueryEngineModelBuilder::try_new(engine_id)?;
        for event in events {
            if let Some(event) = to_query_engine_event(event) {
                builder.try_push(event)?;
            }
        }
        Ok(Self {
            model: builder.try_build()?,
        })
    }

    fn extract_engine(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<Self::Event>>,
    ) -> AnalyzerResult<quent_query_engine_ui::Engine> {
        for event in events {
            if event.id == engine_id
                && let CudfPolarsEvent::Engine(EngineEvent::Init {
                    instance_name,
                    implementation,
                }) = event.data
            {
                return Ok(quent_query_engine_ui::Engine {
                    id: engine_id,
                    start_time_unix_ns: Some(event.timestamp),
                    duration_s: None,
                    instance_name: Some(instance_name),
                    implementation: Some(quent_query_engine_ui::EngineImplementationAttributes {
                        name: Some(implementation.name),
                        version: Some(implementation.version),
                        custom_attributes: implementation.custom_attributes.0,
                    }),
                });
            }
        }
        Ok(quent_query_engine_ui::Engine::new(engine_id))
    }

    fn query_bundle(&self, query_id: Uuid) -> AnalyzerResult<QueryBundle<Self::EntityRef>> {
        let view = self.model.query_view(query_id)?;
        let query = view.query(query_id)?;
        let epoch = view.query_epoch(query_id)?;
        let query_group_id = query.query_group_id().ok_or_else(|| {
            AnalyzerError::IncompleteEntity(format!(
                "query {query_id} has no query-group reference"
            ))
        })?;

        let entities = QueryEntities {
            engine: view.engine()?.to_ui()?,
            query_group: view.query_group(query_group_id)?.to_ui(),
            query: query.to_ui()?,
            workers: view
                .workers()
                .map(|worker| (worker.id(), worker.to_ui(epoch)))
                .collect(),
            plans: view.plans().map(|plan| (plan.id(), plan.to_ui())).collect(),
            operators: view
                .operators()
                .map(|operator| (operator.id(), operator.to_ui(epoch)))
                .collect(),
            ports: view
                .ports()
                .map(|port| (port.id(), port.to_ui(epoch)))
                .collect(),
            resource_types: Default::default(),
            resources: Default::default(),
            resource_groups: Default::default(),
            resource_group_types: Default::default(),
            fsm_types: Default::default(),
        };
        let unique_operator_names = view
            .operators()
            .filter_map(|operator| operator.operator_type_name().map(str::to_owned))
            .collect();
        let plan_tree = view.plan_tree(query_id)?.to_ui();
        let resource_tree = ResourceTree::ResourceGroup(ResourceGroupNode {
            id: UiEntityRef::Engine(view.engine()?.id()),
            children: vec![],
        });

        Ok(QueryBundle {
            query_id,
            entities,
            plan_tree,
            resource_tree,
            unique_operator_names,
            quantity_specs: [
                ("bytes".to_owned(), QuantitySpec::bytes()),
                ("unit".to_owned(), QuantitySpec::unit()),
            ]
            .into(),
            start_time_unix_ns: epoch,
            duration_s: quent_time::to_secs(query.span()?.duration()),
        })
    }

    fn query_engine_model(&self) -> &impl QueryEngineModel {
        &self.model
    }

    fn single_resource_timeline(
        &self,
        _request: SingleTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<SingleTimelineResponse> {
        Err(AnalyzerError::Unsupported)
    }

    fn list_entities(
        &self,
        _request: EntityListRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<EntityListResponse> {
        Err(AnalyzerError::Unsupported)
    }

    fn bulk_resource_timeline(
        &self,
        _request: BulkTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<BulkTimelinesResponse> {
        Err(AnalyzerError::Unsupported)
    }

    fn bulk_chunked_resource_timeline(
        &self,
        _request: BulkChunkedTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<BulkChunkedTimelinesResponse> {
        Err(AnalyzerError::Unsupported)
    }

    fn data_flow_timeline(
        &self,
        _request: CategoricalTimelineRequest<QueryFilter>,
    ) -> AnalyzerResult<DataFlowTimelineBinned> {
        Err(AnalyzerError::Unsupported)
    }
}

fn to_query_engine_event(
    event: Event<CudfPolarsEvent>,
) -> Option<Event<query_engine::QueryEngineEvent>> {
    let Event {
        id,
        timestamp,
        data,
    } = event;
    let data = match data {
        CudfPolarsEvent::Engine(event) => query_engine::QueryEngineEvent::Engine(match event {
            EngineEvent::Init {
                instance_name,
                implementation,
            } => query_engine::engine::EngineEvent::Init(query_engine::engine::Init {
                implementation: query_engine::engine::EngineImplementationAttributes {
                    name: Some(implementation.name),
                    version: Some(implementation.version),
                    custom_attributes: implementation.custom_attributes,
                },
                instance_name: Some(instance_name),
            }),
            EngineEvent::Exit => {
                query_engine::engine::EngineEvent::Exit(query_engine::engine::Exit)
            }
        }),
        CudfPolarsEvent::QueryGroup(QueryGroupEvent::Declared {
            instance_name,
            engine,
        }) => query_engine::QueryEngineEvent::QueryGroup(
            query_engine::query_group::QueryGroupEvent::Declaration(
                query_engine::query_group::Declaration {
                    instance_name: instance_name.unwrap_or_default(),
                    engine_id: engine.target,
                },
            ),
        ),
        CudfPolarsEvent::Worker(event) => query_engine::QueryEngineEvent::Worker(match event {
            WorkerEvent::Init {
                instance_name,
                engine,
                ..
            } => query_engine::worker::WorkerEvent::Init(query_engine::worker::Init {
                parent_engine_id: Ref::new(engine.target),
                instance_name,
            }),
            WorkerEvent::Exit => {
                query_engine::worker::WorkerEvent::Exit(query_engine::worker::Exit)
            }
        }),
        CudfPolarsEvent::Plan(PlanEvent::Declared {
            instance_name,
            query,
            parent_plan,
            worker,
            edges,
        }) => query_engine::QueryEngineEvent::Plan(query_engine::plan::PlanEvent::Declaration(
            query_engine::plan::Declaration {
                parent: query_engine::plan::PlanParent {
                    query_id: parent_plan.is_none().then(|| Ref::new(query.target)),
                    plan_id: parent_plan.map(|parent| Ref::new(parent.target)),
                },
                instance_name,
                edges: edges
                    .into_iter()
                    .map(|edge| query_engine::plan::Edge {
                        source: Ref::new(edge.source.target),
                        target: Ref::new(edge.target.target),
                    })
                    .collect(),
                worker_id: worker.map(|worker| Ref::new(worker.target)),
            },
        )),
        CudfPolarsEvent::Operator(event) => query_engine::QueryEngineEvent::Operator(match event {
            OperatorEvent::Declared {
                plan,
                parent_operators,
                instance_name,
                type_name,
                attributes,
            } => query_engine::operator::OperatorEvent::Declaration(
                query_engine::operator::Declaration {
                    plan_id: Ref::new(plan.target),
                    parent_operator_ids: parent_operators
                        .into_iter()
                        .map(|operator| Ref::new(operator.target))
                        .collect(),
                    instance_name,
                    type_name,
                    custom_attributes: attributes,
                },
            ),
            OperatorEvent::Statistics { values } => {
                let mut custom_attributes = DynamicAttributes::new();
                custom_attributes.add_u64("input_bytes", values.input_bytes);
                custom_attributes.add_u64("output_bytes", values.output_bytes);
                if let Some(output_rows) = values.output_rows {
                    custom_attributes.add_u64("output_rows", output_rows);
                }
                custom_attributes.add_u64("chunk_count", values.chunk_count);
                custom_attributes.add_bool("duplicated", values.duplicated);
                if let Some(decision) = values.decision {
                    custom_attributes.add_string("decision", decision);
                }
                query_engine::operator::OperatorEvent::Statistics(
                    query_engine::operator::Statistics { custom_attributes },
                )
            }
        }),
        CudfPolarsEvent::Port(PortEvent::Declared {
            operator,
            instance_name,
        }) => query_engine::QueryEngineEvent::Port(query_engine::port::PortEvent::Declaration(
            query_engine::port::Declaration {
                operator_id: Ref::new(operator.target),
                instance_name,
            },
        )),
        CudfPolarsEvent::Query(event) => {
            let (seq, state) = match event {
                QueryEvent::Initialized {
                    query_group,
                    instance_name,
                } => (
                    0,
                    query_engine::query::QueryTransition::Init(query_engine::query::Init {
                        instance_name,
                        query_group_id: Ref::new(query_group.target),
                    }),
                ),
                QueryEvent::Planning => (
                    1,
                    query_engine::query::QueryTransition::Planning(
                        query_engine::query::Planning {},
                    ),
                ),
                QueryEvent::Executing => (
                    2,
                    query_engine::query::QueryTransition::Executing(
                        query_engine::query::Executing {},
                    ),
                ),
                QueryEvent::Exited => (3, query_engine::query::QueryTransition::Exit),
            };
            query_engine::QueryEngineEvent::Query(FsmEvent { seq, state })
        }
        CudfPolarsEvent::ThreadPool(_)
        | CudfPolarsEvent::Processor(_)
        | CudfPolarsEvent::Memory(_)
        | CudfPolarsEvent::DataChannel(_)
        | CudfPolarsEvent::Evaluate(_)
        | CudfPolarsEvent::Actor(_) => return None,
    };
    Some(Event::new(id, timestamp, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use generated::{Implementation, PlanEvent};
    use quent_events::{EntityRef, Model};

    #[test]
    fn builds_query_bundle_from_generated_events() {
        let engine_id = Uuid::now_v7();
        let worker_id = Uuid::now_v7();
        let query_group_id = Uuid::now_v7();
        let query_id = Uuid::now_v7();
        let plan_id = Uuid::now_v7();
        let events = vec![
            Event::new(
                engine_id,
                1,
                CudfPolarsEvent::Engine(EngineEvent::Init {
                    instance_name: "cudf-polars".to_owned(),
                    implementation: Implementation {
                        name: "cudf-polars".to_owned(),
                        version: "test".to_owned(),
                        backend: "spmd".to_owned(),
                        custom_attributes: DynamicAttributes::new(),
                    },
                }),
            ),
            Event::new(
                worker_id,
                2,
                CudfPolarsEvent::Worker(WorkerEvent::Init {
                    instance_name: "rank-0".to_owned(),
                    engine: EntityRef::new(engine_id, ()),
                    parent_engine_id: engine_id.to_string(),
                }),
            ),
            Event::new(
                query_group_id,
                3,
                CudfPolarsEvent::QueryGroup(QueryGroupEvent::Declared {
                    instance_name: Some("group".to_owned()),
                    engine: EntityRef::new(engine_id, ()),
                }),
            ),
            Event::new(
                query_id,
                4,
                CudfPolarsEvent::Query(QueryEvent::Initialized {
                    instance_name: "iteration-1".to_owned(),
                    query_group: EntityRef::new(query_group_id, ()),
                }),
            ),
            Event::new(query_id, 5, CudfPolarsEvent::Query(QueryEvent::Planning)),
            Event::new(query_id, 6, CudfPolarsEvent::Query(QueryEvent::Executing)),
            Event::new(
                plan_id,
                7,
                CudfPolarsEvent::Plan(PlanEvent::Declared {
                    instance_name: "logical".to_owned(),
                    query: EntityRef::new(query_id, ()),
                    parent_plan: None,
                    worker: Some(EntityRef::new(worker_id, ())),
                    edges: vec![],
                }),
            ),
            Event::new(query_id, 8, CudfPolarsEvent::Query(QueryEvent::Exited)),
            Event::new(worker_id, 9, CudfPolarsEvent::Worker(WorkerEvent::Exit)),
            Event::new(engine_id, 10, CudfPolarsEvent::Engine(EngineEvent::Exit)),
        ];

        let analyzer = CudfPolarsUiAnalyzer::try_new(engine_id, events.into_iter()).unwrap();
        let bundle = analyzer.query_bundle(query_id).unwrap();

        assert_eq!(
            bundle.entities.query.instance_name.as_deref(),
            Some("iteration-1")
        );
        assert_eq!(
            bundle.entities.query_group.instance_name.as_deref(),
            Some("group")
        );
        assert_eq!(bundle.entities.plans.len(), 1);
    }

    #[test]
    fn imports_generated_filesystem_layout() {
        let root = tempfile::tempdir().unwrap();
        let context_id = Uuid::now_v7();
        let context = root.path().join(context_id.to_string());
        let engine_dir = context.join("Engine");
        std::fs::create_dir_all(&engine_dir).unwrap();
        quent_events::build_info::ArtifactInfo::new(CudfPolars::model_info())
            .write_sidecar(&context)
            .unwrap();

        let event = Event::new(
            Uuid::now_v7(),
            1,
            EngineEvent::Init {
                instance_name: "cudf-polars".to_owned(),
                implementation: Implementation {
                    name: "cudf-polars".to_owned(),
                    version: "test".to_owned(),
                    backend: "spmd".to_owned(),
                    custom_attributes: DynamicAttributes::new(),
                },
            },
        );
        std::fs::write(
            engine_dir.join("events.ndjson"),
            format!("{}\n", serde_json::to_string(&event).unwrap()),
        )
        .unwrap();

        let events = Viewer::import_events(&context).unwrap().collect::<Vec<_>>();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].data,
            CudfPolarsEvent::Engine(EngineEvent::Init { .. })
        ));
    }

    #[test]
    fn engine_and_worker_events_are_compatible_with_quent_open_indexer() {
        let engine_id = Uuid::now_v7();
        let worker_id = Uuid::now_v7();
        let engine = Event::new(
            engine_id,
            1,
            EngineEvent::Init {
                instance_name: "cudf-polars".to_owned(),
                implementation: Implementation {
                    name: "cudf-polars".to_owned(),
                    version: "test".to_owned(),
                    backend: "ray".to_owned(),
                    custom_attributes: DynamicAttributes::new(),
                },
            },
        );
        let worker = Event::new(
            worker_id,
            2,
            WorkerEvent::Init {
                instance_name: "rank-0".to_owned(),
                engine: EntityRef::new(engine_id, ()),
                parent_engine_id: engine_id.to_string(),
            },
        );

        let indexed_engine: Event<query_engine::engine::EngineEvent> =
            serde_json::from_value(serde_json::to_value(engine).unwrap()).unwrap();
        let indexed_worker: Event<query_engine::worker::WorkerEvent> =
            serde_json::from_value(serde_json::to_value(worker).unwrap()).unwrap();

        assert_eq!(indexed_engine.id, engine_id);
        let query_engine::worker::WorkerEvent::Init(init) = indexed_worker.data else {
            panic!("worker init event was not preserved");
        };
        assert_eq!(init.parent_engine_id.uuid(), engine_id);
    }

    /// The snake-case stream aliases written by `cudf_polars.quent._export` are
    /// what `quent-open` scans to populate its engine list, so drive the real
    /// indexer rather than trusting the layout.
    #[test]
    fn quent_open_indexer_discovers_engines_from_alias_streams() {
        use quent_query_engine_server::analyzer_cache::index_query_engines;

        let root = tempfile::tempdir().unwrap();
        let engine_id = Uuid::now_v7();
        let worker_id = Uuid::now_v7();
        let context_id = Uuid::now_v7();
        let context = root.path().join(context_id.to_string());

        let engine = Event::new(
            engine_id,
            1,
            EngineEvent::Init {
                instance_name: "cudf-polars".to_owned(),
                implementation: Implementation {
                    name: "cudf-polars".to_owned(),
                    version: "test".to_owned(),
                    backend: "spmd".to_owned(),
                    custom_attributes: DynamicAttributes::new(),
                },
            },
        );
        let worker = Event::new(
            worker_id,
            2,
            WorkerEvent::Init {
                instance_name: "rank-0".to_owned(),
                engine: EntityRef::new(engine_id, ()),
                parent_engine_id: engine_id.to_string(),
            },
        );
        // Attribute-less events reach the indexer as `{"Exit": null}`; see
        // `cudf_polars.quent._export.to_index_line`.
        let engine_exit = serde_json::json!({
            "id": engine_id, "timestamp": 9, "data": { "Exit": null },
        });
        let worker_exit = serde_json::json!({
            "id": worker_id, "timestamp": 8, "data": { "Exit": null },
        });
        for (stream, lines) in [
            (
                "engine",
                [
                    serde_json::to_string(&engine).unwrap(),
                    engine_exit.to_string(),
                ],
            ),
            (
                "worker",
                [
                    serde_json::to_string(&worker).unwrap(),
                    worker_exit.to_string(),
                ],
            ),
        ] {
            let dir = context.join(stream);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("events.ndjson"), format!("{}\n", lines.join("\n"))).unwrap();
        }

        let index = index_query_engines(root.path()).unwrap();
        assert_eq!(index.engine_ids().collect::<Vec<_>>(), vec![engine_id]);
        assert_eq!(index.contexts_of(engine_id), vec![context_id]);
        assert_eq!(index.workers_of(engine_id), [(worker_id, context_id)]);
    }
}
