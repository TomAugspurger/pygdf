// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Span,
    fsm::{FsmStateTypeDecl, FsmTransitionDecl, FsmTypeDecl},
    resource::{CapacityDecl, CapacityValue, ResourceTypeDecl, Usage},
    timeline::binned::resource::{ResourceTimelineBuilder, ResourceTimelineByKeyBuilder},
};
use quent_dynamic_attributes::DynamicAttributes;
use quent_events::Event;
use quent_model::{FsmEvent, Ref};
use quent_query_engine_analyzer::plain::legacy::{
    InMemoryQueryEngineModel, InMemoryQueryEngineModelBuilder,
};
use quent_query_engine_analyzer::ui::{QuentViewer, UiAnalyzer, ViewerEventStream};
use quent_query_engine_analyzer::{
    EngineEntity, OperatorEntity, OperatorEntityMut, PlanEntity, PortEntity, QueryEngineModel,
    QueryEngineModelMut, QueryEntity, QueryGroupEntity, WorkerEntity,
};
use quent_query_engine_model as query_engine;
use quent_query_engine_ui::{
    DataFlowTimelineBinned, OperatorFilter, QueryBundle, QueryEntities, QueryFilter,
};
use quent_simulator_ui::EntityRef as UiEntityRef;
use quent_store::event::ModelEventLoader;
use quent_time::{
    TimeNanoSec, TimeUnixNanoSec, span::SpanUnixNanoSec, to_nanosecs, to_secs_relative,
};
use quent_ui::{
    FiniteStateMachine, FsmTransition, FsmUsage, Resource, ResourceGroup, ResourceGroupNode,
    ResourceGroupTypeDecl, ResourceTree, ResourceTypeDecl as UiResourceTypeDecl,
    entities::{
        request::{EntityListRequest, EntityScope, SortDir},
        response::{EntityListItem, EntityListResponse},
    },
    quantity::{CapacityDecl as UiCapacityDecl, CapacityKind, QuantitySpec},
    timeline::{
        categorical::CategoricalTimelineRequest,
        request::{BulkTimelineRequest, SingleTimelineRequest, TimelineRequest},
        response::{
            BulkTimelinesResponse, BulkTimelinesResponseEntry,
            ResourceTimeline as UiResourceTimeline, ResourceTimelineBinned,
            ResourceTimelineBinnedByState, SingleTimelineResponse,
        },
    },
};
use uuid::Uuid;

mod generated {
    #![allow(dead_code, unused_imports)]
    include!(concat!(env!("OUT_DIR"), "/cudfpolars.rs"));
}

use generated::{
    ActorEvent, CudfPolars, CudfPolarsEvent, DataChannelEvent, EngineEvent, EvaluateEvent,
    OperatorEvent, PlanEvent, PortEvent, ProcessorEvent, QueryEvent, QueryGroupEvent,
    ThreadPoolEvent, WorkerEvent,
};

const ACTOR_SLOT_RESOURCE_TYPE: &str = "actor_slot";
const PROCESSOR_RESOURCE_TYPE: &str = "processor";
const DATA_CHANNEL_RESOURCE_TYPE: &str = "data_channel";
const EVALUATE_ENTITY_TYPE: &str = "Evaluate";

#[derive(Default)]
struct ActorBuilder {
    operator_id: Option<Uuid>,
    worker_id: Option<Uuid>,
    running_at: Option<TimeUnixNanoSec>,
    finished_at: Option<TimeUnixNanoSec>,
}

impl ActorBuilder {
    fn push(&mut self, timestamp: TimeUnixNanoSec, event: &ActorEvent) {
        match event {
            ActorEvent::Started {
                operator, worker, ..
            } => {
                self.operator_id = Some(operator.target);
                self.worker_id = Some(worker.target);
            }
            ActorEvent::Running { .. } => self.running_at = Some(timestamp),
            ActorEvent::Completed { .. } | ActorEvent::Failed { .. } => {
                self.finished_at = Some(timestamp);
            }
        }
    }

    fn try_build(self, id: Uuid) -> AnalyzerResult<Option<ActorSpan>> {
        let (Some(operator_id), Some(worker_id), Some(start), Some(end)) = (
            self.operator_id,
            self.worker_id,
            self.running_at,
            self.finished_at,
        ) else {
            return Ok(None);
        };
        Ok(Some(ActorSpan {
            id,
            operator_id,
            worker_id,
            span: SpanUnixNanoSec::try_new(start, end)?,
            unit: CapacityValue::new("unit", 1),
        }))
    }
}

struct ActorSpan {
    id: Uuid,
    operator_id: Uuid,
    worker_id: Uuid,
    span: SpanUnixNanoSec,
    unit: CapacityValue,
}

struct ActorUsage<'a>(&'a ActorSpan);

impl<'a> Usage<'a> for ActorUsage<'a> {
    fn entity_id(&self) -> Uuid {
        self.0.id
    }

    fn resource_id(&self) -> Uuid {
        actor_slot_id(self.0.worker_id)
    }

    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        std::iter::once(&self.0.unit)
    }

    fn span(&self) -> SpanUnixNanoSec {
        self.0.span
    }
}

fn actor_slot_id(worker_id: Uuid) -> Uuid {
    const ACTOR_SLOT_MASK: u128 = 0xa3c7_0f5d_e912_4bb8_9f61_4a6d_28c0_7e35;
    Uuid::from_u128(worker_id.as_u128() ^ ACTOR_SLOT_MASK)
}

fn actor_slot_resource_type() -> ResourceTypeDecl {
    ResourceTypeDecl::unit(ACTOR_SLOT_RESOURCE_TYPE)
}

fn evaluate_resource_type(name: &str) -> ResourceTypeDecl {
    let mut resource_type = match name {
        PROCESSOR_RESOURCE_TYPE => ResourceTypeDecl::unit(name),
        DATA_CHANNEL_RESOURCE_TYPE => {
            ResourceTypeDecl::new(name, [CapacityDecl::new_rate("bytes")])
        }
        _ => unreachable!("resource type validated by caller"),
    };
    resource_type
        .used_by
        .insert(EVALUATE_ENTITY_TYPE.to_owned());
    resource_type
}

#[derive(Clone)]
struct DeclaredResource {
    id: Uuid,
    instance_name: String,
    type_name: &'static str,
    parent_group_id: Uuid,
}

#[derive(Clone)]
struct DeclaredResourceGroup {
    id: Uuid,
    instance_name: String,
    type_name: &'static str,
    parent_group_id: Uuid,
}

#[derive(Default)]
struct EvaluateBuilder {
    instance_name: Option<String>,
    actor_id: Option<Uuid>,
    processor_id: Option<Uuid>,
    channel: Option<(Uuid, u64)>,
    queued_at: Option<TimeUnixNanoSec>,
    running_at: Option<TimeUnixNanoSec>,
    finished_at: Option<TimeUnixNanoSec>,
    finished_state: Option<&'static str>,
}

impl EvaluateBuilder {
    fn push(&mut self, timestamp: TimeUnixNanoSec, event: &EvaluateEvent) {
        match event {
            EvaluateEvent::Queued {
                instance_name,
                actor,
                ..
            } => {
                self.instance_name = Some(instance_name.clone());
                self.actor_id = Some(actor.target);
                self.queued_at = Some(timestamp);
            }
            EvaluateEvent::Running {
                processor, channel, ..
            } => {
                self.processor_id = Some(processor.target);
                self.channel = channel
                    .as_ref()
                    .map(|channel| (channel.target, channel.data.bytes));
                self.running_at = Some(timestamp);
            }
            EvaluateEvent::Completed { .. } => {
                self.finished_at = Some(timestamp);
                self.finished_state = Some("completed");
            }
            EvaluateEvent::Failed { .. } => {
                self.finished_at = Some(timestamp);
                self.finished_state = Some("failed");
            }
        }
    }

    fn try_build(self, id: Uuid) -> AnalyzerResult<Option<EvaluateSpan>> {
        let (
            Some(instance_name),
            Some(actor_id),
            Some(processor_id),
            Some(queued_at),
            Some(start),
            Some(end),
            Some(finished_state),
        ) = (
            self.instance_name,
            self.actor_id,
            self.processor_id,
            self.queued_at,
            self.running_at,
            self.finished_at,
            self.finished_state,
        )
        else {
            return Ok(None);
        };
        let channel_bytes = self
            .channel
            .map(|(_, bytes)| CapacityValue::new("bytes", bytes));
        Ok(Some(EvaluateSpan {
            id,
            instance_name,
            actor_id,
            processor_id,
            channel: self.channel,
            queued_at,
            span: SpanUnixNanoSec::try_new(start, end)?,
            finished_state,
            processor_unit: CapacityValue::new("unit", 1),
            channel_bytes,
        }))
    }
}

struct EvaluateSpan {
    id: Uuid,
    instance_name: String,
    actor_id: Uuid,
    processor_id: Uuid,
    channel: Option<(Uuid, u64)>,
    queued_at: TimeUnixNanoSec,
    span: SpanUnixNanoSec,
    finished_state: &'static str,
    processor_unit: CapacityValue,
    channel_bytes: Option<CapacityValue>,
}

struct EvaluateUsage<'a> {
    evaluate: &'a EvaluateSpan,
    resource_id: Uuid,
    capacity: &'a CapacityValue,
}

impl<'a> Usage<'a> for EvaluateUsage<'a> {
    fn entity_id(&self) -> Uuid {
        self.evaluate.id
    }

    fn resource_id(&self) -> Uuid {
        self.resource_id
    }

    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        std::iter::once(self.capacity)
    }

    fn span(&self) -> SpanUnixNanoSec {
        self.evaluate.span
    }
}

impl EvaluateSpan {
    fn to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> FiniteStateMachine {
        let mut running_usages = vec![FsmUsage {
            resource: self.processor_id,
            capacities: vec![("unit".to_owned(), Some(1))],
        }];
        if let Some((channel_id, bytes)) = self.channel {
            running_usages.push(FsmUsage {
                resource: channel_id,
                capacities: vec![("bytes".to_owned(), Some(bytes))],
            });
        }
        let transition = |name: &str, timestamp, usages| FsmTransition {
            name: name.to_owned(),
            usages,
            timestamp: to_secs_relative(timestamp, epoch),
            attributes: vec![],
            derived_attributes: vec![],
        };
        FiniteStateMachine {
            id: self.id,
            type_name: EVALUATE_ENTITY_TYPE.to_owned(),
            instance_name: self.instance_name.clone(),
            transitions: vec![
                transition("queued", self.queued_at, vec![]),
                transition("running", self.span.start(), running_usages),
                transition(self.finished_state, self.span.end(), vec![]),
            ],
        }
    }
}

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
    actors: Vec<ActorSpan>,
    evaluates: Vec<EvaluateSpan>,
    resources: HashMap<Uuid, DeclaredResource>,
    resource_groups: HashMap<Uuid, DeclaredResourceGroup>,
}

impl CudfPolarsUiAnalyzer {
    fn query_operator_ids(&self, query_id: Uuid) -> AnalyzerResult<HashSet<Uuid>> {
        Ok(self
            .model
            .query_view(query_id)?
            .operators()
            .map(|operator| operator.id())
            .collect())
    }

    fn evaluate_operator_id(&self, evaluate: &EvaluateSpan) -> Option<Uuid> {
        self.actors
            .iter()
            .find(|actor| actor.id == evaluate.actor_id)
            .map(|actor| actor.operator_id)
    }

    fn resource_in_group(
        &self,
        resource: &DeclaredResource,
        group_id: Uuid,
        engine_id: Uuid,
    ) -> bool {
        group_id == engine_id
            || resource.parent_group_id == group_id
            || self
                .resource_groups
                .get(&resource.parent_group_id)
                .is_some_and(|group| group.parent_group_id == group_id)
    }

    fn build_timeline(
        &self,
        query_id: Uuid,
        request: TimelineRequest<OperatorFilter>,
    ) -> AnalyzerResult<(quent_time::bin::BinnedSpanSec, UiResourceTimeline)> {
        let epoch = self.model.query_epoch(query_id)?;
        let config = request.config().try_into_binned_span(epoch)?;
        let query_operator_ids = self.query_operator_ids(query_id)?;
        let engine_id = self.model.query_view(query_id)?.engine()?.id();

        let (
            selected_resource_ids,
            entity_type_name,
            requested_operator_ids,
            resource_type_name,
            long_entities_threshold,
        ): (
            HashSet<Uuid>,
            Option<String>,
            HashSet<Uuid>,
            String,
            Option<TimeNanoSec>,
        ) = match request {
            TimelineRequest::Resource(request) => (
                [request.resource_id].into_iter().collect(),
                request.entity_filter.entity_type_name,
                request.application.operator_ids.into_iter().collect(),
                self.resources
                    .get(&request.resource_id)
                    .map_or(ACTOR_SLOT_RESOURCE_TYPE, |resource| resource.type_name)
                    .to_owned(),
                request.long_entities_threshold_s.map(to_nanosecs),
            ),
            TimelineRequest::ResourceGroup(request) => {
                let resource_ids: HashSet<Uuid> =
                    if request.resource_type_name == ACTOR_SLOT_RESOURCE_TYPE {
                        if request.resource_group_id == engine_id {
                            self.actors
                                .iter()
                                .filter(|actor| query_operator_ids.contains(&actor.operator_id))
                                .map(|actor| actor_slot_id(actor.worker_id))
                                .collect()
                        } else {
                            [actor_slot_id(request.resource_group_id)]
                                .into_iter()
                                .collect()
                        }
                    } else {
                        self.resources
                            .values()
                            .filter(|resource| {
                                resource.type_name == request.resource_type_name
                                    && self.resource_in_group(
                                        resource,
                                        request.resource_group_id,
                                        engine_id,
                                    )
                            })
                            .map(|resource| resource.id)
                            .collect()
                    };
                (
                    resource_ids,
                    request.entity_filter.entity_type_name,
                    request.app_params.operator_ids.into_iter().collect(),
                    request.resource_type_name,
                    request.long_entities_threshold_s.map(to_nanosecs),
                )
            }
        };

        if resource_type_name == ACTOR_SLOT_RESOURCE_TYPE {
            if entity_type_name
                .as_deref()
                .is_some_and(|name| name != "Actor")
            {
                return Err(AnalyzerError::InvalidArgument(format!(
                    "unknown timeline entity type {:?}; expected \"Actor\"",
                    entity_type_name.unwrap()
                )));
            }

            let actors = self.actors.iter().filter(|actor| {
                query_operator_ids.contains(&actor.operator_id)
                    && selected_resource_ids.contains(&actor_slot_id(actor.worker_id))
                    && (requested_operator_ids.is_empty()
                        || requested_operator_ids.contains(&actor.operator_id))
            });
            let resource_type = actor_slot_resource_type();

            let data = if entity_type_name.is_some() {
                let mut builder =
                    ResourceTimelineByKeyBuilder::try_new(&resource_type, config, None)?;
                for actor in actors {
                    builder.try_push("running", &ActorUsage(actor))?;
                }
                let result = builder.build();
                let mut capacities_states_values = HashMap::new();
                for ((state_name, capacity_name), values) in result.data {
                    capacities_states_values
                        .entry(capacity_name.to_owned())
                        .or_insert_with(HashMap::new)
                        .insert(state_name.to_owned(), values);
                }
                UiResourceTimeline::BinnedByState(ResourceTimelineBinnedByState {
                    config: result.config.try_to_secs_relative(epoch)?,
                    capacities_states_values,
                    long_fsms: vec![],
                })
            } else {
                let mut builder = ResourceTimelineBuilder::try_new(&resource_type, config, None)?;
                for actor in actors {
                    builder.try_push(&ActorUsage(actor))?;
                }
                let result = builder.build();
                UiResourceTimeline::Binned(ResourceTimelineBinned {
                    config: result.config.try_to_secs_relative(epoch)?,
                    capacities_values: result
                        .data
                        .into_iter()
                        .map(|(name, values)| (name.to_owned(), values))
                        .collect(),
                    long_fsms: vec![],
                })
            };
            return Ok((config.try_to_secs_relative(epoch)?, data));
        }

        if !matches!(
            resource_type_name.as_str(),
            PROCESSOR_RESOURCE_TYPE | DATA_CHANNEL_RESOURCE_TYPE
        ) {
            return Err(AnalyzerError::InvalidArgument(format!(
                "unknown resource type {resource_type_name:?}"
            )));
        }
        if entity_type_name
            .as_deref()
            .is_some_and(|name| name != EVALUATE_ENTITY_TYPE)
        {
            return Err(AnalyzerError::InvalidArgument(format!(
                "unknown timeline entity type {:?}; expected {EVALUATE_ENTITY_TYPE:?}",
                entity_type_name.unwrap()
            )));
        }

        let evaluates = self.evaluates.iter().filter(|evaluate| {
            self.evaluate_operator_id(evaluate)
                .is_some_and(|operator_id| {
                    query_operator_ids.contains(&operator_id)
                        && (requested_operator_ids.is_empty()
                            || requested_operator_ids.contains(&operator_id))
                })
        });
        let resource_type = evaluate_resource_type(&resource_type_name);
        let data = if entity_type_name.is_some() {
            let mut builder = ResourceTimelineByKeyBuilder::try_new(
                &resource_type,
                config,
                long_entities_threshold,
            )?;
            for evaluate in evaluates {
                if resource_type_name == PROCESSOR_RESOURCE_TYPE
                    && selected_resource_ids.contains(&evaluate.processor_id)
                {
                    builder.try_push(
                        "running",
                        &EvaluateUsage {
                            evaluate,
                            resource_id: evaluate.processor_id,
                            capacity: &evaluate.processor_unit,
                        },
                    )?;
                }
                if resource_type_name == DATA_CHANNEL_RESOURCE_TYPE
                    && let (Some((channel_id, _)), Some(channel_bytes)) =
                        (evaluate.channel, evaluate.channel_bytes.as_ref())
                    && selected_resource_ids.contains(&channel_id)
                {
                    builder.try_push(
                        "running",
                        &EvaluateUsage {
                            evaluate,
                            resource_id: channel_id,
                            capacity: channel_bytes,
                        },
                    )?;
                }
            }
            let result = builder.build();
            let long_fsms = result
                .long_entities
                .iter()
                .filter_map(|id| self.evaluates.iter().find(|evaluate| evaluate.id == *id))
                .map(|evaluate| evaluate.to_ui_fsm(epoch))
                .collect();
            let mut capacities_states_values = HashMap::new();
            for ((state_name, capacity_name), values) in result.data {
                capacities_states_values
                    .entry(capacity_name.to_owned())
                    .or_insert_with(HashMap::new)
                    .insert(state_name.to_owned(), values);
            }
            UiResourceTimeline::BinnedByState(ResourceTimelineBinnedByState {
                config: result.config.try_to_secs_relative(epoch)?,
                capacities_states_values,
                long_fsms,
            })
        } else {
            let mut builder =
                ResourceTimelineBuilder::try_new(&resource_type, config, long_entities_threshold)?;
            for evaluate in evaluates {
                if resource_type_name == PROCESSOR_RESOURCE_TYPE
                    && selected_resource_ids.contains(&evaluate.processor_id)
                {
                    builder.try_push(&EvaluateUsage {
                        evaluate,
                        resource_id: evaluate.processor_id,
                        capacity: &evaluate.processor_unit,
                    })?;
                }
                if resource_type_name == DATA_CHANNEL_RESOURCE_TYPE
                    && let (Some((channel_id, _)), Some(channel_bytes)) =
                        (evaluate.channel, evaluate.channel_bytes.as_ref())
                    && selected_resource_ids.contains(&channel_id)
                {
                    builder.try_push(&EvaluateUsage {
                        evaluate,
                        resource_id: channel_id,
                        capacity: channel_bytes,
                    })?;
                }
            }
            let result = builder.build();
            let long_fsms = result
                .long_entities
                .iter()
                .filter_map(|id| self.evaluates.iter().find(|evaluate| evaluate.id == *id))
                .map(|evaluate| evaluate.to_ui_fsm(epoch))
                .collect();
            UiResourceTimeline::Binned(ResourceTimelineBinned {
                config: result.config.try_to_secs_relative(epoch)?,
                capacities_values: result
                    .data
                    .into_iter()
                    .map(|(name, values)| (name.to_owned(), values))
                    .collect(),
                long_fsms,
            })
        };
        Ok((config.try_to_secs_relative(epoch)?, data))
    }
}

impl UiAnalyzer for CudfPolarsUiAnalyzer {
    type Event = CudfPolarsEvent;
    type EntityRef = UiEntityRef;

    fn try_new(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<Self::Event>>,
    ) -> AnalyzerResult<Self> {
        let mut builder = InMemoryQueryEngineModelBuilder::try_new(engine_id)?;
        let mut actor_builders = HashMap::<Uuid, ActorBuilder>::new();
        let mut evaluate_builders = HashMap::<Uuid, EvaluateBuilder>::new();
        let mut resources = HashMap::new();
        let mut resource_groups = HashMap::new();
        for event in events {
            match &event.data {
                CudfPolarsEvent::Actor(actor_event) => {
                    actor_builders
                        .entry(event.id)
                        .or_default()
                        .push(event.timestamp, actor_event);
                }
                CudfPolarsEvent::Evaluate(evaluate_event) => {
                    evaluate_builders
                        .entry(event.id)
                        .or_default()
                        .push(event.timestamp, evaluate_event);
                }
                CudfPolarsEvent::ThreadPool(ThreadPoolEvent::Declared {
                    instance_name,
                    worker,
                }) => {
                    resource_groups.insert(
                        event.id,
                        DeclaredResourceGroup {
                            id: event.id,
                            instance_name: instance_name.clone(),
                            type_name: "thread_pool",
                            parent_group_id: worker.target,
                        },
                    );
                }
                CudfPolarsEvent::Processor(ProcessorEvent::Declared {
                    instance_name,
                    thread_pool,
                }) => {
                    resources.insert(
                        event.id,
                        DeclaredResource {
                            id: event.id,
                            instance_name: instance_name.clone(),
                            type_name: PROCESSOR_RESOURCE_TYPE,
                            parent_group_id: thread_pool.target,
                        },
                    );
                }
                CudfPolarsEvent::DataChannel(DataChannelEvent::Declared {
                    instance_name,
                    worker,
                    ..
                }) => {
                    resources.insert(
                        event.id,
                        DeclaredResource {
                            id: event.id,
                            instance_name: instance_name.clone(),
                            type_name: DATA_CHANNEL_RESOURCE_TYPE,
                            parent_group_id: worker.target,
                        },
                    );
                }
                _ => {}
            }
            if let Some(event) = to_query_engine_event(event) {
                builder.try_push(event)?;
            }
        }
        let actors: Vec<ActorSpan> = actor_builders
            .into_iter()
            .map(|(id, builder)| builder.try_build(id))
            .collect::<AnalyzerResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        let evaluates = evaluate_builders
            .into_iter()
            .map(|(id, builder)| builder.try_build(id))
            .collect::<AnalyzerResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        let mut model = builder.try_build()?;
        for actor in &actors {
            if let Ok(operator) = model.operator_mut(actor.operator_id) {
                operator.extend_active_span(actor.span);
            }
        }
        Ok(Self {
            model,
            actors,
            evaluates,
            resources,
            resource_groups,
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
                    ..
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
        let query_operator_ids = self.query_operator_ids(query_id)?;
        let worker_ids: HashSet<_> = self
            .actors
            .iter()
            .filter(|actor| query_operator_ids.contains(&actor.operator_id))
            .map(|actor| actor.worker_id)
            .collect();
        let query_resource_group_ids: HashSet<_> = self
            .resource_groups
            .values()
            .filter(|group| worker_ids.contains(&group.parent_group_id))
            .map(|group| group.id)
            .collect();
        let mut resources: HashMap<_, _> = worker_ids
            .iter()
            .map(|&worker_id| {
                let id = actor_slot_id(worker_id);
                (
                    id,
                    Resource {
                        id,
                        instance_name: "actor slot".to_owned(),
                        type_name: ACTOR_SLOT_RESOURCE_TYPE.to_owned(),
                        parent_group_id: worker_id,
                    },
                )
            })
            .collect();
        resources.extend(
            self.resources
                .values()
                .filter(|resource| {
                    worker_ids.contains(&resource.parent_group_id)
                        || query_resource_group_ids.contains(&resource.parent_group_id)
                })
                .map(|resource| {
                    (
                        resource.id,
                        Resource {
                            id: resource.id,
                            instance_name: resource.instance_name.clone(),
                            type_name: resource.type_name.to_owned(),
                            parent_group_id: resource.parent_group_id,
                        },
                    )
                }),
        );

        let resource_types = [
            (
                ACTOR_SLOT_RESOURCE_TYPE.to_owned(),
                UiResourceTypeDecl {
                    name: ACTOR_SLOT_RESOURCE_TYPE.to_owned(),
                    capacities: vec![UiCapacityDecl {
                        name: "unit".to_owned(),
                        kind: CapacityKind::Occupancy,
                        quantity: "unit".to_owned(),
                    }],
                    used_by: vec!["Actor".to_owned()],
                },
            ),
            (
                PROCESSOR_RESOURCE_TYPE.to_owned(),
                UiResourceTypeDecl {
                    name: PROCESSOR_RESOURCE_TYPE.to_owned(),
                    capacities: vec![UiCapacityDecl {
                        name: "unit".to_owned(),
                        kind: CapacityKind::Occupancy,
                        quantity: "unit".to_owned(),
                    }],
                    used_by: vec![EVALUATE_ENTITY_TYPE.to_owned()],
                },
            ),
            (
                DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                UiResourceTypeDecl {
                    name: DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                    capacities: vec![UiCapacityDecl {
                        name: "bytes".to_owned(),
                        kind: CapacityKind::Rate,
                        quantity: "bytes".to_owned(),
                    }],
                    used_by: vec![EVALUATE_ENTITY_TYPE.to_owned()],
                },
            ),
        ]
        .into_iter()
        .collect();
        let group_type = |name: &str, contains_resource_types: Vec<String>| ResourceGroupTypeDecl {
            name: name.to_owned(),
            used_by_entity_types: vec!["Actor".to_owned(), EVALUATE_ENTITY_TYPE.to_owned()],
            contains_resource_types,
        };
        let resource_group_types = [
            (
                "engine".to_owned(),
                group_type(
                    "engine",
                    vec![
                        ACTOR_SLOT_RESOURCE_TYPE.to_owned(),
                        PROCESSOR_RESOURCE_TYPE.to_owned(),
                        DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                    ],
                ),
            ),
            (
                "worker".to_owned(),
                group_type(
                    "worker",
                    vec![
                        ACTOR_SLOT_RESOURCE_TYPE.to_owned(),
                        PROCESSOR_RESOURCE_TYPE.to_owned(),
                        DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                    ],
                ),
            ),
            (
                "thread_pool".to_owned(),
                group_type("thread_pool", vec![PROCESSOR_RESOURCE_TYPE.to_owned()]),
            ),
        ]
        .into_iter()
        .collect();
        let resource_groups = self
            .resource_groups
            .values()
            .filter(|group| query_resource_group_ids.contains(&group.id))
            .map(|group| {
                (
                    group.id,
                    ResourceGroup {
                        id: group.id,
                        type_name: group.type_name.to_owned(),
                        instance_name: group.instance_name.clone(),
                        parent_group_id: Some(group.parent_group_id),
                    },
                )
            })
            .collect();
        let resource_tree = ResourceTree::ResourceGroup(ResourceGroupNode {
            id: UiEntityRef::Engine(view.engine()?.id()),
            children: worker_ids
                .iter()
                .map(|&worker_id| {
                    let mut children = vec![ResourceTree::Resource(UiEntityRef::Resource(
                        actor_slot_id(worker_id),
                    ))];
                    children.extend(
                        self.resources
                            .values()
                            .filter(|resource| resource.parent_group_id == worker_id)
                            .map(|resource| {
                                ResourceTree::Resource(UiEntityRef::Resource(resource.id))
                            }),
                    );
                    children.extend(
                        self.resource_groups
                            .values()
                            .filter(|group| group.parent_group_id == worker_id)
                            .map(|group| {
                                ResourceTree::ResourceGroup(ResourceGroupNode {
                                    id: UiEntityRef::ResourceGroup(group.id),
                                    children: self
                                        .resources
                                        .values()
                                        .filter(|resource| resource.parent_group_id == group.id)
                                        .map(|resource| {
                                            ResourceTree::Resource(UiEntityRef::Resource(
                                                resource.id,
                                            ))
                                        })
                                        .collect(),
                                })
                            }),
                    );
                    ResourceTree::ResourceGroup(ResourceGroupNode {
                        id: UiEntityRef::Worker(worker_id),
                        children,
                    })
                })
                .collect(),
        });
        let fsm_types = [(
            EVALUATE_ENTITY_TYPE.to_owned(),
            FsmTypeDecl {
                name: EVALUATE_ENTITY_TYPE.to_owned(),
                states: vec![
                    FsmStateTypeDecl {
                        name: "queued".to_owned(),
                        usages: vec![],
                    },
                    FsmStateTypeDecl {
                        name: "running".to_owned(),
                        usages: vec!["processor".to_owned(), "channel".to_owned()],
                    },
                    FsmStateTypeDecl {
                        name: "completed".to_owned(),
                        usages: vec![],
                    },
                    FsmStateTypeDecl {
                        name: "failed".to_owned(),
                        usages: vec![],
                    },
                ],
                transitions: vec![
                    FsmTransitionDecl::Entry("queued".to_owned()),
                    FsmTransitionDecl::Transition("queued".to_owned(), "running".to_owned()),
                    FsmTransitionDecl::Transition("running".to_owned(), "completed".to_owned()),
                    FsmTransitionDecl::Transition("running".to_owned(), "failed".to_owned()),
                    FsmTransitionDecl::Exit("completed".to_owned()),
                    FsmTransitionDecl::Exit("failed".to_owned()),
                ],
            },
        )]
        .into_iter()
        .collect();

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
            resource_types,
            resources,
            resource_groups,
            resource_group_types,
            fsm_types,
        };
        let unique_operator_names = view
            .operators()
            .filter_map(|operator| operator.operator_type_name().map(str::to_owned))
            .collect();
        let plan_tree = view.plan_tree(query_id)?.to_ui();
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
        request: SingleTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<SingleTimelineResponse> {
        let (config, data) = self.build_timeline(request.app_params.query_id, request.entry)?;
        Ok(SingleTimelineResponse { config, data })
    }

    fn list_entities(
        &self,
        request: EntityListRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<EntityListResponse> {
        let query_id = request.app_params.query_id;
        let epoch = self.model.query_epoch(query_id)?;
        let query_operator_ids = self.query_operator_ids(query_id)?;
        let entry = request.entry;
        let window = entry.window.try_into_span(epoch)?;
        let requested_operator_ids: HashSet<_> =
            entry.application.operator_ids.into_iter().collect();
        if entry
            .filter
            .entity_type_name
            .as_deref()
            .is_some_and(|name| name != EVALUATE_ENTITY_TYPE)
        {
            return Ok(EntityListResponse {
                items: vec![],
                total: 0,
            });
        }
        let engine_id = self.model.query_view(query_id)?.engine()?.id();
        let scope = entry.filter.scope.as_ref().map(|scope| match scope {
            EntityScope::Resource { resource_id } => {
                [*resource_id].into_iter().collect::<HashSet<_>>()
            }
            EntityScope::ResourceGroup {
                resource_group_id,
                resource_type_name,
            } => self
                .resources
                .values()
                .filter(|resource| {
                    resource.type_name == resource_type_name
                        && self.resource_in_group(resource, *resource_group_id, engine_id)
                })
                .map(|resource| resource.id)
                .collect(),
        });
        let min_usage = entry.filter.min_usage_s.map(to_nanosecs);
        let mut ranked: Vec<_> = self
            .evaluates
            .iter()
            .filter_map(|evaluate| {
                let operator_id = self.evaluate_operator_id(evaluate)?;
                if !query_operator_ids.contains(&operator_id)
                    || (!requested_operator_ids.is_empty()
                        && !requested_operator_ids.contains(&operator_id))
                {
                    return None;
                }
                let lifecycle =
                    SpanUnixNanoSec::try_new(evaluate.queued_at, evaluate.span.end()).ok()?;
                if !lifecycle.intersects(&window) {
                    return None;
                }
                let uses_scope = |resource_id| {
                    scope
                        .as_ref()
                        .is_none_or(|resource_ids| resource_ids.contains(&resource_id))
                };
                let metric = [
                    Some(evaluate.processor_id),
                    evaluate.channel.map(|(id, _)| id),
                ]
                .into_iter()
                .flatten()
                .filter(|&resource_id| uses_scope(resource_id))
                .filter_map(|_| evaluate.span.intersection(&window))
                .map(|span| span.duration())
                .max();
                let metric = match (&scope, metric) {
                    (Some(_), None) => return None,
                    (_, metric) => metric.unwrap_or(0),
                };
                if min_usage.is_some_and(|minimum| metric < minimum) {
                    return None;
                }
                Some((evaluate, metric))
            })
            .collect();
        ranked.sort_by(|(left, left_metric), (right, right_metric)| {
            let order = left_metric.cmp(right_metric);
            let order = match entry.sort.dir {
                SortDir::Asc => order,
                SortDir::Desc => order.reverse(),
            };
            order.then_with(|| left.id.cmp(&right.id))
        });
        let total = ranked.len() as u32;
        let mut ranked = ranked.into_iter();
        let items: Vec<_> = if let Some(page) = entry.page {
            ranked
                .by_ref()
                .skip(page.page.saturating_mul(page.max) as usize)
                .take(page.max as usize)
                .collect()
        } else {
            ranked.collect()
        };
        Ok(EntityListResponse {
            items: items
                .into_iter()
                .map(|(evaluate, usage_duration)| EntityListItem {
                    entity: evaluate.to_ui_fsm(epoch),
                    usage_duration_s: quent_time::to_secs(usage_duration),
                })
                .collect(),
            total,
        })
    }

    fn bulk_resource_timeline(
        &self,
        request: BulkTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<BulkTimelinesResponse> {
        let entries = request
            .entries
            .into_iter()
            .map(|(entry_id, entry)| {
                let response = match self.build_timeline(request.app_params.query_id, entry) {
                    Ok((config, data)) => BulkTimelinesResponseEntry::Ok {
                        message: String::new(),
                        config,
                        data,
                    },
                    Err(error) => BulkTimelinesResponseEntry::Error {
                        message: error.to_string(),
                    },
                };
                (entry_id, response)
            })
            .collect();
        Ok(BulkTimelinesResponse { entries })
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
                ..
            } => query_engine::engine::EngineEvent::Init(query_engine::engine::Init {
                implementation: query_engine::engine::EngineImplementationAttributes {
                    name: Some(implementation.name),
                    version: Some(implementation.version),
                    custom_attributes: implementation.custom_attributes,
                },
                instance_name: Some(instance_name),
            }),
            EngineEvent::Exit { .. } => {
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
            WorkerEvent::Exit { .. } => {
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
                custom_attributes.add("input_bytes", values.input_bytes);
                custom_attributes.add("output_bytes", values.output_bytes);
                if let Some(output_rows) = values.output_rows {
                    custom_attributes.add("output_rows", output_rows);
                }
                custom_attributes.add("chunk_count", values.chunk_count);
                custom_attributes.add("duplicated", values.duplicated);
                if let Some(decision) = values.decision {
                    custom_attributes.add("decision", decision);
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
                    seq,
                } => (
                    seq,
                    query_engine::query::QueryTransition::Init(query_engine::query::Init {
                        instance_name,
                        query_group_id: Ref::new(query_group.target),
                    }),
                ),
                QueryEvent::Planning { seq } => (
                    seq,
                    query_engine::query::QueryTransition::Planning(
                        query_engine::query::Planning {},
                    ),
                ),
                QueryEvent::Executing { seq } => (
                    seq,
                    query_engine::query::QueryTransition::Executing(
                        query_engine::query::Executing {},
                    ),
                ),
                QueryEvent::Completed { seq } => (seq, query_engine::query::QueryTransition::Exit),
                QueryEvent::Failed { seq, .. } => (seq, query_engine::query::QueryTransition::Exit),
            };
            query_engine::QueryEngineEvent::Query(FsmEvent { seq, state })
        }
        CudfPolarsEvent::ThreadPool(_)
        | CudfPolarsEvent::Processor(_)
        | CudfPolarsEvent::DeviceMemory(_)
        | CudfPolarsEvent::Storage(_)
        | CudfPolarsEvent::DataChannel(_)
        | CudfPolarsEvent::Evaluate(_)
        | CudfPolarsEvent::Actor(_) => return None,
    };
    Some(Event::new(id, timestamp, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use generated::{Implementation, OperatorStatistics, PlanEvent};
    use quent_events::{EntityRef, Model};

    #[test]
    fn builds_query_bundle_from_generated_events() {
        let engine_id = Uuid::now_v7();
        let worker_id = Uuid::now_v7();
        let query_group_id = Uuid::now_v7();
        let query_id = Uuid::now_v7();
        let plan_id = Uuid::now_v7();
        let operator_id = Uuid::now_v7();
        let actor_id = Uuid::now_v7();
        let thread_pool_id = Uuid::now_v7();
        let processor_id = Uuid::now_v7();
        let evaluate_id = Uuid::now_v7();
        let events = vec![
            Event::new(
                engine_id,
                1,
                CudfPolarsEvent::Engine(EngineEvent::Init {
                    seq: 0,
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
                    seq: 0,
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
                    seq: 0,
                    instance_name: "iteration-1".to_owned(),
                    query_group: EntityRef::new(query_group_id, ()),
                }),
            ),
            Event::new(
                query_id,
                5,
                CudfPolarsEvent::Query(QueryEvent::Planning { seq: 1 }),
            ),
            Event::new(
                query_id,
                6,
                CudfPolarsEvent::Query(QueryEvent::Executing { seq: 2 }),
            ),
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
            Event::new(
                operator_id,
                8,
                CudfPolarsEvent::Operator(OperatorEvent::Declared {
                    plan: EntityRef::new(plan_id, ()),
                    parent_operators: vec![],
                    instance_name: "scan".to_owned(),
                    type_name: "Scan".to_owned(),
                    attributes: DynamicAttributes::new(),
                }),
            ),
            Event::new(
                thread_pool_id,
                8,
                CudfPolarsEvent::ThreadPool(ThreadPoolEvent::Declared {
                    instance_name: "host threads".to_owned(),
                    worker: EntityRef::new(worker_id, ()),
                }),
            ),
            Event::new(
                processor_id,
                8,
                CudfPolarsEvent::Processor(ProcessorEvent::Declared {
                    instance_name: "thread-0".to_owned(),
                    thread_pool: EntityRef::new(thread_pool_id, ()),
                }),
            ),
            Event::new(
                actor_id,
                9,
                CudfPolarsEvent::Actor(ActorEvent::Started {
                    seq: 0,
                    operator: EntityRef::new(operator_id, ()),
                    worker: EntityRef::new(worker_id, ()),
                }),
            ),
            Event::new(
                actor_id,
                10,
                CudfPolarsEvent::Actor(ActorEvent::Running { seq: 1 }),
            ),
            Event::new(
                evaluate_id,
                11,
                CudfPolarsEvent::Evaluate(EvaluateEvent::Queued {
                    seq: 0,
                    instance_name: "Scan-evaluate".to_owned(),
                    actor: EntityRef::new(actor_id, ()),
                }),
            ),
            Event::new(
                evaluate_id,
                12,
                CudfPolarsEvent::Evaluate(EvaluateEvent::Running {
                    seq: 1,
                    io: false,
                    input_bytes: 10,
                    processor: EntityRef::new(processor_id, generated::ProcessorUsage {}),
                    channel: None,
                }),
            ),
            Event::new(
                evaluate_id,
                18,
                CudfPolarsEvent::Evaluate(EvaluateEvent::Completed {
                    seq: 2,
                    output_bytes: 20,
                }),
            ),
            Event::new(
                actor_id,
                20,
                CudfPolarsEvent::Actor(ActorEvent::Completed {
                    seq: 2,
                    values: OperatorStatistics {
                        input_bytes: 10,
                        output_bytes: 20,
                        output_rows: Some(1),
                        chunk_count: 1,
                        duplicated: false,
                        decision: None,
                    },
                }),
            ),
            Event::new(
                query_id,
                21,
                CudfPolarsEvent::Query(QueryEvent::Completed { seq: 3 }),
            ),
            Event::new(
                worker_id,
                22,
                CudfPolarsEvent::Worker(WorkerEvent::Exit { seq: 1 }),
            ),
            Event::new(
                engine_id,
                23,
                CudfPolarsEvent::Engine(EngineEvent::Exit { seq: 1 }),
            ),
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
        assert!(
            bundle.entities.operators[&operator_id]
                .active_span
                .is_some()
        );
        assert!(
            bundle
                .entities
                .resource_types
                .contains_key(ACTOR_SLOT_RESOURCE_TYPE)
        );
        assert!(
            bundle
                .entities
                .resources
                .contains_key(&actor_slot_id(worker_id))
        );
        assert!(bundle.entities.resources.contains_key(&processor_id));
        assert!(
            bundle
                .entities
                .resource_groups
                .contains_key(&thread_pool_id)
        );
        assert!(
            bundle
                .entities
                .resource_types
                .contains_key(PROCESSOR_RESOURCE_TYPE)
        );
        assert!(bundle.entities.fsm_types.contains_key(EVALUATE_ENTITY_TYPE));

        let response = analyzer
            .bulk_resource_timeline(BulkTimelineRequest {
                entries: [
                    (
                        "engine:base".to_owned(),
                        TimelineRequest::ResourceGroup(
                            quent_ui::timeline::request::ResourceGroupTimelineRequest {
                                resource_group_id: engine_id,
                                resource_type_name: ACTOR_SLOT_RESOURCE_TYPE.to_owned(),
                                long_entities_threshold_s: None,
                                entity_filter: quent_ui::timeline::request::EntityFilter {
                                    entity_type_name: Some("Actor".to_owned()),
                                },
                                app_params: OperatorFilter {
                                    operator_ids: vec![],
                                },
                                config: quent_ui::timeline::request::TimelineConfig {
                                    num_bins: 4,
                                    start: 0.0,
                                    end: 20e-9,
                                },
                            },
                        ),
                    ),
                    (
                        "processor:evaluate".to_owned(),
                        TimelineRequest::Resource(
                            quent_ui::timeline::request::ResourceTimelineRequest {
                                resource_id: processor_id,
                                long_entities_threshold_s: Some(1e-9),
                                entity_filter: quent_ui::timeline::request::EntityFilter {
                                    entity_type_name: Some(EVALUATE_ENTITY_TYPE.to_owned()),
                                },
                                application: OperatorFilter {
                                    operator_ids: vec![],
                                },
                                config: quent_ui::timeline::request::TimelineConfig {
                                    num_bins: 4,
                                    start: 0.0,
                                    end: 20e-9,
                                },
                            },
                        ),
                    ),
                ]
                .into_iter()
                .collect(),
                app_params: QueryFilter { query_id },
            })
            .unwrap();
        let BulkTimelinesResponseEntry::Ok { data, .. } = &response.entries["engine:base"] else {
            panic!("timeline entry failed")
        };
        let UiResourceTimeline::BinnedByState(timeline) = data else {
            panic!("expected a state-keyed timeline")
        };
        assert_eq!(
            timeline.capacities_states_values["unit"]["running"].len(),
            4
        );
        assert!(
            timeline.capacities_states_values["unit"]["running"]
                .iter()
                .any(|&v| v > 0.0)
        );
        let BulkTimelinesResponseEntry::Ok { data, .. } = &response.entries["processor:evaluate"]
        else {
            panic!("evaluate timeline entry failed")
        };
        let UiResourceTimeline::BinnedByState(timeline) = data else {
            panic!("expected a state-keyed evaluate timeline")
        };
        assert_eq!(
            timeline.capacities_states_values["unit"]["running"].len(),
            4
        );
        assert!(
            timeline.capacities_states_values["unit"]["running"]
                .iter()
                .any(|&v| v > 0.0)
        );
        assert_eq!(timeline.long_fsms.len(), 1);
        assert_eq!(timeline.long_fsms[0].id, evaluate_id);
        assert_eq!(timeline.long_fsms[0].instance_name, "Scan-evaluate");

        let entities = analyzer
            .list_entities(EntityListRequest {
                entry: quent_ui::entities::request::EntityListEntry {
                    window: quent_ui::entities::request::TimeWindow {
                        start: 0.0,
                        end: 20e-9,
                    },
                    filter: quent_ui::entities::request::EntityListFilter {
                        scope: Some(EntityScope::Resource {
                            resource_id: processor_id,
                        }),
                        entity_type_name: Some(EVALUATE_ENTITY_TYPE.to_owned()),
                        min_usage_s: Some(1e-9),
                    },
                    sort: quent_ui::entities::request::Sort {
                        key: quent_ui::entities::request::EntitySortKey::UsageDuration,
                        dir: SortDir::Desc,
                    },
                    page: None,
                    application: OperatorFilter {
                        operator_ids: vec![],
                    },
                },
                app_params: QueryFilter { query_id },
            })
            .unwrap();
        assert_eq!(entities.total, 1);
        assert_eq!(entities.items[0].entity.id, evaluate_id);
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
                seq: 0,
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
                seq: 0,
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
                seq: 0,
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
                seq: 0,
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
                seq: 0,
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
