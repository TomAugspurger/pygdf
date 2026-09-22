// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{
    AnalyzerResult,
    resource::{CapacityValue, Usage},
};
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec};
use uuid::Uuid;

use crate::{generated::ActorEvent, resource::actor_slot_id};

#[derive(Default)]
pub(crate) struct ActorBuilder {
    operator_id: Option<Uuid>,
    worker_id: Option<Uuid>,
    running_at: Option<TimeUnixNanoSec>,
    finished_at: Option<TimeUnixNanoSec>,
}

impl ActorBuilder {
    pub(crate) fn push(&mut self, timestamp: TimeUnixNanoSec, event: &ActorEvent) {
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

    pub(crate) fn try_build(self, id: Uuid) -> AnalyzerResult<Option<ActorSpan>> {
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

pub(crate) struct ActorSpan {
    pub(crate) id: Uuid,
    pub(crate) operator_id: Uuid,
    pub(crate) worker_id: Uuid,
    pub(crate) span: SpanUnixNanoSec,
    unit: CapacityValue,
}

pub(crate) struct ActorUsage<'a>(pub(crate) &'a ActorSpan);

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
