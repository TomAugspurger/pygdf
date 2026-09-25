// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::AnalyzerResult;
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec};
use uuid::Uuid;

use crate::generated::ActorEvent;

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
        }))
    }
}

pub(crate) struct ActorSpan {
    pub(crate) id: Uuid,
    pub(crate) operator_id: Uuid,
    pub(crate) worker_id: Uuid,
    pub(crate) span: SpanUnixNanoSec,
}
