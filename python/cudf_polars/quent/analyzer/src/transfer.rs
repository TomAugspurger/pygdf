// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{CapacityValue, Usage};
use quent_time::span::SpanUnixNanoSec;
use uuid::Uuid;

const UNIX_2000_NS: u64 = 946_684_800_000_000_000;

/// A completed inter-rank receive represented as a one-nanosecond rate usage.
///
/// RapidsMPF records receive completion, not transfer start, so the analyzer
/// treats the bytes as an impulse ending at the completion timestamp. The
/// resource timeline's span weighting then reports the correct bytes per bin.
pub(crate) struct TransferUsage {
    pub(crate) channel_id: Uuid,
    pub(crate) span: SpanUnixNanoSec,
    pub(crate) bytes: CapacityValue,
}

impl TransferUsage {
    pub(crate) fn new(
        channel_id: Uuid,
        event_timestamp_ns: u64,
        completion_timestamp_ns: u64,
        metadata_bytes: u64,
        payload_bytes: u64,
    ) -> Self {
        // Older emitters stored the steady-clock value directly. Such values
        // are not on Quent's Unix timeline, so retain compatibility by using
        // the drain event timestamp when the clocks are clearly unrelated.
        let completion_timestamp_ns =
            if event_timestamp_ns >= UNIX_2000_NS && completion_timestamp_ns < UNIX_2000_NS {
                event_timestamp_ns
            } else {
                completion_timestamp_ns.min(event_timestamp_ns)
            };
        let (start, end) = if completion_timestamp_ns == 0 {
            (0, 1)
        } else {
            (completion_timestamp_ns - 1, completion_timestamp_ns)
        };
        Self {
            channel_id,
            span: SpanUnixNanoSec::try_new(start, end)
                .expect("one-nanosecond transfer span is valid"),
            bytes: CapacityValue::new("bytes", metadata_bytes.saturating_add(payload_bytes)),
        }
    }
}

impl<'a> Usage<'a> for &'a TransferUsage {
    fn entity_id(&self) -> Uuid {
        self.channel_id
    }

    fn resource_id(&self) -> Uuid {
        self.channel_id
    }

    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        std::iter::once(&self.bytes)
    }

    fn span(&self) -> SpanUnixNanoSec {
        self.span
    }
}
