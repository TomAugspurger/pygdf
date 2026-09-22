// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{CapacityDecl, ResourceTypeDecl};
use uuid::Uuid;

pub(crate) const ACTOR_SLOT_RESOURCE_TYPE: &str = "actor_slot";
pub(crate) const PROCESSOR_RESOURCE_TYPE: &str = "processor";
pub(crate) const DATA_CHANNEL_RESOURCE_TYPE: &str = "data_channel";
pub(crate) const EVALUATE_ENTITY_TYPE: &str = "Evaluate";

#[derive(Clone)]
pub(crate) struct DeclaredResource {
    pub(crate) id: Uuid,
    pub(crate) instance_name: String,
    pub(crate) type_name: &'static str,
    pub(crate) parent_group_id: Uuid,
}

#[derive(Clone)]
pub(crate) struct DeclaredResourceGroup {
    pub(crate) id: Uuid,
    pub(crate) instance_name: String,
    pub(crate) type_name: &'static str,
    pub(crate) parent_group_id: Uuid,
}

pub(crate) fn actor_slot_id(worker_id: Uuid) -> Uuid {
    const ACTOR_SLOT_MASK: u128 = 0xa3c7_0f5d_e912_4bb8_9f61_4a6d_28c0_7e35;
    Uuid::from_u128(worker_id.as_u128() ^ ACTOR_SLOT_MASK)
}

pub(crate) fn actor_slot_resource_type() -> ResourceTypeDecl {
    ResourceTypeDecl::unit(ACTOR_SLOT_RESOURCE_TYPE)
}

pub(crate) fn evaluate_resource_type(name: &str) -> ResourceTypeDecl {
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
