// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAGIC: &[u8; 8] = b"QNTBR001";
const HEADER_LEN: usize = 28;
pub const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Capabilities {
    pub data_flow: bool,
    pub nvtx: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Metadata {
    pub format_version: u32,
    pub model_name: String,
    pub schema_digest: String,
    pub model_provenance: Option<serde_json::Value>,
    pub analyzer_revision: String,
    pub quent_revision: String,
    pub codec: String,
    pub engine_id: Uuid,
    pub context_ids: Vec<Uuid>,
    pub capabilities: Capabilities,
}

pub struct Bundle<'a> {
    pub metadata: Metadata,
    pub payload: &'a [u8],
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("browser bundle is truncated")]
    Truncated,
    #[error("browser bundle magic is invalid")]
    InvalidMagic,
    #[error("unsupported browser bundle version {0}")]
    UnsupportedVersion(u32),
    #[error("browser bundle length is invalid")]
    InvalidLength,
    #[error("browser bundle checksum does not match")]
    Checksum,
    #[error("browser bundle metadata is invalid: {0}")]
    Metadata(#[from] serde_json::Error),
}

pub fn encode(metadata: &Metadata, payload: &[u8]) -> Result<Vec<u8>, Error> {
    if metadata.format_version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion(metadata.format_version));
    }
    let metadata = serde_json::to_vec(metadata)?;
    let metadata_len = u32::try_from(metadata.len()).map_err(|_| Error::InvalidLength)?;
    let payload_len = u64::try_from(payload.len()).map_err(|_| Error::InvalidLength)?;
    let mut bytes = Vec::with_capacity(HEADER_LEN + metadata.len() + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&metadata_len.to_be_bytes());
    bytes.extend_from_slice(&payload_len.to_be_bytes());
    bytes.extend_from_slice(&checksum(payload).to_be_bytes());
    bytes.extend_from_slice(&metadata);
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

pub fn decode(bytes: &[u8]) -> Result<Bundle<'_>, Error> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::Truncated);
    }
    if &bytes[..8] != MAGIC {
        return Err(Error::InvalidMagic);
    }
    let metadata_len = u32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let payload_len = usize::try_from(u64::from_be_bytes(bytes[12..20].try_into().unwrap()))
        .map_err(|_| Error::InvalidLength)?;
    let expected_checksum = u64::from_be_bytes(bytes[20..28].try_into().unwrap());
    let metadata_end = HEADER_LEN
        .checked_add(metadata_len)
        .ok_or(Error::InvalidLength)?;
    let payload_end = metadata_end
        .checked_add(payload_len)
        .ok_or(Error::InvalidLength)?;
    if payload_end != bytes.len() {
        return Err(Error::InvalidLength);
    }
    let metadata: Metadata = serde_json::from_slice(&bytes[HEADER_LEN..metadata_end])?;
    if metadata.format_version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion(metadata.format_version));
    }
    let payload = &bytes[metadata_end..];
    if checksum(payload) != expected_checksum {
        return Err(Error::Checksum);
    }
    Ok(Bundle { metadata, payload })
}

pub fn schema_digest() -> String {
    format!("{:016x}", checksum(include_bytes!("../../model.yaml")))
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> Metadata {
        Metadata {
            format_version: FORMAT_VERSION,
            model_name: "cudf-polars".into(),
            schema_digest: schema_digest(),
            model_provenance: None,
            analyzer_revision: "test".into(),
            quent_revision: "test".into(),
            codec: "postcard".into(),
            engine_id: Uuid::nil(),
            context_ids: vec![Uuid::nil()],
            capabilities: Capabilities::default(),
        }
    }

    #[test]
    fn round_trip_and_corruption() {
        let bytes = encode(&metadata(), b"events").unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.metadata, metadata());
        assert_eq!(decoded.payload, b"events");

        let mut corrupt = bytes;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(matches!(decode(&corrupt), Err(Error::Checksum)));
    }

    #[test]
    fn rejects_truncated_and_wrong_version_bundles() {
        assert!(matches!(decode(b"QNT"), Err(Error::Truncated)));

        let mut wrong_version = metadata();
        wrong_version.format_version = FORMAT_VERSION + 1;
        assert!(matches!(
            encode(&wrong_version, b"events"),
            Err(Error::UnsupportedVersion(2))
        ));
    }
}
