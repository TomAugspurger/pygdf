# cudf-polars Quent analyzer

This crate is the entrypoint for `quent-open`. [./build.rs](./build.rs)
generates typed event importers from the same [Model Schema](../model.yaml) that
cudf-polars uses for generating and exporting events.

`cudf-polars-quent-analyzer` is typically built on demand by `quent-open` using
the versions embedded in the archive's `model.qmi`. Check the local contents with:

```sh
cargo check
cargo test
```

Quent upgrades must update every `rev` in this crate and in
[`../bridge/Cargo.toml`](../bridge/Cargo.toml) to the same full commit SHA, then
refresh and commit both Cargo lockfiles. See the bridge's [Updating
Quent](../bridge/README.md#updating-quent) section for the complete update and
verification checklist.
