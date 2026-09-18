# cudf-polars Quent bindings

This optional Maturin project generates the `cudf_polars._quent` extension from
`../model.yaml`. All Quent crates are pinned to the same Git revision so the
schema types used by YAML parsing, Rust instrumentation, and Python code
generation have one Cargo package identity. The extension embeds the model and
Quent build provenance used to write `model.qmi`; the
`cudf-polars-quent-analyzer` package in `../analyzer` is the corresponding
`quent-open` viewer entry.

For local development, install into the active cudf development environment:

```sh
python -m maturin develop
```

Build a distributable wheel with:

```sh
python -m maturin build --release
```

The main cudf-polars package imports this extension only when telemetry is
enabled, so environments that do not use Quent do not need Rust or this wheel.

## Updating Quent

Quent is pinned by full Git commit SHA in both
`bridge/Cargo.toml` and `../analyzer/Cargo.toml`. To update Quent:

1. Replace every Quent dependency's `rev` in both manifests with the same full
   commit SHA. Do not use a branch, tag, abbreviated SHA, or different revision
   spelling: Cargo must resolve one package identity for the generated model,
   analyzer, and `quent-open` viewer traits.
2. Resolve each crate once without `--locked` so Cargo replaces the old Quent
   Git source and records the new source SHA and transitive dependency graph:

   ```sh
   # From python/cudf_polars/quent/bridge
   cargo update
   (cd ../analyzer && cargo update)
   ```

   Do not edit either lockfile by hand. Review both diffs for unexpected
   non-Quent upgrades, then commit `bridge/Cargo.lock` and
   `../analyzer/Cargo.lock`. Subsequent reproducibility checks may use
   `cargo check --locked`.
3. Activate the cudf development environment and rebuild from
   `python/cudf_polars/quent/bridge` (the directory containing this README):

   ```sh
   python -m maturin develop
   python -c "import cudf_polars._quent as q; print(q.ExporterOptions.collector)"
   ```

   The build script generates the Rust bridge and a PEP 561 stub under Cargo's
   `OUT_DIR`; it does not overwrite the tracked
   `../../cudf_polars/_quent.pyi`. When the generated API changes, refresh that
   tracked stub from the most recent debug build:

   ```sh
   generated_stub="$(
     ls -t target/debug/build/cudf-polars-quent-*/out/_quent/__init__.pyi |
       head -n 1
   )"
   cp "${generated_stub}" ../../cudf_polars/_quent.pyi
   ```

   The generated output already includes the cudf-specific transformations
   made in `build.rs`. Review and commit the stub diff, then run mypy so stale
   handle names, event methods, and argument types are caught.
4. From `../analyzer`, run `cargo check` and `cargo test`. A Protocol Buffers
   compiler may be required by Quent's transitive collector dependencies.
5. Run the focused tests under `python/cudf_polars/tests/quent`. Update Python
   instrumentation call sites or the analyzer adapter when a Quent schema,
   generated API, storage format, or viewer interface changed.
6. Commit and push the analyzer changes to the Git remote recorded in the
   bridge's build provenance. Rebuild the bridge after committing, then
   regenerate traces. `quent-open` checks out the analyzer package at the exact
   remote and commit embedded in `model.qmi`; a local-only commit or a trace
   generated from the previous commit cannot be loaded elsewhere.

New trace archives embed the updated Quent Git provenance in `model.qmi`.
Existing archives retain the revision with which they were generated.
