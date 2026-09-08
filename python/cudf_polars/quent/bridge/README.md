# cudf-polars Quent bindings

This optional Maturin project generates the `cudf_polars._quent` extension from
`../model.yaml`. All Quent crates are pinned to the same Git revision so the
schema types used by YAML parsing, Rust instrumentation, and Python code
generation have one Cargo package identity.

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
