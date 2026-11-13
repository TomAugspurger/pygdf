# cudf-polars Quent browser analyzer

This crate converts completed cudf-polars Quent traces to a versioned,
schema-specific browser bundle and exposes the analyzer through WASM.

Generate a payload from either the collector output directory or its ZIP:

```bash
cargo run --features generate --bin cudf-polars-quent-bundle -- \
  logs/<run-id>.zip logs/<run-id>.quent-browser
```

Build the analyzer:

```bash
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen \
  target/wasm32-unknown-unknown/release/cudf_polars_quent_browser.wasm \
  --target web \
  --out-dir generated
```

Benchmark runs using `--collect-traces` generate both files automatically. The
ZIP remains the archival `quent-open` input. The `.quent-browser` file is the
relocatable static-viewer payload and contains exactly one engine with all of
its controller and worker contexts.

The generated JavaScript module exports Quent's standardized `BrowserServer`
interface, so no cudf-polars-specific Web Worker is required. Host the generated
JavaScript and WASM files on the same origin as the Quent UI, then select them
with the static viewer URL:

```text
https://quent.example/simulator/?module=/analyzers/cudf-polars/cudf_polars_quent_browser.js&data=https://traces.example/run.quent-browser#/
```
