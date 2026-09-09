# cudf-polars Quent analyzer

This crate is the `quent-open` viewer entry for telemetry generated from
`../model.yaml`. Its build script generates typed event importers from the same
schema used by the Python bridge. The analyzer adapts cudf-polars' query-engine
entities to Quent's standard query-engine UI.

`quent-open` currently indexes contexts with its legacy query-engine `Engine`
and `Worker` wire types before loading this viewer, and reads them from
snake-case stream directories rather than the schema-generated entity names. The
corresponding schema events therefore stay wire-compatible, and
`cudf_polars.quent._export` mirrors those two streams under `engine/` and
`worker/`, rewriting attribute-less events to the mapping form those types
expect. Each part is load-bearing: without them the viewer still builds but
lists no engines. Analyzer tests drive the real indexer to guard this.

The crate is normally built on demand by `quent-open` using the package name
and git provenance embedded in each archive's `model.qmi`. To check it locally:

```sh
cargo check
```

The pinned Quent analyzer stack currently enables Quent's collector support,
which requires a Protocol Buffers compiler even though this analyzer only reads
local NDJSON files.
