#!/usr/bin/env bash
# Credential handling
# IMDSv2 token (TTL in seconds, max 21600)
TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

ROLE_NAME=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/iam/security-credentials/")

CREDS=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/iam/security-credentials/${ROLE_NAME}")

export AWS_ACCESS_KEY_ID=$(echo "$CREDS" | jq -r .AccessKeyId)
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS" | jq -r .SecretAccessKey)
export AWS_SESSION_TOKEN=$(echo "$CREDS" | jq -r .Token)
export AWS_DEFAULT_REGION="us-east-2"
export CUDF_POLARS_LOG_TRACES=1
export CUDF_POLARS_LOG_MEMORY=False
export CUDF_POLARS__PARQUET_OPTIONS__CUCASCADE_POOL_CAPACITY=68719476736
export MAX_IO_THREADS=4
export PINNED_MEMORY=true
export PINNED_INITIAL_POOL_SIZE=68719476736
export CUDF_POLARS__PARQUET_OPTIONS__USE_HYBRID_SCAN=true
export CUDF_POLARS__PARQUET_OPTIONS__PREFETCH_FILE_METADATA=1
export CUDF_POLARS__PARQUET_OPTIONS__PREFETCH_BACKEND=cucascade
export CUDF_POLARS__PARQUET_OPTIONS__CUCASCADE_MAX_CONNECTIONS=256
export CUDF_POLARS__PARQUET_OPTIONS__CUCASCADE_N_REACTORS=4
export CUDF_POLARS__PARQUET_OPTIONS__CUCASCADE_MAX_N_CHUNKS=3
export KVIKIO_NTHREADS=256
export KVIKIO_TASK_SIZE=67108864

# ---------------------------------------------------------------------------
# Byte-range prefetch experiment
# ---------------------------------------------------------------------------
# Three scenarios (uncomment the matching block, leave the others commented):
#
# STEP 0 — Record byte ranges (run once to generate the cache file).
#   Set PREFETCHING_BYTE_RANGES to a path that does NOT yet exist.
#   The engine will compute and persist the ranges, then run normally.
#
#   export CUDF_POLARS__PARQUET_OPTIONS__PREFETCHING_BYTE_RANGES="/tmp/byte_ranges_q1.json"
#   export CUDF_POLARS__PARQUET_OPTIONS__WAIT_FOR_PREFETCH=false
#
# SCENARIO a — Read from pinned host memory (perfect foresight + wait).
#   All fadvise calls fired at query start; scan loop blocks until every
#   read_all_ranges_async completes before the first scan task runs.
#   Measures pure H→D transfer + decode; zero S3 latency on the critical path.
#
#   export CUDF_POLARS__PARQUET_OPTIONS__PREFETCHING_BYTE_RANGES="/tmp/byte_ranges_q1.json"
#   export CUDF_POLARS__PARQUET_OPTIONS__WAIT_FOR_PREFETCH=true
#
# SCENARIO b — Perfect foresight, no wait.
#   All fadvise calls fired at query start; scans run concurrently with
#   in-flight IO.  Ideal overlap with perfect foreknowledge of all ranges.
#
#   export CUDF_POLARS__PARQUET_OPTIONS__PREFETCHING_BYTE_RANGES="/tmp/byte_ranges_q1.json"
#   export CUDF_POLARS__PARQUET_OPTIONS__WAIT_FOR_PREFETCH=false
#
# SCENARIO c — Realistic foresight, no wait (current baseline, default).
#   Prefetch fires only after metadata is read; scans overlap with IO.
#   Leave PREFETCHING_BYTE_RANGES unset (empty string is NOT safe — leave it unset).
# ---------------------------------------------------------------------------

python -m cudf_polars.streaming.benchmarks.pdsh \
  1 --iterations 2 --path "s3://rapids-tpch/tpch-rs/scale-300" \
  --output "pdsh_results_scale-300.jsonl" \
  --suffix "/" \
  --frontend spmd \
  --extra-info '{"sku_name": "g7e.8xlarge", "node_count": 1, "storage_configuration_name": "s3-us-east-2-tpch-rs-scale-300", "benchmark_definition_name": "tpch-rs-300", "cache_state": "warm", "identifier_hash": "598cce17341ac111e578bfd33006d1fc242f6a1353d9749613a007a77f56dd2c"}' \
  --no-print-results \
  --collect-traces \
  --rapidsmpf-statistics \
  --explain \
  --pinned-memory \
  --pinned-initial-pool-size 68719476736 \
  --max-io-threads 4 \
  --validate-directory "s3://rapids-tpch/tpch-rs/scale-300/expected/"
