# StatsCollector Usage

| Location | Stage |
|---|---|
| `statistics.py:40` — `stats = StatsCollector()` | pre-lowering (construction) |
| `statistics.py:50` — `stats.scan_stats[node] = source` (parquet) | pre-lowering (population) |
| `statistics.py:54` — `stats.scan_stats[node] = _build_source_info(...)` (DataFrameScan) | pre-lowering (population) |
| `parallel.py:103` — `state["stats"] = stats` (into `State` for lowering) | lowering (plumbing) |
| `parallel.py:284` — `stats = collect_statistics(...)` then passed to `lower_ir_graph` | pre-lowering (caller) |
| `io.py:98` — `stats.scan_stats.get(ir)` in `scan_partition_plan` | lowering (via `rec.state["stats"]` at call site) |
| `io.py:301` — `rec.state["stats"]` passed to `scan_partition_plan` | lowering |
| `rapidsmpf/io.py:336` — `rec.state["stats"]` passed to `scan_partition_plan` | lowering |
| `rapidsmpf/io.py:647` — `stats.scan_stats.get(ir)` in `make_rapidsmpf_read_parquet_node` | post-lowering (network generation) |
| `rapidsmpf/io.py:724` — `rec.state["stats"]` passed to `make_rapidsmpf_read_parquet_node` | post-lowering (network generation) |
| `rapidsmpf/core.py:107` — `stats = collect_statistics(...)` then passed to `lower_ir_graph` | pre-lowering (caller) |
| `rapidsmpf/core.py:532` — `state["stats"] = stats` (into `GenState` for network gen) | post-lowering (plumbing) |
| `explain.py:84` — `stats = collect_statistics(...)` | pre-lowering (caller) |
| `explain.py:91` — `_repr_ir_tree(ir, stats=stats)` (logical plan display) | pre-lowering (display) |
| `explain.py:189` — `stats.scan_stats.get(ir)` in `_repr_ir_tree` | pre-lowering (display) |
| `explain.py:451` — passed to `lower_ir_graph` in `SerializablePlan.from_ir` | pre-lowering (caller) |

## Summary

`StatsCollector` is accessed in three stages:

- **Pre-lowering** — created/populated in `collect_statistics`, then used in `explain.py` for logical plan display and passed into `lower_ir_graph`
- **Lowering** — threaded through `dispatch.State["stats"]` and accessed via `rec.state["stats"]` in `io.py` and `rapidsmpf/io.py` to compute `scan_partition_plan`
- **Post-lowering** — threaded through `rapidsmpf/dispatch.GenState["stats"]` and accessed via `rec.state["stats"]` during rapidsmpf network generation (`rapidsmpf/io.py:724` → `make_rapidsmpf_read_parquet_node` at line 647)
