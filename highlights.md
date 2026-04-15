# DuckDB local combined build: iteration notes

## Goal
Fix the notebook code cell for the one-time local combined parquet build so it runs without errors and optimizes load speed for repeat analysis.

## Iteration 1 (baseline failure)
- Tried reading remote parquet URLs directly in DuckDB with `read_parquet([...])`.
- Result: failed because DuckDB attempted to auto-install/load `httpfs` and extension download was not available in this environment.
- Conclusion: relying on remote URL reads inside DuckDB is fragile in restricted environments.

## Iteration 2 (robust architecture)
- Switched flow to:
  1. Materialize each monthly parquet file locally once using `fsspec` + streamed copy.
  2. Run DuckDB combine from local parquet paths only.
- Why: removes `httpfs` dependency and network overhead from iterative analysis after first local cache pass.

## Iteration 3 (speed-focused DuckDB settings)
- Used `union_by_name=false` in the combine read because earlier schema checks ensure month schemas match.
- Used `PRAGMA threads=<cpu_count>` to maximize parallel local parquet scan.
- Used `SET preserve_insertion_order = false` for lower overhead in write path.
- Used `COMPRESSION ZSTD` and `ROW_GROUP_SIZE 500000` for a balanced local analytical parquet output.

## Verification performed
- Ran an end-to-end smoke test using local `file://` parquet URLs:
  - Local monthly cache step succeeded.
  - DuckDB combine step succeeded.
  - Output parquet created and row count validated.
- Smoke test output (local synthetic test):
  - `Monthly cache ready in 0.0015s`
  - `Built once in 0.0030s`
  - `combined_rows 2`
- Note: these timings are from a tiny synthetic dataset used to verify correctness only; real NYC monthly datasets will take materially longer.

## Final result
- No errors in the updated code path.
- Local-first architecture minimizes repeated load times after one-time monthly materialization.
