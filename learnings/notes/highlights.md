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

## Reviewer follow-up fixes (robustness)
- Issue: interrupted monthly downloads could leave partial parquet files that were later reused.
  - Solution: download each month to `*.tmp` and atomically promote with `os.replace(...)`; clean up temp files on failure.
- Issue: DuckDB connection cleanup only happened on success.
  - Solution: wrap combine execution in `with duckdb.connect() as con:` so the connection always closes, including on exceptions.

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

## Statistical significance of correlation coefficients

A new code cell (after the correlation matrix heatmap) calculates p-values for all pairwise Pearson correlations among numeric columns (`passenger_count`, `trip_distance`, `PULocationID`, `DOLocationID`, `total_amount`) using the t-distribution:

- **Test**: `t = r × √((n-2)/(1-r²))`,  `p = 2 × (1 − CDF(|t|))` with `df = n − 2`
- **n** ≈ 35M rows → extremely high power

**Expected outcome :** With ~35 million observations, even minuscule correlation coefficients (|r| > ~0.0004) will yield p-values well below 0.05. Therefore **every non-zero correlation in the matrix is statistically significant at α = 0.05**, even though many are practically negligible. The only exception would be the diagonal (r = 1.0, self-correlation, p = 0.0).

Correlation coefficients (r):
                 passenger_count  trip_distance  PULocationID  DOLocationID  \
passenger_count           1.0000         0.0019       -0.0118       -0.0072   
trip_distance             0.0019         1.0000       -0.0112       -0.0078   
PULocationID             -0.0118        -0.0112        1.0000        0.0837   
DOLocationID             -0.0072        -0.0078        0.0837        1.0000   
total_amount              0.0090         0.0092       -0.0185       -0.0119   

                 total_amount  
passenger_count        0.0090  
trip_distance          0.0092  
PULocationID          -0.0185  
DOLocationID          -0.0119  
total_amount           1.0000  

P-values:
                passenger_count trip_distance PULocationID DOLocationID  \
passenger_count        0.00e+00      0.00e+00     0.00e+00     0.00e+00   
trip_distance          0.00e+00      0.00e+00     0.00e+00     0.00e+00   
PULocationID           0.00e+00      0.00e+00     0.00e+00     0.00e+00   
DOLocationID           0.00e+00      0.00e+00     0.00e+00     0.00e+00   
total_amount           0.00e+00      0.00e+00     0.00e+00     0.00e+00   

                total_amount  
passenger_count     0.00e+00  
trip_distance       0.00e+00  
PULocationID        0.00e+00  
DOLocationID        0.00e+00  
total_amount        0.00e+00  

Statistically significant at α = 0.05:
                 passenger_count  trip_distance  PULocationID  DOLocationID  \
passenger_count             True           True          True          True   
trip_distance               True           True          True          True   
PULocationID                True           True          True          True   
DOLocationID                True           True          True          True   
total_amount                True           True          True          True   

                 total_amount  
passenger_count          True  
trip_distance            True  
PULocationID             True  
DOLocationID             True  
total_amount             True
