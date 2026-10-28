# Run a fresh KuaiRand audit

From the checkout with the package installed:

```sh
python -m experiments.eda_kuairand --events data/raw/log_standard_fixture.csv --out runs/audit
```

Use `--table users=path.csv --table basic=path.csv --table statistics=path.csv`
to profile descriptive fields outside model inputs. Each source is hashed; the
audit records its configuration, source code and whether the row cap truncated
each table. Outputs are `audit.json` and `cohorts.png` in a new directory.

The default limit is the first 100,000 rows per table in sorted file order.
It is explicitly a prefix audit, not a representative sample or full census.
Memory is proportional to that cap plus profiling intermediates; `--max-rows 0`
requests a full in-memory audit and should only be used with an appropriate
resource budget. Hashing reads source files sequentially even when rows are capped.
No DuckDB database, earlier census, generated report or private fixture is needed.

Counts distinguish nulls, sentinel values, invalid known values and valid zeroes.
Complete-record deduplication precedes temporal/cohort analysis. The output keeps
raw and deduplicated traffic counts separate, discloses invalid identity/time
rows, preserves repeated pairs/timestamp ties, and reports validation coverage
both before and after restricting to users observed in training. Default cutoff
times use the existing UTC+8 convention. Descriptive fields do not become model
features, and no audit operation fits or selects a model.
