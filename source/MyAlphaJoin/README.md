# AlphaJoin

AlphaJoin: Join Order Selection à la AlphaGo.

Full pipeline: data generation → model training → obtaining plans (leading) → testing queries with hints.

---

## Data and plans in this repository

- **`data/`** — data used for training Alpha Join (runtime samples, preprocessed training/test data).
- **`original PG master plans`** — plans from Postgres master (baseline).
- **`alpha_join_plans`** — plans produced by Alpha Join.
- **`best_plans.csv`** — measurements from Alpha Join testing (query, plan, runtime).
- **`AlphaJoin1.0/t6.sql`** — data Alpha Join was actually trained on (runtime samples in `queryName,hintCore,runtime,...` format).
- **`AlphaJoin1.0/t6_new.sql`** — training data after re-running with Leading (via **`rerun_t6_with_psql.py`**); only part of the run was completed.
- **`rerun_t6_with_psql.py`** — script used to regenerate `t6_new.sql` using Leading hints.

---

## Database and connection

The Java utilities (`tools/JDBCUtil.java`) read the connection from the standard
libpq environment variables — nothing is hardcoded. Set whichever differ from the
defaults before running any database step:

| Variable | Default | Meaning |
|---|---|---|
| `PGHOST` | `localhost` | database host |
| `PGPORT` | `5432` | database port |
| `PGDATABASE` | `postgres` | IMDB database name |
| `PGUSER` | `postgres` | database user |
| `PGPASSWORD` | *(empty)* | database password |

Before any database step, load the IMDB schema and data into that database
(`schema.sql`, `fkindexes.sql`, `copy.sql`).

> During data generation the utility can appear to hang for a long time when
> moving between query groups (see §1.2) — this is expected, not a crash.

---

## 1. Data generation for training

### 1.1. Resources and query encoding (no database)

```bash
cd AlphaJoin1.0
python 1.getResource.py
python 2.getQueryEncode.py
```

This creates: `resource/jobtablename/*`, `resource/shorttolong`, and in `AlphaJoin1.0` — `predicatesEncodedDict` and `queryEncodedDict`.

Copy the dictionary to the `tools` directory (required by the Java utility):

```bash
cp AlphaJoin1.0/queryEncodedDict tools/
```

### 1.2. Collecting query execution times (requires a running database)

Set the `PG*` environment variables for your database if they differ from the defaults (see the database section above).

```bash
cd tools
javac -cp .:postgresql-42.2.6.jar JDBCUtil.java getruntimerandom.java
java -cp .:postgresql-42.2.6.jar getruntimerandom
```

**Behaviour during generation:**

- The utility runs in an infinite loop, enumerating join orders by query groups (1a–1d, 2a–2d, …, 10a–10c, 11a–11d, …).
- For each group it tries many combinations and runs a query against PostgreSQL for each, so the process is **very slow**.
- When moving to the next group (e.g. after 10a, 10b, 10c when starting the 11x queries), the program can **appear to hang** for a long time — 30 minutes, an hour or more — **with no console output**. This is not a crash: it is enumerating and executing all variants for the new group. Just wait.
- Lines are written to `tools/t6.sql` in the format: `queryName,hintCore,runtime,...`
- To stop: press **Ctrl+C** manually after collecting enough data.

### 1.3. Preparing training data

```bash
mkdir -p AlphaJoin1.0/data
cp tools/t6.sql AlphaJoin1.0/data/runtime.csv
cd AlphaJoin1.0
python 5.pretreatment.py
```

This produces `traindata.sql` and `testdata.sql` in `AlphaJoin1.0/data/`.

---

## 2. Model training

From the `AlphaJoin1.0` directory:

```bash
cd AlphaJoin1.0
python 6.train_network.py
```

Arguments are taken from `0.arguments.py` (default `--save-dir saved_models/`). The trained model is saved to that directory.

**Monitoring training with plots:** The Jupyter notebook `training_curves.ipynb` reads a training log file (e.g. `AlphaJoin1.0/learning_logs.out` or a `train_*_log.log` you point it to), plots **Loss** and **Train/Test accuracy**, and can update on button click or auto-refresh when the log file changes (every 3 sec; auto-refresh stops after 10 minutes without changes). Adjust `LOG_PATH` in the notebook to match your log file. Run training in a separate terminal (e.g. `cd AlphaJoin1.0 && bash run_train.sh`) while viewing the notebook.

---

## 3. Obtaining predictions (best plans, leading)

The script `8.findBestPlan.py` uses the trained model and MCTS to output, for each query, a recommended join order in **Leading** hint format.

Run from `AlphaJoin1.0` (where `saved_models/` and resources are available):

```bash
cd AlphaJoin1.0
python 8.findBestPlan.py > ../tools/hint.txt
```

Format of lines in `hint.txt`: `queryName , leading_hint , search_time` (comma-separated). This file is needed for the next step — testing queries with hints.

---

## 4. Testing: running queries with Leading

Correct order:

1. Run **findBestPlan** and save its output to **hint.txt** (as above).
2. Run the utility that, using **hint.txt**, executes queries with the `/*+ Leading(...) */` hint and measures time.

From the `tools` directory:

```bash
cd tools
# hint.txt must already exist from the previous step
javac -cp .:postgresql-42.2.6.jar JDBCUtil.java gethintresult.java
java -cp .:postgresql-42.2.6.jar gethintresult
```

The utility reads `tools/hint.txt`, prepends `Leading(...)` to each query, runs `EXPLAIN ANALYZE`, and prints execution time and plan.

---

## Quick command sequence

```bash
# 1. Resources and encoding
cd AlphaJoin1.0
python 1.getResource.py && python 2.getQueryEncode.py
cp AlphaJoin1.0/queryEncodedDict tools/

# 2. Generation (database must be up, config as in JDBCUtil)
cd tools
javac -cp .:postgresql-42.2.6.jar JDBCUtil.java getruntimerandom.java
java -cp .:postgresql-42.2.6.jar getruntimerandom
# Wait for enough data; long "hangs" between query groups are possible. Stop with Ctrl+C.

# 3. Data preparation and training
mkdir -p AlphaJoin1.0/data
cp tools/t6.sql AlphaJoin1.0/data/runtime.csv
cd AlphaJoin1.0
python 5.pretreatment.py
python 6.train_network.py

# 4. Leading prediction and testing
python 8.findBestPlan.py > ../tools/hint.txt
cd tools
javac -cp .:postgresql-42.2.6.jar JDBCUtil.java gethintresult.java
java -cp .:postgresql-42.2.6.jar gethintresult
```

Adjust paths and directory names as needed for your setup. The `tools/dropCache.sh` script restarts PostgreSQL and drops OS cache — change the service name inside it if needed (e.g. `postgresql-10` to your version).
