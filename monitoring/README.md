# Student Habits vs Academic Performance — Monitoring Pipeline

Monitoring new incoming data batches are scored, compared against a reference baseline, and their drift + performance metrics are stored in PostgreSQL and visualised in Grafana.

---

## Monitoing Pipeline

The monitoring pipeline detects data drift and tracks model performance over time by comparing new incoming data batches against the reference dataset.

1. Reference dataset (baseline)
2. Create Dataset batches
3. Monitor (drift + performance)
4. Store in PostgreSQL (Adminer)
5. Visualise in Grafana

---

## Structure

├── data/
│   └── raw/
│       └── student_habits_performance.csv
│   └── current_batches/            # Simulated incoming data batches
|
├── monitoring/
│   ├── docker-compose.yaml       
│   └── scripts/
│       ├── prepare_reference.py    
│       ├── generate_batch.py       
│       └── calculate_metrics.py
|

--- 

## How to Run

### 1. Start Process

```bash
cd .../monitoring
docker compose up -d
```
*note this must be done in the folder that holds the docker-compose.yaml file.

This starts:
- PostgreSQL on port 5432 — stores all monitoring metrics
- Adminer on port 8080 — web UI to inspect the database (http://localhost:8080)
- Grafana on port 3000 — dashboard for visualising metrics (http://localhost:3000)

### 2. Create the Reference Dataset

```bash
cd .../monitoring/scripts
python prepare_reference.py
```

Generates data/reference.csv — the baseline distribution the model was trained on.


### 3. Generate Batch and Calculate / Store Metrics

```bash
python generate_batch.py
python calculate_metrics.py
```

*run these steps multiple times to generate multiple batches

Samples 50 rows from the raw dataset, generates predictions, and saves a CSV to data/current_batches/.
Then loads most recent batch and runs a **Kolmogorov-Smirnov test** on each numeric feature to detect distributional drift. Computes **RMSE, MAE, R²** using the batch's actual and predicted exam scores.
This is then written into the metrics table in PostgreSQL.
