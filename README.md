# CorpIQ - Corporate Data Harmonization & ML Platform
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![MLflow 3.8.0](https://img.shields.io/badge/mlflow-3.8.0-green.svg)](https://mlflow.org/docs/latest/index.html)
[![Apache Iceberg 1.7.1](https://img.shields.io/badge/iceberg-1.7.1-lightgrey.svg)](https://iceberg.apache.org/)
[![Apache Airflow 3.1.5](https://img.shields.io/badge/airflow-3.1.5-red.svg)](https://airflow.apache.org/docs/apache-airflow/stable/index.html)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Apache Spark 3.5.0](https://img.shields.io/badge/spark-3.5.0-orange.svg)](https://spark.apache.org/releases/spark-release-3-5-0.html)

**CorpIQ** is an enterprise-grade data harmonization and machine learning platform that performs entity resolution across multiple corporate data sources, enriches data with LLM-powered adverse media analysis, stores harmonized records in an Apache Iceberg lakehouse, and trains ML models to predict corporate profitability.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Entity Resolution Heuristic](#entity-resolution-heuristic)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Infrastructure Setup](#infrastructure-setup)
  - [Iceberg Metastore Configuration](#iceberg-metastore-configuration)
  - [Docker Services](#docker-services)
- [Running the Pipelines](#running-the-pipelines)
  - [Data Ingestion Pipeline](#data-ingestion-pipeline)
  - [ML Training Pipeline](#ml-training-pipeline)
  - [Orchestration with Airflow](#orchestration-with-airflow)
- [CI/CD Pipeline](#cicd-pipeline)
- [Querying the Iceberg Table](#querying-the-iceberg-table)
- [Viewing Registered Models](#viewing-registered-models)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture Overview

CorpIQ follows a modern lakehouse architecture pattern with the following components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│  ┌──────────────┐              ┌──────────────┐                 │
│  │  Source 1    │              │  Source 2    │                 │
│  │(Supply Chain)│              │ (Financials) │                 │
│  └──────────────┘              └──────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Data Quality Validation                        │
│  • Schema Validation                                            │
│  • Null Checks & Range Constraints                              │
│  • Statistical Profiling                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Entity Resolution Engine                       │
│  • Blocking Strategy (Country + Industry)                       │
│  • Fuzzy Matching (Jaro-Winkler, Token Sort)                    │
│  • Graph-based Connected Components (GraphFrames)               │
│  • Hash-based Deterministic IDs                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             LLM-Powered Adverse Media Analysis                  │
│  • OpenAI/Anthropic Integration                                 │
│  • Keyword-based Risk Scoring                                   │
│  • Batch Processing with Rate Limiting                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               Apache Iceberg Lakehouse                          │
│  • ACID Transactions                                            │
│  • Time Travel & Schema Evolution                               │
│  • Merge-on-Read Optimization                                   │
│  • S3/HDFS Storage Layer                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ML Pipeline (Spark ML)                        │
│  • Feature Engineering (OneHot, Scaling, TF-IDF)                │
│  • Model Training (LR, RF, GBT)                                 │
│  • Hyperparameter Tuning (Grid Search, Cross-Validation)        │
│  • Model Evaluation & Validation                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     MLflow Model Registry                       │
│  • Experiment Tracking                                          │
│  • Model Versioning                                             │
│  • Stage Transitions (Staging → Production)                     │
│  • Model Serving via FastAPI                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| **Data Processing** | Apache Spark 3.5.0 (PySpark) |
| **Data Storage** | Apache Iceberg 1.7.1 (REST Catalog) |
| **ML Tracking** | MLflow 3.8.0 |
| **Orchestration** | Apache Airflow 3.1.5 |
| **API** | FastAPI + Uvicorn |
| **LLM** | OpenAI GPT-4 / Anthropic Claude |
| **Containerization** | Docker Compose |
| **Testing** | pytest + pyspark |

---

## Entity Resolution Heuristic

The entity resolution engine implements a **multi-stage probabilistic matching strategy** optimized for large-scale corporate data:

### 1. **Blocking Strategy**
To reduce the O(n²) comparison space, records are grouped into blocks using:
- **Country** (normalized to uppercase)
- **First 3 characters of corporate name** (after cleaning)

```python
blocking_key = concat_ws("||", 
    coalesce(country_clean, "UNK"),
    substring(corporate_name_clean, 1, 3)
)
```

This reduces comparisons from ~1M² to ~1000 blocks of ~1000 records each.

### 2. **Text Normalization**
Corporate names undergo aggressive cleaning:
```python
def clean_text(text: str) -> str:
    text = text.lower()
    # Remove legal suffixes: Inc, Corp, Ltd, LLC, etc.
    text = re.sub(r'\b(inc|corp|ltd|llc|plc|co|company|limited)\b', '', text)
    # Remove punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', text).strip()
```

### 3. **Fuzzy Matching with Weighted Scoring**
Within each block, candidate pairs are scored using:

| Field | Method | Weight | Library |
|-------|--------|--------|---------|
| **Corporate Name** | Jaro-Winkler Similarity | 0.6 | `jellyfish` |
| **Address** | Token Sort Ratio | 0.3 | `fuzzywuzzy` |
| **Postal Code** | Exact Match | 0.1 | native |

```python
score = (jaro_winkler(name1, name2) * 0.6) + 
        (token_sort_ratio(addr1, addr2) * 0.3) + 
        (1.0 if postal1 == postal2 else 0.0) * 0.1
```

**Match Threshold**: Pairs with `score ≥ 0.85` (configurable) are considered duplicates.

### 4. **Graph-Based Connected Components**
Matched pairs form edges in an undirected graph. We use **GraphFrames** to compute connected components:

```python
vertices = union(source1_records, source2_records).select("id")
edges = matched_pairs.select("id_s1" as "src", "id_s2" as "dst")
graph = GraphFrame(vertices, edges)
components = graph.connectedComponents()
```

Each connected component represents a unique corporate entity.

### 5. **Deterministic Hash-Based IDs**
Instead of sequential IDs, we generate **content-addressable corporate IDs**:

```python
fingerprint = sha2(
    concat_ws("|",
        sort_array(collect_set(corporate_name_clean)),
        sort_array(collect_set(country_clean)),
        sort_array(collect_set(postal_code_clean))
    ), 256
)
corporate_id = concat("CORP-", substring(fingerprint, 1, 16))
```

This ensures:
- **Idempotency**: Re-running produces the same IDs
- **Traceability**: IDs encode entity attributes
- **Collision Resistance**: SHA-256 truncation provides ~2⁶⁴ unique IDs

---

## Prerequisites

- **Python 3.12+** (tested on 3.12.0)
- **Java 17+** (required for Spark)
- **Docker & Docker Compose** (for infrastructure services)
- **AWS Account** (for S3 storage, optional for local dev)
- **16GB+ RAM** (recommended for Spark executors)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/corpiq.git
cd corpiq
```

### 2. Create Python Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the project root:

```bash
# AWS Credentials (for S3 storage)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Iceberg Configuration
ICEBERG_REST_URI=http://localhost:8181
ICEBERG_WAREHOUSE=s3a://your-bucket/iceberg-warehouse/
# Or for local testing: file:///tmp/iceberg-warehouse

# Data Sources
SOURCE1_PATH=./data/source1.csv
SOURCE2_PATH=./data/source2.csv

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5112

# LLM API Keys (choose one)
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Airflow Database
AIRFLOW_DB_CONN=postgresql+psycopg2://admin:admin@localhost:5432/app

# Monitoring
CHECKPOINT_LOCATION=./tmp/checkpoints
LOG_FILE_PATH=./logs/pipeline.log
```

### 5. Verify Installation

```bash
# Check Java version
java -version  # Should be 17+

# Check Python packages
python -c "import pyspark; print(pyspark.__version__)"  # Should print 3.5.0

# Run basic Spark test
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[2]').getOrCreate(); print('Spark OK')"
```

---

## Infrastructure Setup

### Iceberg Metastore Configuration

CorpIQ uses **Apache Iceberg REST Catalog** for metadata management. There are two deployment options:

#### Option A: Docker-based REST Catalog (Recommended for Development)

The `docker-compose.yml` includes a pre-configured Iceberg REST service:

```yaml
iceberg-rest:
  image: tabulario/iceberg-rest:0.6.0
  ports:
    - "8181:8181"
  environment:
    CATALOG_WAREHOUSE: ${ICEBERG_WAREHOUSE}
    CATALOG_IO__IMPL: org.apache.iceberg.aws.s3.S3FileIO
    AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
    AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    AWS_REGION: ${AWS_REGION}
```

Start the service:

```bash
docker-compose up -d iceberg-rest
```

Verify it's running:

```bash
curl http://localhost:8181/v1/config
```

Expected response:
```json
{
  "defaults": {},
  "overrides": {}
}
```

#### Option B: Hadoop Catalog (For Local Testing)

For lightweight local development without S3, use Hadoop catalog:

1. Update `src/config/config.yaml`:

```yaml
iceberg:
  catalog_name: local_catalog
  catalog_type: hadoop  # Changed from 'rest'
  warehouse: file:///tmp/iceberg-warehouse
```

2. Update Spark configuration in `src/utils/helper.py`:

```python
.config("spark.sql.catalog.local_catalog", "org.apache.iceberg.spark.SparkCatalog")
.config("spark.sql.catalog.local_catalog.type", "hadoop")
.config("spark.sql.catalog.local_catalog.warehouse", "/tmp/iceberg-warehouse")
```

3. Create warehouse directory:

```bash
mkdir -p /tmp/iceberg-warehouse
```

### Docker Services

Start all infrastructure services:

```bash
# Start all services
docker-compose up -d

# Verify services are healthy
docker-compose ps

# View logs
docker-compose logs -f
```

Services and their ports:

| Service | Port | UI/Endpoint |
|---------|------|-------------|
| **MLflow Tracking Server** | 5112 | http://localhost:5112 |
| **Airflow Webserver** | 8823 | http://localhost:8823 |
| **Iceberg REST Catalog** | 8181 | http://localhost:8181 |
| **Spark Master UI** | 8080 | http://localhost:8080 |
| **Model Serving API** | 8000 | http://localhost:8000/docs |
| **PostgreSQL** | 5432 | - |

#### Service Health Checks

```bash
# MLflow
curl http://localhost:5112/health

# Airflow
curl http://localhost:8823/api/v2/monitor/health

# Model API
curl http://localhost:8000/health

# Spark Master
curl http://localhost:8080
```

---

## Running the Pipelines

### Data Ingestion Pipeline

The ingestion pipeline performs entity resolution, data quality checks, and adverse media analysis.

#### Basic Execution

```bash
# Run full ingestion pipeline
python -m src.pipelines.ingestion_pipeline \
    --config src/config/config.yaml

# Skip adverse media analysis (faster)
python -m src.pipelines.ingestion_pipeline \
    --config src/config/config.yaml \
    --skip-adverse-media

# Run with table optimization
python -m src.pipelines.ingestion_pipeline \
    --config src/config/config.yaml \
    --optimize
```

#### Query the Registry

```bash
# Query all records (limit 100)
python -m src.pipelines.ingestion_pipeline \
    --config src/config/config.yaml \
    --query "1=1"

# Filter by country
python -m src.pipelines.ingestion_pipeline \
    --config src/config/config.yaml \
    --query "country = 'USA'"

# Filter by high-risk entities
python -m src.pipelines.ingestion_pipeline \
    --config src/config/config.yaml \
    --query "risk_score > 0.7"
```

#### Expected Output

```
2025-01-01 10:00:00 | INFO | Starting Corporate Data Harmonization Pipeline
2025-01-01 10:00:02 | INFO | Database corporate_db.corporate_registry ready
2025-01-01 10:00:05 | INFO | Loaded 50,000 records from Source 1
2025-01-01 10:00:08 | INFO | Loaded 45,000 records from Source 2
2025-01-01 10:01:23 | INFO | Entity resolution complete. Found 72,341 unique entities
2025-01-01 10:03:45 | INFO | Upsert complete. Records after merge: 72,341
2025-01-01 10:03:46 | INFO | Pipeline completed successfully in 226.3 seconds

DATA METRICS SUMMARY
--------------------
Record Counts:
  Source 1 Records: 50,000
  Source 2 Records: 45,000
  Unique Entities: 72,341
  Adverse Media Found: 234

Execution Times:
  data_loading: 8.2s
  entity_resolution: 78.5s
  data_quality: 3.1s
  adverse_media: 125.7s
  iceberg_upsert: 10.8s
  total_data_pipeline: 226.3s
```

### ML Training Pipeline

The ML pipeline trains models to predict corporate profitability.

#### Basic Execution

```bash
# Run full ML pipeline
python -m src.pipelines.ml_pipeline \
    --config src/config/config.yaml

# Train only (skip evaluation)
python -m src.pipelines.ml_pipeline \
    --config src/config/config.yaml \
    --train-only

# Skip validation checks
python -m src.pipelines.ml_pipeline \
    --config src/config/config.yaml \
    --skip-validation

# Resume from checkpoint
python -m src.pipelines.ml_pipeline \
    --config src/config/config.yaml \
    --resume
```

#### Expected Output

```
2025-01-01 11:00:00 | INFO | Starting ML Pipeline
2025-01-01 11:00:05 | INFO | Data split complete
  Train: 57,873 samples
  Val: 7,234 samples
  Test: 7,234 samples

2025-01-01 11:00:10 | INFO | Class balance:
  Class 0: 43,405 (60%)
  Class 1: 28,936 (40%)

2025-01-01 11:02:34 | INFO | Training complete in 144.2s (57,873 samples)
2025-01-01 11:03:12 | INFO | Evaluation complete

ML METRICS SUMMARY
------------------
Classification Metrics:
  Accuracy:     0.8734
  Precision:    0.8512
  Recall:       0.8901
  F1 Score:     0.8702

ROC Metrics:
  AUC-ROC:      0.9245
  AUC-PR:       0.8967

Confusion Matrix:
                Predicted
                0        1
  Actual 0    4,123      298
  Actual 1      217    2,596

2025-01-01 11:03:15 | INFO | Model registered: corporate_profit_predictor version 3
2025-01-01 11:03:15 | INFO | Model version 3 moved to Staging
```

### Orchestration with Airflow

The Airflow DAG runs both pipelines sequentially on a daily schedule.

#### Access Airflow UI

1. Navigate to http://localhost:8823
2. Default credentials: `admin` / `admin` (configured in Airflow standalone mode)
3. Enable the DAG: `corporate_data_ml_pipeline`

#### DAG Structure

```
load_and_validate_config
         ↓
data_harmonization_ingestion
         ↓
ml_model_training
```

#### Manual Trigger

Via UI:
- Click on `corporate_data_ml_pipeline`
- Click "Trigger DAG" button (play icon)

Via CLI:
```bash
docker exec -it airflow-webserver airflow dags trigger corporate_data_ml_pipeline
```

#### Monitor DAG Execution

```bash
# View task logs
docker exec -it airflow-webserver airflow tasks logs corporate_data_ml_pipeline data_harmonization_ingestion 2025-01-01

# List DAG runs
docker exec -it airflow-webserver airflow dags list-runs -d corporate_data_ml_pipeline
```

---

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment.

### CI Workflow (`.github/workflows/ci.yaml`)

Runs on **all pull requests** and **pushes to non-main branches**:

1. **Setup**: Python 3.12 + Java 17
2. **Linting**: `ruff check src tests`
3. **Testing**: `pytest -v tests/`

### CD Workflow (`.github/workflows/cd.yaml`)

Runs on **pushes to main branch**:

1. **Build**: Package ETL code as `etl_bundle.tar.gz`
2. **Deploy**: Upload to S3 at `s3://my-etl-artifacts/`

### Required GitHub Secrets

Configure these in your repository settings (Settings → Secrets and variables → Actions):

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

### Manual Trigger

You can manually trigger workflows from the Actions tab in GitHub.

### Local CI/CD Testing with Act

Test GitHub Actions workflows locally before pushing using [act](https://github.com/nektos/act):

#### Install Act

```bash
# macOS
brew install act

# Linux
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Windows (via Chocolatey)
choco install act-cli
```

#### Run Workflows Locally

```bash
# Test CI workflow (pull_request trigger)
act pull_request

# Test CD workflow (push to main)
act push -b main

# List available workflows
act -l

# Run specific job
act -j ci

# Dry run (show what would run without executing)
act -n

# Use specific runner image
act --container-architecture linux/amd64
```

#### Configure Secrets for Act

Create `.secrets` file in project root (add to `.gitignore`):

```bash
AWS_ACCESS_KEY_ID=your_test_key
AWS_SECRET_ACCESS_KEY=your_test_secret
```

Run with secrets:

```bash
act --secret-file .secrets
```

#### Common Act Options

```bash
# Use specific event file
act -e .github/workflows/event.json

# Verbose output
act -v

# Use specific platform
act -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

**Note**: Act runs workflows in Docker containers. Ensure Docker is running before using act.

### Local CI Simulation

```bash
# Run linting
ruff check src tests

# Run tests with coverage
pytest --cov=src --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

---

## Querying the Iceberg Table

### Using Python (PySpark)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Iceberg Query") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local_catalog.type", "rest") \
    .config("spark.sql.catalog.local_catalog.uri", "http://localhost:8181") \
    .getOrCreate()

# Read the table
df = spark.table("local_catalog.corporate_db.corporate_registry")

# Query examples
df.show(10, truncate=False)

# Filter by country
df.filter("country = 'USA'").show()

# High-risk entities
df.filter("risk_score > 0.7").select("corporate_id", "corporate_name_clean", "risk_score").show()

# Recent updates
df.filter("updated_at > current_date() - interval 7 days").show()
```

### Using SQL (Spark SQL)

```python
spark.sql("""
    SELECT corporate_id, corporate_name_clean, country, revenue, profit, risk_score
    FROM local_catalog.corporate_db.corporate_registry
    WHERE profit > 10000000
    ORDER BY profit DESC
    LIMIT 20
""").show()
```

### Time Travel Queries

```python
# Query as of specific snapshot
spark.sql("""
    SELECT * FROM local_catalog.corporate_db.corporate_registry
    VERSION AS OF 42
""").show()

# Query as of timestamp
spark.sql("""
    SELECT * FROM local_catalog.corporate_db.corporate_registry
    TIMESTAMP AS OF '2025-01-01 10:00:00'
""").show()

# View snapshot history
spark.sql("""
    SELECT * FROM local_catalog.corporate_db.corporate_registry.snapshots
    ORDER BY committed_at DESC
""").show()
```

### Using the FastAPI Endpoint

```bash
# Query via REST API
curl -X POST http://localhost:8000/iceberg/query \
  -H "Content-Type: application/json" \
  -d '{
    "filter_condition": "country = '\''USA'\'' AND profit > 5000000",
    "limit": 50
  }'

# Get table statistics
curl http://localhost:8000/iceberg/stats

# Time travel query
curl -X POST http://localhost:8000/iceberg/time-travel \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-01-01 10:00:00"
  }'
```

### Table Metadata Queries

```python
# Get table statistics
from src.storage.iceberg_manager import IcebergTableManager
from src.utils.helper import create_spark_session, load_config

config = load_config("src/config/config.yaml")
spark = create_spark_session(config)
manager = IcebergTableManager(config, spark)

stats = manager.get_table_statistics()
print(f"Total rows: {stats['row_count']:,}")
print(f"Total files: {stats['file_count']}")
print(f"Total size: {stats['total_size_mb']:.2f} MB")
print(f"Snapshots: {stats['snapshot_count']}")

# View file-level details
spark.sql("""
    SELECT file_path, file_size_in_bytes, record_count
    FROM local_catalog.corporate_db.corporate_registry.files
    ORDER BY record_count DESC
""").show()
```

---

## Viewing Registered Models

### MLflow UI

1. Navigate to http://localhost:5112
2. Click on "Models" in the left sidebar
3. Find the model: `corporate_profit_predictor`

### Model Details

Click on the model to view:
- **Versions**: All registered versions with stage (None/Staging/Production)
- **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC for each version
- **Artifacts**: Model files, feature importance plots, confusion matrices
- **Tags**: Custom metadata (algorithm, training date, etc.)

### Programmatic Access

```python
from src.storage.mlflow_manager import MLflowManager
from src.utils.helper import load_config

config = load_config("src/config/config.yaml")
manager = MLflowManager(config)

# Load production model
model = manager.load_model(
    model_name="corporate_profit_predictor",
    stage="Production"
)

# Load specific version
model = manager.load_model(
    model_name="corporate_profit_predictor",
    version="3"
)

# Compare models
run_ids = ["run_abc123", "run_def456", "run_ghi789"]
comparison_df = manager.compare_models(run_ids)
print(comparison_df[["run_name", "accuracy", "f1", "areaUnderROC"]])
```

### Transition Model Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promote to production
client.transition_model_version_stage(
    name="corporate_profit_predictor",
    version="3",
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="corporate_profit_predictor",
    version="2",
    stage="Archived"
)
```

---

## API Usage

The FastAPI server provides REST endpoints for model serving and data queries.

### Start the API Server

```bash
# Development mode (auto-reload)
uvicorn src.server.app:app --reload --host 0.0.0.0 --port 8000

# Production mode (via Docker)
docker-compose up -d model-server
```

### API Documentation

Interactive docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Requests

#### Health Check

```bash
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

#### Model Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "corporate_profit_predictor",
    "stage": "Production",
    "records": [
      {
        "revenue": 150000000.0,
        "profit": 25000000.0,
        "country": "USA",
        "industry_sector": "Technology",
        "num_top_suppliers": 12,
        "num_main_customers": 8
      }
    ]
  }'

# Response:
# {
#   "predictions": [1.0]
# }
```

#### Run ML Pipeline

```bash
curl -X POST http://localhost:8000/pipeline/ml/run \
  -H "Content-Type: application/json" \
  -d '{
    "train_only": false,
    "skip_validation": false,
    "resume_from_checkpoint": false
  }'
```

#### Run Ingestion Pipeline

```bash
curl -X POST http://localhost:8000/pipeline/data/run \
  -H "Content-Type: application/json" \
  -d '{
    "run_adverse_media": true,
    "optimize_table": false
  }'
```

---

## Configuration

All configuration is centralized in `src/config/config.yaml`. Key sections:

### Entity Resolution

```yaml
entity_resolution:
  matching_threshold: 0.85  # Similarity threshold for matches
  blocking_fields:
    - country
    - industry_sector
  fuzzy_match_fields:
    - name: corporate_name
      weight: 0.6
      method: jaro_winkler
    - name: address
      weight: 0.3
      method: token_sort
    - name: postal_code
      weight: 0.1
      method: exact
```

### ML Model

```yaml
model:
  algorithm: logistic_regression  # Options: logistic_regression, random_forest, gradient_boosted_trees
  hyperparameters:
    logistic_regression:
      maxIter: 100
      regParam: 0.01
      elasticNetParam: 0.0

hyperparameter_tuning:
  enabled: true
  method: grid_search
  param_grid:
    logistic_regression:
      regParam: [0.01, 0.1, 0.5]
      elasticNetParam: [0.0, 0.5, 1.0]
```

### Adverse Media Analysis

```yaml
llm:
  provider: openai  # Options: openai, anthropic
  model: gpt-4
  api_key_env: OPENAI_API_KEY
  adverse_media:
    search_enabled: true
    keywords:
      - fraud
      - lawsuit
      - scandal
      - violation
      - investigation
    search_days_back: 365
    batch_size: 10
```

### Data Quality

```yaml
data_quality:
  validation_rules:
    corporate_name:
      - type: not_null
      - type: min_length
        value: 2
    revenue:
      - type: not_null
      - type: positive
      - type: range
        min: 0
        max: 1000000000000
```

---

## Testing

### Run All Tests

```bash
pytest -v tests/
```

### Run Specific Test Suites

```bash
# Unit tests only
pytest -v tests/unit/

# Integration tests only
pytest -v tests/integration/

# Specific test file
pytest -v tests/unit/test_entity_resolution.py

# Specific test
pytest -v tests/unit/test_entity_resolution.py::TestEntityMatcher::test_fuzzy_matching
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html
```

### Test Requirements

- Tests require **Java 17+** for Spark
- Set `SPARK_LOCAL_IP=127.0.0.1` if running locally
- Integration tests use temporary Iceberg tables

---

## Troubleshooting

### Common Issues

#### 1. Spark Java Heap Space Error

**Symptom**: `java.lang.OutOfMemoryError: Java heap space`

**Solution**: Increase Spark memory allocation in `.env`:

```bash
SPARK_DRIVER_MEMORY=4g
SPARK_EXECUTOR_MEMORY=4g
```

#### 2. Iceberg Table Not Found

**Symptom**: `org.apache.iceberg.exceptions.NoSuchTableException`

**Solution**: Verify catalog initialization:

```python
from src.storage.iceberg_manager import IcebergTableManager
manager = IcebergTableManager(config, spark)
manager.initialize_catalog_and_database()
manager.create_table_if_not_exists()
```

#### 3. MLflow Connection Error

**Symptom**: `ConnectionError: Max retries exceeded with url: http://localhost:5112`

**Solution**: Ensure MLflow server is running:

```bash
docker-compose up -d mlflow-server
curl http://localhost:5112/health
```

#### 4. GraphFrames Not Found

**Symptom**: `java.lang.ClassNotFoundException: org.graphframes.GraphFrame`

**Solution**: Verify Spark packages are loaded:

```python
spark = SparkSession.builder \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .getOrCreate()
```

#### 5. Airflow DAG Not Showing

**Symptom**: DAG not visible in Airflow UI

**Solution**:
1. Check DAG file syntax: `python src/airflow/dags/full_pipeline.py`
2. Verify `AIRFLOW__CORE__DAGS_FOLDER` points to correct directory
3. Restart Airflow: `docker-compose restart airflow-server`

### Logs

- **Pipeline Logs**: `logs/pipeline.log`
- **Docker Logs**: `docker-compose logs -f <service-name>`
- **Checkpoint Metadata**: `./tmp/checkpoints/`
- **MLflow Artifacts**: Check MLflow UI → Experiments → Artifacts tab

---

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run linting: `ruff check --fix src tests`
5. Run tests: `pytest -v tests/`
6. Commit: `git commit -m "Add amazing feature"`
7. Push: `git push origin feature/amazing-feature`
8. Open a Pull Request
