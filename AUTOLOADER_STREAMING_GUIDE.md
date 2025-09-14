# NYC Taxi Autoloader Streaming Guide

This guide explains the new file-based streaming approach using Databricks Autoloader, which provides true streaming ingestion without full table refreshes.

## Architecture Overview

The solution consists of two main components:

1. **Data Export Job** (`src/data_export_job.py`): Exports historical taxi data to volume files
2. **Autoloader DLT Pipeline** (`src/dlt_autoloader_pipeline.ipynb`): Streams data from volume files using Autoloader

## Data Flow

```
Historical Data (samples.nyctaxi.trips)
    ↓ (Export Job - every 5 minutes)
Volume Files (/Volumes/nyc_trips_dev/bronze/landing_zone)  
    ↓ (Autoloader - real-time streaming)
DLT Pipeline (Bronze → Silver → Gold)
    ↓
Analytics Tables
```

## Setup Instructions

### 1. Create Volume in Databricks

First, create the required volume in your Databricks workspace:

```sql
-- Create volume for landing zone files
CREATE VOLUME IF NOT EXISTS nyc_trips_dev.bronze.landing_zone;

-- Verify volume exists
SHOW VOLUMES IN nyc_trips_dev.bronze;
```

### 2. Deploy Data Export Job

The data export job is configured in `resources/data_export.job.yml`:

```bash
# Deploy the export job
databricks bundle deploy --target dev

# Start the export job (initially paused)
databricks jobs run-now --job-id <job_id>
```

**Job Configuration:**
- **Schedule**: Every 5 minutes (configurable)
- **Data Selection**: Current timestamp - 1 hour
- **Output Format**: Parquet files in volume
- **Sample Rate**: 75% of matching historical data

### 3. Deploy Autoloader Pipeline

The streaming pipeline is configured in `resources/autoloader_streaming.pipeline.yml`:

```bash
# Deploy the autoloader pipeline
databricks bundle deploy --target dev

# Start the pipeline
databricks pipelines start-update <pipeline_id>
```

## Key Features

### Data Export Job

**File:** `src/data_export_job.py`

- **Time-shifted Processing**: Exports data for `current_time - 1 hour`
- **Batch Identification**: Each file has unique batch ID and timestamp
- **Schema Consistency**: Ensures consistent parquet schema
- **Volume Integration**: Writes directly to Databricks volumes

**Key Functions:**
- `get_previous_hour_data()`: Retrieves and transforms historical data
- `export_batch()`: Writes single batch to volume
- `continuous_export()`: Runs continuous export for testing

### Autoloader Pipeline

**File:** `src/dlt_autoloader_pipeline.ipynb`

**Landing Zone (`taxi_raw_landing`)**:
- Uses `cloudFiles` format for Autoloader
- Automatic schema inference and evolution
- File metadata tracking (`_metadata.file_path`)

**Bronze Layer (`taxi_bronze`)**:
- Data quality validations with DLT expectations
- Schema enforcement and type casting
- Trip duration calculations

**Silver Layer (`taxi_silver_trips`)**:
- Business logic transformations
- Trip categorization (short/medium/long)
- Time-based enrichments (hour, day of week)

**Gold Layer (`taxi_gold_hourly_metrics`)**:
- Hourly aggregated metrics
- Business intelligence ready data
- Quality metrics tracking

## Volume Structure

```
/Volumes/nyc_trips_dev/bronze/landing_zone/
├── taxi_batch_20241201_143022_a1b2c3d4.parquet
├── taxi_batch_20241201_143522_e5f6g7h8.parquet
├── taxi_batch_20241201_144022_i9j0k1l2.parquet
└── _schemas/
    └── (Autoloader schema files)
```

## Configuration Options

### Export Job Parameters

In `resources/data_export.job.yml`:

```yaml
parameters:
  - name: "volume_path"
    default: "/Volumes/nyc_trips_dev/bronze/landing_zone"
  - name: "sample_fraction" 
    default: "0.75"
```

### Pipeline Configuration

In `resources/autoloader_streaming.pipeline.yml`:

```yaml
configuration:
  # Checkpoint location for streaming state
  "spark.sql.streaming.checkpointLocation": "/Volumes/nyc_trips_dev/bronze/_checkpoints/taxi_autoloader"
  
  # Processing frequency
  "spark.sql.streaming.trigger.processingTime": "30 seconds"
  
  # Schema management
  "pipelines.autoloader.schemaLocation": "/Volumes/nyc_trips_dev/bronze/landing_zone/_schemas"
```

## Monitoring and Operations

### Export Job Monitoring

```python
# Check export job status
from data_export_job import DataExporter

exporter = DataExporter()
result = exporter.export_batch(spark)
print(f"Exported batch: {result}")
```

### Pipeline Monitoring

```sql
-- Check pipeline status
DESCRIBE HISTORY taxi_bronze;

-- Monitor data quality
SELECT 
    export_hour,
    COUNT(*) as records,
    AVG(fare_amount) as avg_fare
FROM taxi_bronze 
GROUP BY export_hour 
ORDER BY export_hour DESC
LIMIT 10;
```

### Volume Monitoring

```sql
-- List files in volume
LIST '/Volumes/nyc_trips_dev/bronze/landing_zone/';

-- Check volume usage
DESCRIBE VOLUME nyc_trips_dev.bronze.landing_zone;
```

## Advantages of Autoloader Approach

1. **True Streaming**: No full table refreshes, only processes new files
2. **Cost Effective**: Pay only for processed data
3. **Schema Evolution**: Automatic handling of schema changes
4. **Fault Tolerance**: Built-in checkpointing and recovery
5. **Scalability**: Handles varying data volumes automatically
6. **File Tracking**: Prevents duplicate processing

## Troubleshooting

### Common Issues

**Issue**: Pipeline fails with schema mismatch
**Solution**: Check schema hints in pipeline configuration

**Issue**: No new data flowing
**Solution**: Verify export job is running and writing to volume

**Issue**: Duplicate records
**Solution**: Check Autoloader checkpoint location

### Useful Commands

```bash
# Reset pipeline
databricks pipelines stop <pipeline_id>
databricks pipelines reset <pipeline_id>

# Clean volume (CAREFUL!)
# This will delete all files - only for development
rm -rf /Volumes/nyc_trips_dev/bronze/landing_zone/*
```

## Next Steps

1. **Production Deployment**: Update target from `dev` to `prod`
2. **Monitoring Setup**: Configure alerts and dashboards
3. **Data Retention**: Implement file cleanup policies
4. **Security**: Configure proper access controls
5. **Performance Tuning**: Optimize batch sizes and triggers

## Cost Optimization

- **Serverless Compute**: Uses pay-per-use serverless DLT
- **Efficient Batching**: Configurable file processing limits
- **Auto-scaling**: Export job cluster scales 1-3 workers
- **Volume Storage**: Cost-effective Unity Catalog volumes
