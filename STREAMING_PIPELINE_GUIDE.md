# 🚀 NYC Taxi Streaming Simulation Pipeline

This guide explains how to use the enhanced DLT pipeline that simulates streaming NYC taxi data by mapping historical 2016 data to current dates.

## 📋 Overview

The pipeline implements a **data simulation strategy** where:
- Historical data from **January/February 2016** is mapped to **current dates**
- Data is injected **hourly** based on current day and time
- **75% sampling** is applied to control data volume
- Full **medallion architecture** (Landing Zone → Bronze Layer) is implemented

## 🏗️ Architecture

```
NYC Sample Data (Jan/Feb 2016)
    ↓
Date Mapping Service (maps to current dates)
    ↓  
Landing Zone Table (taxi_landing_zone)
    ↓
Bronze Layer (taxi_trips_bronze) + Data Quality Checks
    ↓
Quality Monitoring (taxi_data_quality_metrics)
```

## 🛠️ Components

### 1. **Data Simulation Engine** (`src/data_simulation/`)
- **`DateMapper`**: Maps 2016 dates to current dates while preserving day/hour
- **`StreamingSimulator`**: Creates landing zone data with metadata

### 2. **Enhanced DLT Pipeline** (`src/dlt_enhanced_pipeline.ipynb`)
- **Landing Zone**: Raw data ingestion with simulation metadata
- **Bronze Layer**: Cleansed data with quality validations
- **Quality Monitoring**: Real-time metrics and health checks

## 📊 Data Schema

| Column | Type | Description |
|--------|------|-------------|
| `tpep_pickup_datetime` | timestamp | Pickup time (mapped to current dates) |
| `tpep_dropoff_datetime` | timestamp | Dropoff time (mapped to current dates) |
| `trip_distance` | double | Trip distance in miles |
| `fare_amount` | double | Fare amount in USD |
| `pickup_zip` | int | Pickup ZIP code |
| `dropoff_zip` | int | Dropoff ZIP code |

## 🚀 Deployment Instructions

### Step 1: Build and Deploy
```bash
# Build the wheel package
uv build --wheel

# Deploy to Databricks
databricks bundle deploy
```

### Step 2: Start the Pipeline
```bash
# Start the DLT pipeline
databricks bundle run ghithub_trends_digger_pipeline
```

### Step 3: Monitor the Job
```bash
# Monitor the scheduled job
databricks bundle run ghithub_trends_digger_job
```

## 📈 Data Quality Expectations

The bronze layer implements the following quality checks:

- **✅ Valid Timestamps**: Pickup and dropoff times must be non-null and logically ordered
- **✅ Positive Trip Distance**: Trip distance must be > 0
- **✅ Positive Fare Amount**: Fare amount must be > 0

Records failing these expectations are **automatically dropped**.

## 🧪 Testing Your Pipeline

Use the test script to validate functionality:

```python
# Run in Databricks notebook or cluster
%run src/test_simulation

# This will test:
# - Date mapping logic
# - Streaming simulation
# - Schema compliance
# - Data quality validation
```

## 📊 Monitoring and Analytics

### Access Your Data Tables

```sql
-- View landing zone data
SELECT * FROM workspace.ghithub_trends_digger_dev.taxi_landing_zone LIMIT 10;

-- View bronze layer data
SELECT * FROM workspace.ghithub_trends_digger_dev.taxi_trips_bronze LIMIT 10;

-- Check data quality metrics
SELECT * FROM workspace.ghithub_trends_digger_dev.taxi_data_quality_metrics;
```

### Sample Analytics Queries

```sql
-- Hourly trip counts
SELECT 
    hour(tpep_pickup_datetime) as pickup_hour,
    count(*) as trip_count,
    avg(fare_amount) as avg_fare
FROM workspace.ghithub_trends_digger_dev.taxi_trips_bronze 
GROUP BY hour(tpep_pickup_datetime)
ORDER BY pickup_hour;

-- Top pickup ZIP codes
SELECT 
    pickup_zip,
    count(*) as trip_count,
    avg(trip_distance) as avg_distance
FROM workspace.ghithub_trends_digger_dev.taxi_trips_bronze 
GROUP BY pickup_zip
ORDER BY trip_count DESC
LIMIT 10;
```

## ⚙️ Configuration Options

### Sampling Rate
Modify sampling in `DateMapper.simulate_current_hour_data()`:
```python
# Change from default 75% to different rate
simulated_df = date_mapper.simulate_current_hour_data(spark, sample_fraction=0.5)  # 50%
```

### Pipeline Schedule
Update in `resources/ghithub_trends_digger.job.yml`:
```yaml
trigger:
  periodic:
    interval: 1  # Change to different interval
    unit: HOURS   # Change from DAYS to HOURS for more frequent runs
```

## 🔍 Troubleshooting

### Common Issues

1. **No data generated**: Check that current day/hour has data in Jan/Feb 2016
2. **Import errors**: Ensure `bundle.sourcePath` is configured correctly
3. **Schema errors**: Verify all required columns are present

### Debug Commands
```python
# Test date mapping
from data_simulation.date_mapper import DateMapper
mapper = DateMapper()
day, hour, month = mapper.get_current_simulation_params()
print(f"Looking for data: Day={day}, Hour={hour}")

# Check source data availability
spark.sql("""
SELECT dayofmonth(tpep_pickup_datetime) as day, 
       hour(tpep_pickup_datetime) as hour, 
       count(*) as records
FROM samples.nyctaxi.trips 
WHERE month(tpep_pickup_datetime) IN (1,2) 
  AND year(tpep_pickup_datetime) = 2016
GROUP BY day, hour
ORDER BY day, hour
""").show()
```

## 🎯 Next Steps

This foundation enables you to:
- **Add Silver Layer**: Implement data enrichment and aggregations
- **Build Gold Layer**: Create analytics-ready views and dashboards  
- **Add ML Pipeline**: Implement outlier detection and predictive models
- **Integrate LLM**: Add contextual data enrichment capabilities

## 📞 Support

For issues or questions:
1. Check the `taxi_data_quality_metrics` view for pipeline health
2. Review DLT pipeline logs in Databricks
3. Run the test script to validate components
4. Check job execution logs for scheduling issues
