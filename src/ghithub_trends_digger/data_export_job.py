# COMMAND ----------
"""
Data Export Job for NYC Taxi Streaming Simulation

This job exports data batches to volume files that will be consumed by the
DLT Autoloader pipeline for true streaming ingestion.

Volume Path: /Volumes/nyc_trips_dev/bronze/landing_zone
"""

import sys
import uuid
from datetime import datetime, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import current_timestamp, lit


# Get Spark session first (for path configuration)
def get_spark() -> SparkSession:
    """Get or create Spark session"""
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        return SparkSession.builder.getOrCreate()


# Add src path for custom modules (will be set when class is instantiated)
try:
    spark = get_spark()
    bundle_path = spark.conf.get("bundle.sourcePath", ".")
    if bundle_path:
        sys.path.append(bundle_path)
except:
    # Fallback if spark is not available during import
    sys.path.append(".")
from data_simulation.date_mapper import DateMapper


class DataExporter:
    """
    Exports NYC taxi data to volume files for Autoloader consumption.

    This class handles the batch export logic that simulates streaming data
    by writing files with data from current_timestamp - 1 hour.
    """

    def __init__(self, volume_path: str = "/Volumes/nyc_trips_dev/bronze/landing_zone"):
        self.volume_path = volume_path
        self.date_mapper = DateMapper()

    def get_previous_hour_data(self, spark: SparkSession) -> DataFrame:
        """
        Get data for the previous hour (current_timestamp - 1).

        This simulates the behavior where we process data that's 1 hour behind
        the current time, which is typical in streaming scenarios.

        Args:
            spark: SparkSession instance

        Returns:
            DataFrame with previous hour's data
        """
        # Calculate previous hour parameters
        previous_hour = datetime.now() - timedelta(hours=1)
        target_day = previous_hour.day
        target_hour = previous_hour.hour
        target_month = previous_hour.month
        target_year = previous_hour.year

        print(f"Exporting data for: {previous_hour.strftime('%Y-%m-%d %H:00:00')}")

        # Read source data
        source_df = spark.read.table("samples.nyctaxi.trips")

        # Get historical filter condition for the target hour
        filter_condition = self.date_mapper.get_historical_filter_condition(target_day, target_hour)

        # Filter and sample data (using 75% sample rate)
        filtered_df = source_df.filter(filter_condition).sample(fraction=0.75, seed=42)

        # Transform dates to target time
        transformed_df = self.date_mapper.transform_dates_to_current(filtered_df, target_year, target_month)

        # Add export metadata
        export_df = (
            transformed_df.withColumn("export_timestamp", current_timestamp())
            .withColumn("batch_id", lit(str(uuid.uuid4())))
            .withColumn("source_system", lit("nyc_taxi_export"))
            .withColumn("export_hour", lit(f"{target_year}-{target_month:02d}-{target_day:02d}T{target_hour:02d}"))
        )

        # Select final schema
        final_df = export_df.select(
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "fare_amount",
            "pickup_zip",
            "dropoff_zip",
            "export_timestamp",
            "batch_id",
            "source_system",
            "export_hour",
        )

        return final_df

    def export_batch(self, spark: SparkSession) -> str:
        """
        Export a single batch of data to the volume.

        Args:
            spark: SparkSession instance

        Returns:
            Path to the exported batch
        """
        # Get data for export
        export_df = self.get_previous_hour_data(spark)

        # Generate unique batch filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"taxi_batch_{timestamp}_{uuid.uuid4().hex[:8]}.parquet"
        batch_path = f"{self.volume_path}/{batch_filename}"

        # Write to volume as parquet
        print(f"Writing batch to: {batch_path}")
        export_df.write.mode("overwrite").parquet(batch_path)

        # Log export statistics
        record_count = export_df.count()
        print(f"Exported {record_count} records to {batch_path}")

        return batch_path

    def continuous_export(self, spark: SparkSession, interval_minutes: int = 5):
        """
        Run continuous export for testing purposes.
        In production, this would be scheduled as a Databricks job.

        Args:
            spark: SparkSession instance
            interval_minutes: Minutes between exports
        """
        import time

        print(f"Starting continuous export every {interval_minutes} minutes...")
        print(f"Target volume: {self.volume_path}")

        try:
            while True:
                try:
                    batch_path = self.export_batch(spark)
                    print(f"✓ Batch exported: {batch_path}")
                except Exception as e:
                    print(f"✗ Export failed: {str(e)}")

                print(f"Waiting {interval_minutes} minutes until next export...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("Export stopped by user")


# COMMAND ----------
# Databricks job entry points


def export_single_batch():
    """Entry point for single batch export (for scheduled jobs)"""
    spark = SparkSession.builder.getOrCreate()
    exporter = DataExporter()
    return exporter.export_batch(spark)


def export_continuous():
    """Entry point for continuous export (for testing)"""
    spark = SparkSession.builder.getOrCreate()
    exporter = DataExporter()
    exporter.continuous_export(spark, interval_minutes=5)


# COMMAND ----------
# Main execution
if __name__ == "__main__":
    # For testing - run a single export
    result = export_single_batch()
    print(f"Single batch export completed: {result}")
