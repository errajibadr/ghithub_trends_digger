# COMMAND ----------
import dlt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import current_timestamp, lit

from .date_mapper import DateMapper


class StreamingSimulator:
    """
    Simulates streaming taxi data by continuously generating
    data based on historical patterns mapped to current time.
    """

    def __init__(self):
        self.date_mapper = DateMapper()

    def create_landing_zone_data(self, spark: SparkSession) -> DataFrame:
        """
        Create data for the landing zone by simulating current hour data.
        This represents the "raw" data stream from external sources.

        Args:
            spark: SparkSession instance

        Returns:
            DataFrame ready for landing zone ingestion
        """
        # Get simulated data for current hour
        simulated_df = self.date_mapper.simulate_current_hour_data(spark)

        # Add metadata columns for landing zone
        landing_df = (
            simulated_df.withColumn("ingestion_timestamp", current_timestamp())
            .withColumn("source_system", lit("nyc_taxi_simulation"))
            .withColumn("data_quality_flag", lit("pending_validation"))
        )

        return landing_df

    def validate_schema(self, df: DataFrame) -> DataFrame:
        """
        Validate and ensure correct schema for NYC taxi data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with validated schema
        """
        required_columns = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "fare_amount",
            "pickup_zip",
            "dropoff_zip",
        ]

        # Ensure all required columns exist
        for col_name in required_columns:
            if col_name not in df.columns:
                raise ValueError(f"Missing required column: {col_name}")

        return df.select(*required_columns, "ingestion_timestamp", "source_system", "data_quality_flag")


# COMMAND ----------
# DLT-compatible functions for pipeline integration


def get_streaming_simulator():
    """Factory function to create StreamingSimulator instance"""
    return StreamingSimulator()


def get_spark() -> SparkSession:
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        return SparkSession.builder.getOrCreate()


def simulate_taxi_stream():
    """
    Generate simulated taxi data stream for DLT pipeline consumption.
    This function is designed to be used within DLT table definitions.
    """
    simulator = get_streaming_simulator()
    return simulator.create_landing_zone_data(get_spark())
