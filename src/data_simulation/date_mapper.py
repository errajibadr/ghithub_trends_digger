# COMMAND ----------
from datetime import datetime
from typing import Tuple

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofmonth, expr, hour, month, when, year
from pyspark.sql.types import TimestampType


class DateMapper:
    """
    Maps historical NYC taxi data dates (Jan/Feb 2016) to current dates
    for realistic streaming simulation.
    """

    def __init__(self):
        self.source_year = 2016
        self.source_months = [1, 2]  # January and February 2016

    def get_current_simulation_params(self) -> Tuple[int, int, int]:
        """
        Get current date parameters for simulation mapping.
        Returns: (current_day, current_hour, current_month)
        """
        now = datetime.now()
        return now.day, now.hour, now.month

    def get_historical_filter_condition(self, current_day: int, current_hour: int):
        """
        Create filter condition to select data from Jan/Feb 2016
        that matches current day and hour.

        Args:
            current_day: Current day of month (1-31)
            current_hour: Current hour (0-23)

        Returns:
            PySpark filter condition
        """
        return (
            (
                (month(col("tpep_pickup_datetime")) == 1)  # January
                | (month(col("tpep_pickup_datetime")) == 2)  # February
            )
            & (dayofmonth(col("tpep_pickup_datetime")) == current_day)
            & (hour(col("tpep_pickup_datetime")) == current_hour)
            & (year(col("tpep_pickup_datetime")) == self.source_year)
        )

    def transform_dates_to_current(self, df, current_year: int, current_month: int):
        """
        Transform 2016 dates to current year/month while preserving day and time.

        Args:
            df: Input DataFrame with timestamp columns
            current_year: Target year for transformation
            current_month: Target month for transformation

        Returns:
            DataFrame with updated timestamps
        """
        # Calculate the year and month difference
        year_diff = current_year - self.source_year

        # Transform pickup datetime
        df_transformed = df.withColumn(
            "tpep_pickup_datetime",
            when(
                month(col("tpep_pickup_datetime")) == 1,
                # January 2016 -> current month
                col("tpep_pickup_datetime")
                + expr(f"interval {year_diff} years")
                + expr(f"interval {current_month - 1} months"),
            )
            .when(
                month(col("tpep_pickup_datetime")) == 2,
                # February 2016 -> current month
                col("tpep_pickup_datetime")
                + expr(f"interval {year_diff} years")
                + expr(f"interval {current_month - 2} months"),
            )
            .otherwise(col("tpep_pickup_datetime")),
        )

        # Transform dropoff datetime
        df_transformed = df_transformed.withColumn(
            "tpep_dropoff_datetime",
            when(
                month(col("tpep_dropoff_datetime")) == 1,
                col("tpep_dropoff_datetime")
                + expr(f"interval {year_diff} years")
                + expr(f"interval {current_month - 1} months"),
            )
            .when(
                month(col("tpep_dropoff_datetime")) == 2,
                col("tpep_dropoff_datetime")
                + expr(f"interval {year_diff} years")
                + expr(f"interval {current_month - 2} months"),
            )
            .otherwise(col("tpep_dropoff_datetime")),
        )

        return df_transformed

    def simulate_current_hour_data(self, spark: SparkSession, sample_fraction: float = 0.75):
        """
        Simulate current hour data by selecting and transforming historical data.

        Args:
            spark: SparkSession instance
            sample_fraction: Fraction of data to sample (default 0.75)

        Returns:
            DataFrame with simulated current data
        """
        # Get current simulation parameters
        current_day, current_hour, current_month = self.get_current_simulation_params()
        current_year = datetime.now().year

        # Read source data
        source_df = spark.read.table("samples.nyctaxi.trips")

        # Filter for matching day and hour from Jan/Feb 2016
        filtered_df = source_df.filter(self.get_historical_filter_condition(current_day, current_hour))

        # Sample the data (75% by default)
        sampled_df = filtered_df.sample(fraction=sample_fraction, seed=42)

        # Transform dates to current year/month
        transformed_df = self.transform_dates_to_current(sampled_df, current_year, current_month)

        # Ensure schema compliance
        final_df = transformed_df.select(
            col("tpep_pickup_datetime").cast(TimestampType()),
            col("tpep_dropoff_datetime").cast(TimestampType()),
            col("trip_distance").cast("double"),
            col("fare_amount").cast("double"),
            col("pickup_zip").cast("int"),
            col("dropoff_zip").cast("int"),
        )

        return final_df


# COMMAND ----------
# Helper function for DLT pipeline usage
def get_date_mapper():
    """Factory function to create DateMapper instance"""
    return DateMapper()
    """Factory function to create DateMapper instance"""
    return DateMapper()
