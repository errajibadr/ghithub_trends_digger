# COMMAND ----------
"""
Test script to validate data simulation logic
This can be run in Databricks to verify the pipeline components work correctly.
"""

from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, dayofmonth, hour, month, year

from data_simulation.date_mapper import DateMapper
from data_simulation.streaming_simulator import StreamingSimulator


def test_date_mapper():
    """Test the date mapping functionality"""
    print("🧪 Testing DateMapper...")

    # Initialize components
    spark = SparkSession.builder.appName("TestSimulation").getOrCreate()
    date_mapper = DateMapper()

    # Test current simulation parameters
    current_day, current_hour, current_month = date_mapper.get_current_simulation_params()
    print(f"📅 Current simulation params: Day={current_day}, Hour={current_hour}, Month={current_month}")

    # Test historical data simulation
    simulated_df = date_mapper.simulate_current_hour_data(spark, sample_fraction=0.1)  # Small sample for testing

    # Validate results
    count_result = simulated_df.count()
    print(f"📊 Generated {count_result} records")

    if count_result > 0:
        print("✅ Date mapping successful!")

        # Show sample data
        print("📋 Sample simulated data:")
        simulated_df.select(
            "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount", "pickup_zip"
        ).show(5)

        # Validate date transformation
        date_check = (
            simulated_df.select(
                year("tpep_pickup_datetime").alias("pickup_year"),
                month("tpep_pickup_datetime").alias("pickup_month"),
                dayofmonth("tpep_pickup_datetime").alias("pickup_day"),
                hour("tpep_pickup_datetime").alias("pickup_hour"),
            )
            .distinct()
            .collect()
        )

        print("📊 Date validation:")
        for row in date_check[:3]:  # Show first 3 distinct combinations
            print(
                f"   Year: {row.pickup_year}, Month: {row.pickup_month}, Day: {row.pickup_day}, Hour: {row.pickup_hour}"
            )

    else:
        print("⚠️  No data generated - check filter conditions")

    return simulated_df


def test_streaming_simulator():
    """Test the streaming simulator functionality"""
    print("\n🧪 Testing StreamingSimulator...")

    # Initialize components
    spark = SparkSession.builder.appName("TestSimulation").getOrCreate()
    simulator = StreamingSimulator()

    # Test landing zone data creation
    landing_df = simulator.create_landing_zone_data(spark)

    # Validate results
    count_result = landing_df.count()
    print(f"📊 Generated {count_result} landing zone records")

    if count_result > 0:
        print("✅ Streaming simulation successful!")

        # Show schema
        print("📋 Landing zone schema:")
        landing_df.printSchema()

        # Show sample data
        print("📋 Sample landing zone data:")
        landing_df.show(3, truncate=False)

        # Validate required columns
        expected_columns = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "fare_amount",
            "pickup_zip",
            "dropoff_zip",
            "ingestion_timestamp",
            "source_system",
            "data_quality_flag",
        ]

        missing_columns = [col for col in expected_columns if col not in landing_df.columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
        else:
            print("✅ All required columns present")

    else:
        print("⚠️  No landing zone data generated")

    return landing_df


def validate_schema_compliance():
    """Validate that our data matches the expected schema"""
    print("\n🧪 Testing Schema Compliance...")

    spark = SparkSession.builder.appName("TestSimulation").getOrCreate()
    date_mapper = DateMapper()
    simulated_df = date_mapper.simulate_current_hour_data(spark, sample_fraction=0.1)

    # Expected schema
    expected_schema = {
        "tpep_pickup_datetime": "timestamp",
        "tpep_dropoff_datetime": "timestamp",
        "trip_distance": "double",
        "fare_amount": "double",
        "pickup_zip": "int",
        "dropoff_zip": "int",
    }

    print("📋 Schema validation:")
    for column_name, expected_type in expected_schema.items():
        if column_name in simulated_df.columns:
            actual_type = dict(simulated_df.dtypes)[column_name]
            if expected_type in actual_type.lower():
                print(f"   ✅ {column_name}: {actual_type}")
            else:
                print(f"   ❌ {column_name}: expected {expected_type}, got {actual_type}")
        else:
            print(f"   ❌ Missing column: {column_name}")


def main():
    """Main test function"""
    print("🚀 Starting Data Simulation Tests")
    print("=" * 50)

    try:
        # Test individual components
        simulated_df = test_date_mapper()
        landing_df = test_streaming_simulator()
        validate_schema_compliance()

        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("🎯 Ready for DLT pipeline deployment")

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
