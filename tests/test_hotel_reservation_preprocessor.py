# tests/test_hotel_reservation_preprocessor.py

import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from mlops_course.preprocessing.hotel_reservation_preprocessor import (
    HotelReservationConfig,
    HotelReservationPreprocessor,
)


@pytest.fixture
def sample_data():
    """Create sample hotel reservation data for testing."""
    data = {
        "Booking_ID": ["INN001", "INN002", "INN003", "INN004", "INN005"],
        "no_of_adults": [2, 2, 1, 2, 2],
        "no_of_children": [0, 1, 0, 2, 0],
        "no_of_weekend_nights": [1, 2, 0, 1, 0],
        "no_of_week_nights": [2, 3, 1, 4, 2],
        "type_of_meal_plan": ["Meal Plan 1", "Meal Plan 2", "Meal Plan 1", "Meal Plan 3", "Meal Plan 2"],
        "required_car_parking_space": [0, 1, 0, 0, 1],
        "room_type_reserved": ["Room_Type 1", "Room_Type 2", "Room_Type 1", "Room_Type 4", "Room_Type 2"],
        "lead_time": [85, 65, 30, 98, 45],
        "arrival_year": [2018, 2018, 2018, 2018, 2018],
        "arrival_month": [10, 11, 12, 1, 2],
        "arrival_date": [23, 15, 8, 24, 17],
        "market_segment_type": ["Online", "Offline", "Online", "Online", "Offline"],
        "avg_price_per_room": [99.5, 120.0, 85.0, 105.0, 110.5],
        "no_of_special_requests": [0, 1, 0, 2, 1],
        "booking_status": ["Not_Canceled", "Canceled", "Not_Canceled", "Canceled", "Not_Canceled"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def spark():
    """Create a Spark session for testing—use Databricks Connect if available, otherwise local PySpark."""
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        from pyspark.sql import SparkSession

        return SparkSession.builder.master("local[*]").appName("hotel-preprocessor-test").getOrCreate()


@pytest.fixture
def config():
    """Create a test configuration."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        return HotelReservationConfig(
            input_path=f.name,
            output_catalog="test_catalog",
            output_schema="test_schema",
            output_table="test_table",
        )


class TestHotelReservationPreprocessor:
    """Test the HotelReservationPreprocessor class."""

    def test_initialization(self, config, spark):
        """Test that the preprocessor initializes correctly."""
        preprocessor = HotelReservationPreprocessor(config=config, spark=spark)
        assert preprocessor.config == config
        assert preprocessor.spark == spark

    @patch("mlops_course.preprocessing.hotel_reservation_preprocessor.HotelReservationPreprocessor.load_data")
    def test_handle_missing_values(self, mock_load_data, config, spark, sample_data):
        """Test handling of missing values."""
        preprocessor = HotelReservationPreprocessor(config=config, spark=spark)
        df = spark.createDataFrame(sample_data)

        test_data = sample_data.copy()
        test_data.loc[0, "no_of_adults"] = None
        test_data.loc[1, "type_of_meal_plan"] = None
        spark_df_with_nulls = spark.createDataFrame(test_data)

        result_df = preprocessor.handle_missing_values(spark_df_with_nulls)

        assert result_df.filter("no_of_adults IS NULL").count() == 0
        assert result_df.filter("type_of_meal_plan IS NULL").count() == 0

    @patch("mlops_course.preprocessing.hotel_reservation_preprocessor.HotelReservationPreprocessor.load_data")
    def test_create_engineered_features(self, mock_load_data, config, spark, sample_data):
        """Test creation of engineered features."""
        preprocessor = HotelReservationPreprocessor(config=config, spark=spark)
        df = spark.createDataFrame(sample_data)

        result_df = preprocessor.create_engineered_features(df)

        assert "total_nights" in result_df.columns
        assert "has_children" in result_df.columns
        assert "avg_price_per_person" in result_df.columns
        assert "season" in result_df.columns

        total_nights = [row[0] for row in result_df.select("total_nights").collect()]
        assert total_nights[0] == 3  # 1 weekend + 2 weekday
        assert total_nights[1] == 5  # 2 weekend + 3 weekday

    @patch(
        "mlops_course.preprocessing.hotel_reservation_preprocessor.HotelReservationPreprocessor.save_to_unity_catalog"
    )
    @patch("mlops_course.preprocessing.hotel_reservation_preprocessor.HotelReservationPreprocessor.load_data")
    def test_run_pipeline(self, mock_load_data, mock_save, config, spark, sample_data):
        """Test the full preprocessing pipeline."""
        preprocessor = HotelReservationPreprocessor(config=config, spark=spark)
        df = spark.createDataFrame(sample_data)
        mock_load_data.return_value = df

        result_df = preprocessor.run()

        assert result_df is not None
        assert mock_save.called
