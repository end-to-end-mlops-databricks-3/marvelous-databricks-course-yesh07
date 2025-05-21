# tests/test_hotel_reservation_preprocessor.py

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

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
    """Create a Databricks Connect session for testing."""
    from databricks.connect import DatabricksSession
    
    # This will use the Databricks Connect configuration from your CLI
    return DatabricksSession.builder.getOrCreate()


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
        # Create a preprocessor
        preprocessor = HotelReservationPreprocessor(config=config, spark=spark)
        
        # Convert pandas DataFrame to Spark DataFrame for testing
        df = spark.createDataFrame(sample_data)
        
        # Create a DataFrame with some missing values
        test_data = sample_data.copy()
        test_data.loc[0, "no_of_adults"] = None
        test_data.loc[1, "type_of_meal_plan"] = None
        spark_df_with_nulls = spark.createDataFrame(test_data)
        
        # Process the DataFrame
        result_df = preprocessor.handle_missing_values(spark_df_with_nulls)
        
        # Check that nulls were filled
        assert result_df.filter("no_of_adults IS NULL").count() == 0
        assert result_df.filter("type_of_meal_plan IS NULL").count() == 0

    @patch("mlops_course.preprocessing.hotel_reservation_preprocessor.HotelReservationPreprocessor.load_data")
    def test_create_engineered_features(self, mock_load_data, config, spark, sample_data):
        """Test creation of engineered features."""
        # Create a preprocessor
        preprocessor = HotelReservationPreprocessor(config=config, spark=spark)
        
        # Convert pandas DataFrame to Spark DataFrame for testing
        df = spark.createDataFrame(sample_data)
        
        # Process the DataFrame
        result_df = preprocessor.create_engineered_features(df)
        
        # Check that engineered features were created
        assert "total_nights" in result_df.columns
        assert "has_children" in result_df.columns
        assert "avg_price_per_person" in result_df.columns
        assert "season" in result_df.columns
        
        # Check values
        total_nights = result_df.select("total_nights").collect()
        assert total_nights[0][0] == 3  # 1 weekend + 2 weekday
        assert total_nights[1][0] == 5  # 2 weekend + 3 weekday

    @patch("mlops_course.preprocessing.hotel_reservation_preprocessor.HotelReservationPreprocessor.save_to_unity_catalog")
    @patch("mlops_course.preprocessing.hotel_reservation_preprocessor.HotelReservationPreprocessor.load_data")
    def test_run_pipeline(self, mock_load_data, mock_save, config, spark, sample_data):
        """Test the full preprocessing pipeline."""
        # Create a preprocessor
        preprocessor = HotelReservationPreprocessor(config=config, spark=spark)
        
        # Mock the load_data method to return our test data
        df = spark.createDataFrame(sample_data)
        mock_load_data.return_value = df
        
        # Run the pipeline
        result_df = preprocessor.run()
        
        # Check that the pipeline completed and returned a DataFrame
        assert result_df is not None
        assert mock_save.called