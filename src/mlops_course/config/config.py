# src/mlops_course/config/config.py

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SparkConfig(BaseModel):
    """Spark configuration parameters."""
    
    app_name: str = "HotelReservationPreprocessing"
    # Additional spark settings can be added here as needed
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    executor_cores: int = 2
    shuffle_partitions: int = 8


class DataConfig(BaseModel):
    """Data-related configuration."""
    
    # Input parameters
    input_file_path: str
    input_format: str = "csv"  # csv, parquet, delta
    
    # Output parameters
    output_catalog: str = "mlops_dev"
    output_schema: str = Field(..., description="Your assigned schema name in Unity Catalog")
    output_table: str = "hotel_reservations_processed"


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""
    
    # Feature selections
    categorical_features: List[str] = [
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
        "booking_status",
    ]
    
    numerical_features: List[str] = [
        "no_of_adults", 
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "lead_time",
        "arrival_year", 
        "arrival_month",
        "arrival_date",
        "required_car_parking_space",
        "avg_price_per_room",
        "no_of_special_requests",
    ]
    
    target_column: str = "is_canceled"
    
    # Preprocessing options
    handle_missing: bool = True
    create_features: bool = True
    
    # List of features to engineer
    engineered_features: List[str] = [
        "total_nights",
        "has_children",
        "avg_price_per_person",
        "season",
    ]


class Config(BaseModel):
    """Main configuration class combining all config components."""
    
    spark: SparkConfig = SparkConfig()
    data: DataConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()


def load_config_from_env() -> Config:
    """Load configuration from environment variables."""
    # Required environment variables
    input_file_path = os.environ.get("INPUT_FILE_PATH")
    output_schema = os.environ.get("OUTPUT_SCHEMA")
    
    if not input_file_path:
        raise ValueError("INPUT_FILE_PATH environment variable must be set")
    
    if not output_schema:
        raise ValueError("OUTPUT_SCHEMA environment variable must be set")
    
    # Create data config
    data_config = DataConfig(
        input_file_path=input_file_path,
        output_schema=output_schema,
    )
    
    # Create and return full config
    return Config(data=data_config)


def load_config_from_dict(config_dict: Dict) -> Config:
    """Load configuration from a dictionary."""
    return Config.model_validate(config_dict)