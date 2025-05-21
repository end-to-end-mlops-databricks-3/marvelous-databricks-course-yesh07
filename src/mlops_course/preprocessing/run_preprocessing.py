# src/mlops_course/preprocessing/run_preprocessing.py

import os
import sys
from typing import Dict, Optional

from loguru import logger
from pyspark.sql import SparkSession

from mlops_course.config.config import Config, load_config_from_env
from mlops_course.preprocessing.hotel_reservation_preprocessor import (
    HotelReservationConfig,
    HotelReservationPreprocessor,
)


def create_spark_session(app_name: str, config: Optional[Dict] = None) -> SparkSession:
    """Create a Spark session.
    
    Args:
        app_name: Name of the Spark application
        config: Additional Spark configuration options
        
    Returns:
        Initialized SparkSession
    """
    logger.info(f"Creating Spark session with app name: {app_name}")
    
    # Create builder
    builder = SparkSession.builder.appName(app_name)
    
    # Add config if provided
    if config:
        for key, value in config.items():
            builder = builder.config(key, value)
    
    # Create and return session
    spark = builder.getOrCreate()
    
    # Log Spark configuration
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Spark configuration: {spark.sparkContext.getConf().getAll()}")
    
    return spark


def run_preprocessing(config_path: Optional[str] = None):
    """Run the hotel reservation preprocessing workflow.
    
    Args:
        config_path: Path to config file (optional, will use env vars if not provided)
    """
    # Configure logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("Starting hotel reservation preprocessing workflow")
    
    # Load configuration
    if config_path:
        # TODO: Implement config loading from file
        raise NotImplementedError("Loading config from file not yet implemented")
    else:
        config = load_config_from_env()
    
    logger.info(f"Loaded configuration: {config}")
    
    # Create Spark session
    spark_config = {
        "spark.driver.memory": config.spark.driver_memory,
        "spark.executor.memory": config.spark.executor_memory,
        "spark.executor.cores": str(config.spark.executor_cores),
        "spark.sql.shuffle.partitions": str(config.spark.shuffle_partitions),
    }
    spark = create_spark_session(config.spark.app_name, spark_config)
    
    # Convert to HotelReservationConfig
    hotel_config = HotelReservationConfig(
        input_path=config.data.input_file_path,
        output_catalog=config.data.output_catalog,
        output_schema=config.data.output_schema,
        output_table=config.data.output_table,
        categorical_features=config.preprocessing.categorical_features,
        numerical_features=config.preprocessing.numerical_features,
        target_column=config.preprocessing.target_column,
        handle_missing=config.preprocessing.handle_missing,
        create_features=config.preprocessing.create_features,
    )
    
    # Initialize and run preprocessor
    preprocessor = HotelReservationPreprocessor(config=hotel_config, spark=spark)
    result_df = preprocessor.run()
    
    # Log results
    logger.info(f"Preprocessing completed successfully. Result has {result_df.count()} rows and {len(result_df.columns)} columns")
    logger.info(f"Data saved to {config.data.output_catalog}.{config.data.output_schema}.{config.data.output_table}")
    
    # Stop Spark session
    spark.stop()
    logger.info("Workflow completed. Spark session stopped.")


if __name__ == "__main__":
    # If config file path is provided as argument, use it
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_preprocessing(config_path)