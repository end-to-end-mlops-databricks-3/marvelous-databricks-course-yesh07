# src/mlops_course/preprocessing/hotel_reservation_preprocessor.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

@dataclass
class HotelReservationConfig:
    """Configuration for hotel reservation preprocessing."""
    
    # Input data configs
    input_path: str
    
    # Output data configs
    output_catalog: str = "mlops_dev"
    output_schema: str = "your_schema"  # Replace with your assigned schema
    output_table: str = "hotel_reservations_processed"
    
    # Features to keep
    categorical_features: List[str] = None
    numerical_features: List[str] = None
    target_column: str = "is_canceled"
    
    # Preprocessing configs
    handle_missing: bool = True
    create_features: bool = True
    
    def __post_init__(self):
        """Set default features if none provided."""
        if self.categorical_features is None:
            self.categorical_features = [
                "type_of_meal_plan",
                "room_type_reserved",
                "market_segment_type",
                "booking_status",
            ]
            
        if self.numerical_features is None:
            self.numerical_features = [
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


class HotelReservationPreprocessor:
    """Preprocess hotel reservation data for ML modeling."""
    
    def __init__(self, config: HotelReservationConfig, spark: Optional[SparkSession] = None):
        """Initialize the preprocessor.
        
        Args:
            config: Configuration for preprocessing
            spark: SparkSession (optional, will be created if not provided)
        """
        self.config = config
        self.spark = spark or SparkSession.builder.getOrCreate()
        logger.info(f"Initialized HotelReservationPreprocessor with config: {config}")
        
    def load_data(self) -> SparkDataFrame:
        """Load hotel reservation data from the specified path.
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from: {self.config.input_path}")
        
        # Determine file format based on extension
        if self.config.input_path.endswith(".csv"):
            df = self.spark.read.csv(self.config.input_path, header=True, inferSchema=True)
        elif self.config.input_path.endswith(".parquet"):
            df = self.spark.read.parquet(self.config.input_path)
        else:
            raise ValueError(f"Unsupported file format for {self.config.input_path}")
            
        logger.info(f"Loaded data with {df.count()} rows and {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df: SparkDataFrame) -> SparkDataFrame:
        """Handle missing values in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        if not self.config.handle_missing:
            logger.info("Skipping missing value handling")
            return df
            
        logger.info("Handling missing values")
        
        # Get column counts before filling
        before_counts = df.count()
        
        # Fill missing numerical values with mean
        numeric_cols = [col for col in df.columns if col in self.config.numerical_features]
        for col in numeric_cols:
            mean_value = df.select(F.mean(F.col(col))).collect()[0][0]
            df = df.withColumn(col, F.coalesce(F.col(col), F.lit(mean_value)))
        
        # Fill missing categorical values with mode
        categorical_cols = [col for col in df.columns if col in self.config.categorical_features]
        for col in categorical_cols:
            mode_value = df.groupBy(col).count().orderBy(F.desc("count")).first()[0]
            df = df.withColumn(col, F.coalesce(F.col(col), F.lit(mode_value)))
        
        # Get column counts after filling
        after_counts = df.count()
        logger.info(f"Filled missing values: rows before={before_counts}, after={after_counts}")
        
        return df
    
    def create_engineered_features(self, df: SparkDataFrame) -> SparkDataFrame:
        """Create engineered features from the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        if not self.config.create_features:
            logger.info("Skipping feature engineering")
            return df
            
        logger.info("Creating engineered features")
        
        # Create total_nights feature
        if "no_of_weekend_nights" in df.columns and "no_of_week_nights" in df.columns:
            df = df.withColumn(
                "total_nights", 
                F.col("no_of_weekend_nights") + F.col("no_of_week_nights")
            )
            
        # Create has_children feature
        if "no_of_children" in df.columns:
            df = df.withColumn(
                "has_children", 
                F.when(F.col("no_of_children") > 0, 1).otherwise(0)
            )
            
        # Create avg_price_per_person feature
        if "avg_price_per_room" in df.columns and "no_of_adults" in df.columns:
            df = df.withColumn(
                "avg_price_per_person", 
                F.col("avg_price_per_room") / F.greatest(F.col("no_of_adults"), F.lit(1))
            )
            
        # Create season feature based on arrival month
        if "arrival_month" in df.columns:
            df = df.withColumn(
                "season", 
                F.when((F.col("arrival_month") >= 3) & (F.col("arrival_month") <= 5), "spring")
                .when((F.col("arrival_month") >= 6) & (F.col("arrival_month") <= 8), "summer")
                .when((F.col("arrival_month") >= 9) & (F.col("arrival_month") <= 11), "fall")
                .otherwise("winter")
            )
            
        logger.info(f"Added engineered features. New column count: {len(df.columns)}")
        return df
    
    def convert_categorical_columns(self, df: SparkDataFrame) -> SparkDataFrame:
        """Convert categorical columns to string type for consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with consistent categorical column types
        """
        logger.info("Converting categorical columns to string type")
        
        for col in self.config.categorical_features:
            if col in df.columns:
                df = df.withColumn(col, F.col(col).cast(StringType()))
                
        return df
    
    def select_features(self, df: SparkDataFrame) -> SparkDataFrame:
        """Select relevant features for the final dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with only the selected features
        """
        # Combine all columns to keep
        all_features = self.config.numerical_features + self.config.categorical_features
        
        # Add target column if it exists
        if self.config.target_column in df.columns:
            all_features.append(self.config.target_column)
            
        # Add engineered features if they exist
        engineered_features = ["total_nights", "has_children", "avg_price_per_person", "season"]
        for feat in engineered_features:
            if feat in df.columns:
                all_features.append(feat)
                
        # Get unique column names (in case of duplicates)
        unique_features = list(set(all_features))
        
        # Keep only columns that exist in the DataFrame
        existing_features = [col for col in unique_features if col in df.columns]
        
        logger.info(f"Selecting {len(existing_features)} features from {len(df.columns)} columns")
        return df.select(existing_features)
    
    def preprocess(self) -> SparkDataFrame:
        """Run the full preprocessing pipeline.
        
        Returns:
            Processed DataFrame
        """
        logger.info("Starting preprocessing pipeline")
        
        # Load data
        df = self.load_data()
        
        # Apply preprocessing steps
        df = self.handle_missing_values(df)
        df = self.create_engineered_features(df)
        df = self.convert_categorical_columns(df)
        df = self.select_features(df)
        
        logger.info("Completed preprocessing pipeline")
        return df
    
    def save_to_unity_catalog(self, df: SparkDataFrame) -> None:
        """Save the processed data to Unity Catalog.
        
        Args:
            df: Processed DataFrame to save
        """
        full_table_name = f"{self.config.output_catalog}.{self.config.output_schema}.{self.config.output_table}"
        logger.info(f"Saving processed data to Unity Catalog: {full_table_name}")
        
        # Write to Delta table
        df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)
        
        logger.info(f"Successfully saved data to {full_table_name}")
    
    def run(self) -> SparkDataFrame:
        """Run the entire preprocessing workflow.
        
        Returns:
            Processed DataFrame
        """
        logger.info("Running complete preprocessing workflow")
        
        # Run preprocessing
        processed_df = self.preprocess()
        
        # Save to Unity Catalog
        self.save_to_unity_catalog(processed_df)
        
        logger.info("Preprocessing workflow completed successfully")
        return processed_df