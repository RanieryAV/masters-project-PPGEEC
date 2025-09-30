import os
import traceback
import logging
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from os import path
from dotenv import load_dotenv 

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql.window import Window

preprocess_data_bp = Blueprint('process_data_bp', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_logger(name=__name__):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(fmt)
        log.addHandler(ch)
    return log

logger = get_logger()

# Load environment variables#move to services
load_dotenv()#move to services

class Process_Data_Service:
    @staticmethod
    def adjust_file_permissions(output_path: str):
        """Recursively set file permissions for all files in the given directory."""
        for root_dir, dirs, files in os.walk(output_path):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root_dir, d), 0o777)
                except PermissionError:
                    logger.warning(f"chmod failed on {os.path.join(root_dir, d)}")
            for f in files:
                try:
                    os.chmod(os.path.join(root_dir, f), 0o777)
                except PermissionError:
                    logger.warning(f"chmod failed on {os.path.join(root_dir, f)}")

        # After saving, set permissions to 777 for all files in the output directory
        for root, dirs, files in os.walk(output_path):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o777)

    @staticmethod
    def load_spark_labels_df_from_Pitsikalis_2019_csv(spark, relative_path: str, expected_header: list):
        """Load CSV via Spark, infer schema, rename columns if header matches expected rows."""
        events_path = relative_path  # relative_path is already joined
        logger.info(f"Loading CSV from path: {events_path}")

        df = (
            spark.read
            .option("header", True)
            .option("sep", "|")
            .option("inferSchema", True)
            .csv(events_path)
        )

        if len(df.columns) == len(expected_header):
            df = df.toDF(*expected_header)
        else:
            logger.info(f"Read {len(df.columns)} columns, expected {len(expected_header)}; keeping original schema: {df.columns}")

        return df

    @staticmethod
    def filter_and_transform_Pitsikalis_2019_labels_data(df):
        fluent_name_categories = {
            'highSpeedNC': 'NORMAL',
            'loitering': 'LOITERING',
            'tuggingSpeed': 'LOITERING',
            'stopped': 'STOPPING',
            'anchoredOrMoored': 'STOPPING',
            'rendezVous': 'TRANSSHIPMENT',
        }

        df_filtered = df.filter(F.col("FluentName").isin(list(fluent_name_categories.keys())))
        
        df_transformed = (
            df_filtered
            .withColumn("T_start", F.to_timestamp(F.col("T_start").cast("long")))
            .withColumn("T_end", F.to_timestamp(F.col("T_end").cast("long")))
            .withColumn("Category", F.create_map(
                *sum(([F.lit(k), F.lit(v)] for k, v in fluent_name_categories.items()), [])
            ).getItem(F.col("FluentName")))
        )

        # Correct windowing for row numbers
        window_spec = Window.orderBy("EventIndexLong")
        df_with_index = (
            df_transformed
            .withColumn("EventIndexLong", F.monotonically_increasing_id())
            .withColumn(
                "EventIndex",
                F.row_number().over(Window.orderBy("EventIndexLong"))
            )
            .drop("EventIndexLong")
        )

        return df_with_index

    @staticmethod
    def inspect_spark_labels_dataframe_Pitsikalis_2019(df):
        """Run EDA / summary tasks on the DataFrame, print + log results, and return summary metadata."""
        results = {}

        logger.info("Showing sample T_start / T_end values")
        df.select("T_start").show(5)
        df.select("T_end").show(5)

        print("First 5 rows of the data")
        df.show(5)

        total_count = df.count()
        logger.info(f"Total rows: {total_count}")

        print("Last 5 rows of the data")
        if total_count > 5:
            df.orderBy(F.desc("T_end")).show(5)
        else:
            df.show(total_count)

        print("Shape of the data")
        shape = (total_count, len(df.columns))
        print(shape)
        results['num_rows'] = total_count
        results['num_columns'] = len(df.columns)

        print("Data types of each column")
        dtypes = {field.name: field.dataType for field in df.schema.fields}
        print(dtypes)
        results['dtypes'] = dtypes

        print("Column names")
        cols = df.columns
        print(cols)
        results['columns'] = cols

        print("Number of missing values in each column")
        null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
        null_counts.show()

        print("Number of NaN (or Null) in numeric/double columns")
        nan_null_exprs = []
        for field in df.schema.fields:
            if isinstance(field.dataType, DoubleType) or field.dataType.simpleString().startswith("double") or field.dataType.simpleString().startswith("float"):
                nan_null_exprs.append(
                    F.count(F.when(F.col(field.name).isNull() | F.isnan(F.col(field.name)), field.name)).alias(field.name)
                )
        if nan_null_exprs:
            df.select(nan_null_exprs).show()
        else:
            logger.info("No numeric double/float fields to check NaN")

        print("Number of unique values in each column")
        unique_counts = {}
        for c in df.columns:
            cnt = df.select(c).distinct().count()
            unique_counts[c] = cnt
            print(f"{c}: {cnt}")
        results['unique_counts'] = unique_counts

        print("Summary statistics of the data")
        df.describe().show()

        print("Full summary (including percentiles etc)")
        df.summary().show()

        return results

    