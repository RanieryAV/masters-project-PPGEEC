import os
import traceback
import logging
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from os import path
from dotenv import load_dotenv
from ..services.process_data_service import ProcessDataService
from domain.services.save_ais_data_service import SaveAISDataService
from domain.config.data_processing.spark_session_initializer import SparkSessionInitializer
import glob

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

# Load environment variables
load_dotenv()

@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_PART_1_select_desired_labels.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-PART-1-select-desired-labels', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_PART_1_select_desired_labels():
    """
    Process the Pitsikalis 2019 labels data (recognized composite events - PART 1: Select desired labels).
    Expects a POST request. Loads data from a predefined CSV path, processes it using Spark,
    and saves the processed data as a new CSV file.
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-data-PART-1-select-desired-labels")

    try:
        spark = SparkSessionInitializer.init_spark_session("Pitsikalis_2019_AIS_data_PART_1_select_desired_labels_[Data_Processing_API]")

        expected_header = ["FluentName", "MMSI", "Argument", "Value", "T_start", "T_end"]
        
        is_container = ProcessDataService._is_running_in_container()  # log if in container or not

        # Read Spark master URL and event log dir from environment variables or use defaults
        if is_container:
            csv_path = "/app/datasets/Maritime_Composite_Events/CEs/recognised_CEs.csv"
        else:
            csv_path = "shared/utils/datasets/Maritime_Composite_Events/CEs/recognised_CEs.csv"

        logger.info(f"WARNING: Loading raw labels data from {csv_path}")

        df = ProcessDataService.load_spark_labels_df_from_Pitsikalis_2019_csv(spark, csv_path, expected_header)

        logger.info("WARNING: Filtering and transforming labels data...")

        df_processed = ProcessDataService.filter_and_transform_Pitsikalis_2019_labels_data(df)

        summary = ProcessDataService.inspect_spark_labels_dataframe_Pitsikalis_2019(df_processed)

        # Save the processed dataframe as CSV
        # processed output dir from env var or default
        if is_container:
            output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        new_file_name = os.getenv("OUTPUT_FOLDER_NAME_FOR_LABELS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        output_path = os.path.join(output_dir, new_file_name)
        
        # Save processed Spark DataFrame as CSV
        ProcessDataService.save_spark_df_as_csv(df_processed, output_path, spark)

        logger.info(f"WARNING: Sorted subset of miscellaneous event labels data saved to '{output_path}'")

        logger.info("Task finished. Stopping Spark session...")

        spark.stop()

        response = {
            "status": "success",
            "num_rows": summary.get('num_rows'),
            "num_columns": summary.get('num_columns'),
            "columns": summary.get('columns'),
            "unique_counts": summary.get('unique_counts'),
            "output_path": output_path
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error("Error in process_Pitsikalis_2019_data", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500

@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_PART_2_pre_process_the_previously_selected_labels.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-PART-2-pre-process-the-previously-selected-labels', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_PART_2_pre_process_the_previously_selected_labels():
    """
    Process the Pitsikalis 2019 AIS data (PART 2 - Pre-process the previously selected labels).
    Expects a POST request. Loads data from a predefined CSV path ("Pitsikalis_2019_filtered_fluentname_data_v2"),
    processes it using Spark, and saves the processed data as new CSV files: one for transshipment events
    ("ais_transshipment_events_Pitsikalis_2019"), other for loitering events ("ais_loitering_events_Pitsikalis_2019"),
    other for normal events ("ais_normal_events_Pitsikalis_2019"), and another for stopping events ("ais_stopping_events_Pitsikalis_2019").
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-data-PART-2-pre-process-the-previously-selected-labels")

    try:
        spark = SparkSessionInitializer.init_spark_session("Pitsikalis_2019_AIS_data_PART_2_pre_process_the_previously_selected_labels_[Data_Processing_API]")

        is_container = ProcessDataService._is_running_in_container()  # log if in container or not
        
        # input dir from env var or default
        if is_container:
            input_dir = "/app/processed_output"
        else:
            input_dir = "shared/utils/processed_output"

        # Build paths to the CSV files
        events_dir = os.path.join(input_dir, "Pitsikalis_2019_filtered_fluentname_data_v2")
        csv_files = glob.glob(os.path.join(events_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {events_dir}")
        events_csv_path = csv_files[0]

        if is_container:
            ais_csv_path = "/app/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"
        else:
            ais_csv_path = "shared/utils/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"

        logger.info(f"Loading processed miscellaneous event labels from {events_csv_path}")
        df_events = ProcessDataService.load_events_Pitsikalis_2019(spark, events_csv_path)

        logger.info("Splitting events into transshipment and non-transshipment")
        transshipment_df, loitering_df, normal_df, stopping_df = ProcessDataService.split_events_Pitsikalis_2019(df_events)

        logger.info("Collecting relevant transshipment MMSI pairs")
        relevant_transship_mmsi = ProcessDataService.get_relevant_transship_mmsi_Pitsikalis_2019(transshipment_df)

        # Save the processed dataframe as CSV
        # processed output dir from env var or default
        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        new_file_name_1 = os.getenv("OUTPUT_FOLDER_NAME_FOR_TRANSSHIP_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        transshipment_output_path = os.path.join(base_output_dir, new_file_name_1)

        logger.info("Saving transshipment events (coalesced single CSV)")
        ProcessDataService.save_spark_df_as_csv(transshipment_df, transshipment_output_path, spark)
        logger.info(f"Transshipment events saved to '{transshipment_output_path}'")

        logger.info("WARNING: Transshipment processing done. Now processing non-transshipment AIS data...")
        logger.info("Processing AIS and joining with non-transshipment events...")
        aggregated_loitering_df, aggregated_normal_df, aggregated_stopping_df = ProcessDataService.process_ais_events_Pitsikalis_2019(spark, ais_csv_path, loitering_df, normal_df, stopping_df)
        
        # Write to an output directory (coalesce to 1)
        new_file_name_2 = os.getenv("OUTPUT_FOLDER_NAME_FOR_LOITERING_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        new_file_name_3 = os.getenv("OUTPUT_FOLDER_NAME_FOR_NORMAL_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        new_file_name_4 = os.getenv("OUTPUT_FOLDER_NAME_FOR_STOPPING_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")

        loitering_output_path = os.path.join(base_output_dir, new_file_name_2)
        normal_output_path = os.path.join(base_output_dir, new_file_name_3)
        stopping_output_path = os.path.join(base_output_dir, new_file_name_4)

        # Call helper function (it coalesces and then promotes)
        ProcessDataService.save_spark_df_as_csv(aggregated_loitering_df, loitering_output_path, spark, allow_multiple_files=False, original_filename=new_file_name_2)
        logger.info(f"Loitering events saved to '{loitering_output_path}'")

        ProcessDataService.save_spark_df_as_csv(aggregated_normal_df, normal_output_path, spark, allow_multiple_files=False, original_filename=new_file_name_3)
        logger.info(f"Normal events saved to '{normal_output_path}'")

        ProcessDataService.save_spark_df_as_csv(aggregated_stopping_df, stopping_output_path, spark, allow_multiple_files=False, original_filename=new_file_name_4)
        logger.info(f"Stopping events saved to '{stopping_output_path}'")

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "Transshipment events processed and saved.",
            "relevant_transship_mmsi": relevant_transship_mmsi
        }), 200

    except Exception as e:
        logger.error("Error loading or processing events data", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500
    
@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_PART_3_AGG_TRANSSHIPMENT.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-PART-3-AGG-TRANSSHIPMENT', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_PART_3_AGG_TRANSSHIPMENT():
    """
    Processes the Pitsikalis 2019 AIS data (PART 3) for *TRANSSHIPMENT* events.

    This endpoint expects a POST request and performs the following steps:
    1. Initializes a Spark session with conservative runtime tuning for stability.
    2. Determines the Spark parallelism level robustly, using environment variables as fallback.
    3. Selects input and output directories based on whether the code is running in a container.
    4. Loads preprocessed transshipment event labels from a CSV file.
    5. Cross-references these labels with raw AIS data using Spark to create an aggregated DataFrame.
    6. Saves the processed, aggregated transshipment event data to a specified output directory.
    7. Stops the Spark session and returns a success response with the output path.

    Features:
    - Aggressive repartitioning and reduced chunk size for efficient Spark writing.
    - Driver disk-space pre-check and robust logging.
    - Graceful fallback if helper functions or environment variables are missing.

    Returns:
        JSON response containing status, message, and output path for the aggregated transshipment events.

    Raises:
        Returns a JSON error response with traceback if any step fails.
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-data-PART-3-AGG-TRANSSHIPMENT")

    try:
        # === Create Spark session (unchanged init_spark_session is used) ===
        spark = SparkSessionInitializer.init_spark_session("Pitsikalis_2019_AIS_PART_3_AGG_TRANSSHIPMENT_[Data_Processing_API]")

        # === Conservative Spark runtime tuning (keeps the safe settings you used earlier) ===
        try:
            spark.conf.set("spark.reducer.maxSizeInFlight", "8m")
            spark.conf.set("spark.shuffle.io.maxRetries", "8")
            spark.conf.set("spark.shuffle.io.retryWait", "5s")
            spark.conf.set("spark.network.timeout", "300s")
            spark.conf.set("spark.executor.heartbeatInterval", "150s")
            spark.conf.set("spark.executor.memoryOverhead", "1024")
            spark.conf.set("spark.shuffle.compress", "true")
            spark.conf.set("spark.shuffle.spill.compress", "true")
            logger.info("Applied conservative spark.conf tuning for reducer/fetch/timeout/memoryOverhead")
        except Exception as _e:
            logger.warning("Could not set some spark.conf tuning values: %s", _e)

        # === Determine default_parallelism robustly ===
        try:
            default_parallelism = spark.sparkContext.defaultParallelism
            logger.info("Detected spark.sparkContext.defaultParallelism = %s", default_parallelism)
        except Exception:
            default_parallelism = None
            logger.info("Could not read spark.sparkContext.defaultParallelism")

        if not default_parallelism:
            try:
                cores_max = int(os.getenv("SPARK_CORES_MAX", "4"))
                exec_cores = int(os.getenv("SPARK_EXECUTOR_CORES", "2"))
                inferred_executors = max(1, cores_max // max(1, exec_cores))
                default_parallelism = inferred_executors * exec_cores
                logger.info(
                    "Inferred default_parallelism from env: cores_max=%s exec_cores=%s -> default_parallelism=%s",
                    cores_max, exec_cores, default_parallelism
                )
            except Exception:
                default_parallelism = 8
                logger.info("Falling back to default_parallelism=%s", default_parallelism)

        # === Paths / environment choices ===
        is_container = ProcessDataService._is_running_in_container()

        if is_container:
            input_dir = "/app/processed_output"
        else:
            input_dir = "shared/utils/processed_output"

        if is_container:
            ais_csv_path = "/app/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"
        else:
            ais_csv_path = "shared/utils/datasets/ais_brest_locations.csv"

        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        # === Transshipment branch (unchanged behavior) ===
        preprocessed_transshipment_AIS_dir = os.path.join(input_dir, "ais_transshipment_events_Pitsikalis_2019")
        csv_files = glob.glob(os.path.join(preprocessed_transshipment_AIS_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {preprocessed_transshipment_AIS_dir}")
        preprocessed_transshipment_AIS_csv_path = csv_files[0]

        logger.info(f"Loading pre-processed transshipment events labels from {preprocessed_transshipment_AIS_csv_path}")

        ais_spark_df = spark.read.option("header", True).option("inferSchema", True).csv(preprocessed_transshipment_AIS_csv_path)
        ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
        result_df = ProcessDataService.create_aggregated_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
            ais_spark_df, ais_csv_path, spark
        )

        new_file_name_1 = os.getenv(
            "OUTPUT_FOLDER_NAME_FOR_AGG_TRANSSHIP_UNIQUE_VESSEL_DATA_PITSIKALIS_2019_WITH_EXTRA_FEATURES",
            "Placeholder_folder_data_processed_by_spark"
        )
        aggregated_transshipment_output_path = os.path.join(base_output_dir, new_file_name_1)

        # existing saver (coalesces + promotes) - leave as-is since it works for transshipment
        ProcessDataService.save_spark_df_as_csv(result_df, aggregated_transshipment_output_path, spark)
        logger.info(f"AGGREGATED Transshipment events saved to '{aggregated_transshipment_output_path}'")

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "Aggregated Transshipment events processed and saved.",
            "output_path_transshipment": aggregated_transshipment_output_path
        }), 200

    except Exception as e:
        logger.error("Error loading or processing events data", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500
    
@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_PART_4_AGG_NORMAL.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-PART-4-AGG-NORMAL', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_PART_4_AGG_NORMAL():
    """
    Processes the Pitsikalis 2019 AIS data (PART 4) for *NORMAL* vessel events.
    This endpoint expects a POST request and performs the following steps:
    1. Initializes a Spark session with conservative runtime tuning for reliability.
    2. Determines Spark parallelism settings robustly, using environment variables as fallback.
    3. Selects input/output directories based on containerized or local environment.
    4. Loads preprocessed normal event AIS data from CSV files.
    5. Cross-references preprocessed event labels with raw AIS data using Spark, aggregating vessel-level features.
    6. Saves the processed, aggregated data to a specified output directory, using hash partitioning and chunked streaming for efficient writing.
    7. Logs all major steps and configuration choices, gracefully handling missing helpers or configuration issues.
    8. Stops the Spark session and returns a JSON response indicating success and the output path.

    Returns:
        Flask Response: JSON object with status, message, and output path for the aggregated normal events data.
        On error, returns JSON with error details and traceback.
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-data-PART-4-AGG-NORMAL")

    try:
        # === Create Spark session (unchanged init_spark_session is used) ===
        spark = SparkSessionInitializer.init_spark_session("Pitsikalis_2019_AIS_PART_4_AGG_NORMAL_[Data_Processing_API]")
        # === Conservative Spark runtime tuning (keeps the safe settings you used earlier) ===
        try:
            spark.conf.set("spark.reducer.maxSizeInFlight", "8m")
            spark.conf.set("spark.shuffle.io.maxRetries", "8")
            spark.conf.set("spark.shuffle.io.retryWait", "5s")
            spark.conf.set("spark.network.timeout", "300s")
            spark.conf.set("spark.executor.heartbeatInterval", "150s")
            spark.conf.set("spark.executor.memoryOverhead", "1024")
            spark.conf.set("spark.shuffle.compress", "true")
            spark.conf.set("spark.shuffle.spill.compress", "true")
            logger.info("Applied conservative spark.conf tuning for reducer/fetch/timeout/memoryOverhead")
        except Exception as _e:
            logger.warning("Could not set some spark.conf tuning values: %s", _e)

        # === Determine default_parallelism robustly ===
        try:
            default_parallelism = spark.sparkContext.defaultParallelism
            logger.info("Detected spark.sparkContext.defaultParallelism = %s", default_parallelism)
        except Exception:
            default_parallelism = None
            logger.info("Could not read spark.sparkContext.defaultParallelism")

        if not default_parallelism:
            try:
                cores_max = int(os.getenv("SPARK_CORES_MAX", "4"))
                exec_cores = int(os.getenv("SPARK_EXECUTOR_CORES", "2"))
                inferred_executors = max(1, cores_max // max(1, exec_cores))
                default_parallelism = inferred_executors * exec_cores
                logger.info(
                    "Inferred default_parallelism from env: cores_max=%s exec_cores=%s -> default_parallelism=%s",
                    cores_max, exec_cores, default_parallelism
                )
            except Exception:
                default_parallelism = 8
                logger.info("Falling back to default_parallelism=%s", default_parallelism)

        # === Paths / environment choices ===
        is_container = ProcessDataService._is_running_in_container()

        if is_container:
            input_dir = "/app/processed_output"
        else:
            input_dir = "shared/utils/processed_output"

        if is_container:
            ais_csv_path = "/app/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"
        else:
            ais_csv_path = "shared/utils/datasets/ais_brest_locations.csv"

        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        # === Non-transshipment branch (where we apply chunking + repartition) ===        
        # Normal events
        preprocessed_normal_AIS_dir = os.path.join(input_dir, "ais_normal_events_Pitsikalis_2019")
        csv_files = glob.glob(os.path.join(preprocessed_normal_AIS_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {preprocessed_normal_AIS_dir}")
        preprocessed_normal_AIS_csv_path = csv_files[0]

        logger.info(f"Loading pre-processed normal events labels from {preprocessed_normal_AIS_csv_path}")

        ais_spark_df = spark.read.option("header", True).option("inferSchema", True).csv(preprocessed_normal_AIS_csv_path)
        ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
        result_df = ProcessDataService.create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
            ais_spark_df, ais_csv_path, spark
        )

        new_file_name_2 = os.getenv(
            "OUTPUT_FOLDER_NAME_FOR_AGG_NORMAL_UNIQUE_VESSEL_DATA_PITSIKALIS_2019_WITH_EXTRA_FEATURES",
            "Placeholder_folder_data_processed_by_spark"
        )
        aggregated_normal_output_path = os.path.join(base_output_dir, new_file_name_2)

        ProcessDataService.save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019(
            spark_df=result_df,
            output_dir=aggregated_normal_output_path,
            spark=spark,
            num_buckets=int(os.getenv("BUCKETS_FOR_LARGE_WRITE", "400")),
            bucket_coalesce=True,
            allow_bucket_fallback_to_chunked=True,
            progress_log_every=int(os.getenv("BUCKET_PROGRESS_LOG_EVERY", "20")),
            sample_for_bucket_size=False,
            max_buckets_to_process=10,
            compress_parts=False
        )

        logger.info(f"AGGREGATED Normal events saved to '{aggregated_normal_output_path}'")

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "Aggregated Normal events processed and saved.",
            "output_path_normal": aggregated_normal_output_path
        }), 200

    except Exception as e:
        logger.error("Error loading or processing events data", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500

@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_PART_5_AGG_STOPPING.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-PART-5-AGG-STOPPING', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_PART_5_AGG_STOPPING():
    """
    Processes the Pitsikalis 2019 AIS *STOPPING* events data (PART 5).
    This function is designed to be triggered by a POST request. It performs the following steps:
        - Initializes a Spark session with conservative runtime tuning for reliability.
        - Determines Spark parallelism settings robustly, using environment variables as fallback.
        - Selects input/output directories based on containerized or local environment.
        - Loads preprocessed stopping event labels from a CSV file.
        - Cross-references these labels with raw AIS data using Spark, aggregating vessel-level features.
        - Saves the processed, aggregated data to a specified output directory, using hash partitioning and chunked streaming for efficient writing.
        - Logs all major steps and configuration choices, with graceful fallback if helpers or settings are missing.
        - Stops the Spark session and returns a JSON response indicating success or failure.

    Returns:
            Flask Response: JSON object with status, message, and output path on success; error details on failure.

    Raises:
            FileNotFoundError: If no preprocessed stopping events CSV is found.
            Exception: For any other errors during processing, with traceback included in the response.
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-data-PART-5-AGG-STOPPING")

    try:
        # === Create Spark session (unchanged init_spark_session is used) ===
        spark = SparkSessionInitializer.init_spark_session("Pitsikalis_2019_AIS_PART_5_AGG_STOPPING_[Data_Processing_API]")
        # === Conservative Spark runtime tuning (keeps the safe settings you used earlier) ===
        try:
            spark.conf.set("spark.reducer.maxSizeInFlight", "8m")
            spark.conf.set("spark.shuffle.io.maxRetries", "8")
            spark.conf.set("spark.shuffle.io.retryWait", "5s")
            spark.conf.set("spark.network.timeout", "300s")
            spark.conf.set("spark.executor.heartbeatInterval", "150s")
            spark.conf.set("spark.executor.memoryOverhead", "1024")
            spark.conf.set("spark.shuffle.compress", "true")
            spark.conf.set("spark.shuffle.spill.compress", "true")
            logger.info("Applied conservative spark.conf tuning for reducer/fetch/timeout/memoryOverhead")
        except Exception as _e:
            logger.warning("Could not set some spark.conf tuning values: %s", _e)

        # === Determine default_parallelism robustly ===
        try:
            default_parallelism = spark.sparkContext.defaultParallelism
            logger.info("Detected spark.sparkContext.defaultParallelism = %s", default_parallelism)
        except Exception:
            default_parallelism = None
            logger.info("Could not read spark.sparkContext.defaultParallelism")

        if not default_parallelism:
            try:
                cores_max = int(os.getenv("SPARK_CORES_MAX", "4"))
                exec_cores = int(os.getenv("SPARK_EXECUTOR_CORES", "2"))
                inferred_executors = max(1, cores_max // max(1, exec_cores))
                default_parallelism = inferred_executors * exec_cores
                logger.info(
                    "Inferred default_parallelism from env: cores_max=%s exec_cores=%s -> default_parallelism=%s",
                    cores_max, exec_cores, default_parallelism
                )
            except Exception:
                default_parallelism = 8
                logger.info("Falling back to default_parallelism=%s", default_parallelism)

        # === Paths / environment choices ===
        is_container = ProcessDataService._is_running_in_container()

        if is_container:
            input_dir = "/app/processed_output"
        else:
            input_dir = "shared/utils/processed_output"

        if is_container:
            ais_csv_path = "/app/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"
        else:
            ais_csv_path = "shared/utils/datasets/ais_brest_locations.csv"

        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        # Stopping events
        preprocessed_stopping_AIS_dir = os.path.join(input_dir, "ais_stopping_events_Pitsikalis_2019")
        csv_files = glob.glob(os.path.join(preprocessed_stopping_AIS_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {preprocessed_stopping_AIS_dir}")
        preprocessed_stopping_AIS_csv_path = csv_files[0]

        logger.info(f"Loading pre-processed stopping events labels from {preprocessed_stopping_AIS_csv_path}")

        ais_spark_df = spark.read.option("header", True).option("inferSchema", True).csv(preprocessed_stopping_AIS_csv_path)
        ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
        result_df = ProcessDataService.create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
            ais_spark_df, ais_csv_path, spark
        )

        new_file_name_3 = os.getenv(
            "OUTPUT_FOLDER_NAME_FOR_AGG_STOPPING_UNIQUE_VESSEL_DATA_PITSIKALIS_2019_WITH_EXTRA_FEATURES",
            "Placeholder_folder_data_processed_by_spark"
        )
        aggregated_stopping_output_path = os.path.join(base_output_dir, new_file_name_3)

        ProcessDataService.save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019(
            spark_df=result_df,
            output_dir=aggregated_stopping_output_path,
            spark=spark,
            num_buckets=int(os.getenv("BUCKETS_FOR_LARGE_WRITE", "2000")),
            bucket_coalesce=True,
            allow_bucket_fallback_to_chunked=True,
            progress_log_every=int(os.getenv("BUCKET_PROGRESS_LOG_EVERY", "20")),
            sample_for_bucket_size=False,
            max_buckets_to_process=10,
            compress_parts=False
        )
        logger.info(f"AGGREGATED Stopping events saved to '{aggregated_stopping_output_path}'")

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "Aggregated Stopping events processed and saved.",
            "output_path_stopping": aggregated_stopping_output_path
        }), 200

    except Exception as e:
        logger.error("Error loading or processing events data", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500

@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_PART_6_AGG_LOITERING.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-PART-6-AGG-LOITERING', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_PART_6_AGG_LOITERING():
    """
    Processes the Pitsikalis 2019 AIS data (PART 6) for *LOITERING* events.

    This function is designed to be triggered by a POST request. It performs the following steps:
        - Initializes a Spark session with conservative runtime tuning for stability.
        - Determines parallelism settings based on Spark context or environment variables.
        - Selects input and output directories based on whether the code is running in a container.
        - Loads preprocessed loitering event labels from a CSV file.
        - Cross-references these labels with raw AIS data using Spark to create an aggregated DataFrame.
        - Saves the processed, aggregated loitering event data to a specified output directory, using hash partitioning and chunked streaming for efficient writing.
        - Logs all major steps and configuration choices, with graceful fallback if helpers or files are missing.
        - Stops the Spark session and returns a JSON response indicating success or error.

    Returns:
            A Flask JSON response containing the status, message, and output path for the aggregated loitering events data.
            On error, returns a JSON response with error details and traceback.
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-data-PART-6-AGG-LOITERING")

    try:
        # === Create Spark session (unchanged init_spark_session is used) ===
        spark = SparkSessionInitializer.init_spark_session("Pitsikalis_2019_AIS_PART_6_AGG_LOITERING_[Data_Processing_API]")
        # === Conservative Spark runtime tuning (keeps the safe settings you used earlier) ===
        try:
            spark.conf.set("spark.reducer.maxSizeInFlight", "8m")
            spark.conf.set("spark.shuffle.io.maxRetries", "8")
            spark.conf.set("spark.shuffle.io.retryWait", "5s")
            spark.conf.set("spark.network.timeout", "300s")
            spark.conf.set("spark.executor.heartbeatInterval", "150s")
            spark.conf.set("spark.executor.memoryOverhead", "1024")
            spark.conf.set("spark.shuffle.compress", "true")
            spark.conf.set("spark.shuffle.spill.compress", "true")
            logger.info("Applied conservative spark.conf tuning for reducer/fetch/timeout/memoryOverhead")
        except Exception as _e:
            logger.warning("Could not set some spark.conf tuning values: %s", _e)

        # === Determine default_parallelism robustly ===
        try:
            default_parallelism = spark.sparkContext.defaultParallelism
            logger.info("Detected spark.sparkContext.defaultParallelism = %s", default_parallelism)
        except Exception:
            default_parallelism = None
            logger.info("Could not read spark.sparkContext.defaultParallelism")

        if not default_parallelism:
            try:
                cores_max = int(os.getenv("SPARK_CORES_MAX", "4"))
                exec_cores = int(os.getenv("SPARK_EXECUTOR_CORES", "2"))
                inferred_executors = max(1, cores_max // max(1, exec_cores))
                default_parallelism = inferred_executors * exec_cores
                logger.info(
                    "Inferred default_parallelism from env: cores_max=%s exec_cores=%s -> default_parallelism=%s",
                    cores_max, exec_cores, default_parallelism
                )
            except Exception:
                default_parallelism = 8
                logger.info("Falling back to default_parallelism=%s", default_parallelism)

        # === Paths / environment choices ===
        is_container = ProcessDataService._is_running_in_container()

        if is_container:
            input_dir = "/app/processed_output"
        else:
            input_dir = "shared/utils/processed_output"

        if is_container:
            ais_csv_path = "/app/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"
        else:
            ais_csv_path = "shared/utils/datasets/ais_brest_locations.csv"

        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        # Loitering events
        preprocessed_loitering_AIS_dir = os.path.join(input_dir, "ais_loitering_events_Pitsikalis_2019")
        csv_files = glob.glob(os.path.join(preprocessed_loitering_AIS_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {preprocessed_loitering_AIS_dir}")
        preprocessed_loitering_AIS_csv_path = csv_files[0]

        logger.info(f"Loading pre-processed loitering events labels from {preprocessed_loitering_AIS_csv_path}")

        ais_spark_df = spark.read.option("header", True).option("inferSchema", True).csv(preprocessed_loitering_AIS_csv_path)
        ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
        result_df = ProcessDataService.create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
            ais_spark_df, ais_csv_path, spark
        )

        new_file_name_4 = os.getenv(
            "OUTPUT_FOLDER_NAME_FOR_AGG_LOITERING_UNIQUE_VESSEL_DATA_PITSIKALIS_2019_WITH_EXTRA_FEATURES",
            "Placeholder_folder_data_processed_by_spark"
        )
        aggregated_loitering_output_path = os.path.join(base_output_dir, new_file_name_4)

        ProcessDataService.save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019(
            spark_df=result_df,
            output_dir=aggregated_loitering_output_path,
            spark=spark,
            num_buckets=int(os.getenv("BUCKETS_FOR_LARGE_WRITE", "2000")),
            bucket_coalesce=True,
            allow_bucket_fallback_to_chunked=True,
            progress_log_every=int(os.getenv("BUCKET_PROGRESS_LOG_EVERY", "20")),
            sample_for_bucket_size=False,
            max_buckets_to_process=10,
            compress_parts=False
        )
        logger.info(f"AGGREGATED Loitering events saved to '{aggregated_loitering_output_path}'")

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "Aggregated Loitering events processed and saved.",
            "output_path_loitering": aggregated_loitering_output_path
        }), 200

    except Exception as e:
        logger.error("Error loading or processing events data", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500

    
@swag_from(path.join(path.dirname(__file__), '../docs/write_agg_Pitsikalis_2019_AIS_data_in_database.yml'))
@preprocess_data_bp.route('/write-agg-Pitsikalis-2019-AIS-data-in-database', methods=['POST'])
def write_agg_Pitsikalis_2019_AIS_data_in_database():
    """
    Writes the aggregated Pitsikalis 2019 AIS data into the database.

    Behaviour:
    - Try standard Spark CSV read (header + inferSchema).
    - If that fails (exception) or yields no rows, attempt a fallback read that disables CSV quoting
      (quote and escape set to NUL) and uses permissive/multiLine parsing and large maxColumns.
    - If fallback also fails, skip the file and log an error but continue processing other files.
    """
    logger.info("Received request at /write-agg-Pitsikalis-2019-AIS-data-in-database")

    try:
        spark = SparkSessionInitializer.init_spark_session("Write_Agg_Pitsikalis_2019_AIS_Data_In_Database_[Data_Processing_API]")
        # === Paths / environment choices ===
        is_container = ProcessDataService._is_running_in_container()

        if is_container:
            input_dir = "/app/processed_output"
        else:
            input_dir = "shared/utils/processed_output"

        # Get aggregated events directories (you had these originally)
        aggregated_AIS_dir_names = [
            #"agg_transship_unique_vessel_data_pitsikalis2019_WITH_EXTRA_FEATURES",
            #"agg_loitering_unique_vessel_data_pitsikalis2019_WITH_EXTRA_FEATURES",
            #"agg_normal_unique_vessel_data_pitsikalis2019_WITH_EXTRA_FEATURES",
            "agg_stopping_unique_vessel_data_pitsikalis2019_WITH_EXTRA_FEATURES"
        ]

        for dir_name in aggregated_AIS_dir_names:
            aggregated_AIS_dir = os.path.join(input_dir, dir_name)
            csv_files = glob.glob(os.path.join(aggregated_AIS_dir, "*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in directory: {aggregated_AIS_dir} (skipping)")
                continue

            for individual_aggregated_AIS_csv_file in csv_files:
                logger.info(f"Loading aggregated events labels from {individual_aggregated_AIS_csv_file}")

                ais_spark_df = None
                # --- 1) Standard read attempt (keep identical to original behaviour) ---
                try:
                    ais_spark_df = (
                        spark.read
                        .option("header", True)
                        .option("inferSchema", True)
                        .csv(individual_aggregated_AIS_csv_file)
                    )
                    # normalize column names
                    ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
                except Exception as std_err:
                    logger.warning("Standard spark.read CSV failed for %s: %s", individual_aggregated_AIS_csv_file, std_err)
                    ais_spark_df = None

                # --- 2) If standard read produced an empty DataFrame or failed, attempt fallback ---
                def _df_has_rows(df) -> bool:
                    try:
                        # Prefer rdd.isEmpty() if available (cheap on driver), else take(1)
                        return not df.rdd.isEmpty()
                    except Exception:
                        return df.take(1) != []

                if ais_spark_df is None or (ais_spark_df is not None and not _df_has_rows(ais_spark_df)):
                    # Log reason
                    if ais_spark_df is None:
                        logger.info("Attempting fallback read for file (disabling CSV quoting / permissive parsing).")
                    else:
                        logger.info("Standard read returned no rows; attempting fallback read (disabling CSV quoting).")

                    try:
                        # Fallback read: disable quoting/escaping by using NUL char; allow multiline & permissive mode;
                        # increase maxColumns to avoid univocity column bounds errors for very long rows.
                        ais_spark_df = (
                            spark.read
                            .option("header", True)
                            .option("inferSchema", True)  # keep inferSchema if you want types; can be False if expensive
                            .option("multiLine", True)
                            .option("mode", "PERMISSIVE")
                            .option("quote", "\u0000")   # disable parsing of quoted fields
                            .option("escape", "\u0000")
                            .option("maxColumns", "500000")  # keep large to cover extremely long rows
                            .csv(individual_aggregated_AIS_csv_file)
                        )
                        ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
                        logger.info("Fallback read succeeded for file %s (rows approx: %s partitions).", individual_aggregated_AIS_csv_file, getattr(ais_spark_df.rdd, "getNumPartitions", lambda: "?")())
                    except Exception as fb_err:
                        logger.exception("Fallback read also failed for %s: %s -- skipping this file.", individual_aggregated_AIS_csv_file, fb_err)
                        ais_spark_df = None

                # --- 3) If still no DataFrame or empty, skip file ---
                if ais_spark_df is None:
                    logger.error("Could not read CSV file %s with either standard or fallback reader; skipping.", individual_aggregated_AIS_csv_file)
                    continue

                try:
                    # double-check non-empty
                    if not _df_has_rows(ais_spark_df):
                        logger.info("No data rows found after parsing %s — skipping.", individual_aggregated_AIS_csv_file)
                        continue
                except Exception as e_check:
                    logger.debug("Could not determine if DataFrame has rows; proceeding cautiously: %s", e_check)

                # --- 4) Continue with existing logic: write to DB via upsert function (unchanged) ---
                try:
                    SaveAISDataService.upsert_aggregated_ais_spark_df_to_db(ais_spark_df)
                    logger.info(f"Written aggregated data from '{individual_aggregated_AIS_csv_file}' to database.")
                except Exception as upsert_err:
                    logger.exception("Failed to upsert DataFrame from %s: %s", individual_aggregated_AIS_csv_file, upsert_err)
                    # continue processing other files

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "Aggregated Pitsikalis 2019 AIS data written to database."
        }), 200

    except Exception as e:
        logger.error("Error processing aggregated Pitsikalis 2019 AIS data", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@swag_from(path.join(path.dirname(__file__), '../docs/generate_image_trajectory_dataset_for_all_behavior_types.yml'))
@preprocess_data_bp.route('/generate-image-trajectory-dataset-for-all-behavior-types', methods=['POST'])
def generate_image_trajectory_dataset_for_all_behavior_types():
    """
    Generates grayscale image trajectory datasets (in .png format) for all behavior types using Spark.
    The image resolutions used are 120x120, 128x128, 224x224, 256x256, 299x299, 384x384, 512x512, and 640x640 pixels.
    The resulting grayscale images are saved in this structure:
    - output_directory/
        - image_resolution_120x120/
            - behavior_type_1/
                - image_1.png
                - image_2.png
                - ...
            - behavior_type_2/
                - image_1.png
                - image_2.png
                - ...
            - ...
        - ...
    Returns:
        Flask Response: JSON object with status and message indicating success or failure.
    """
    logger.info("Received request at /generate-image-trajectory-dataset-for-all-behavior-types")

    behavior_types_to_generate_dataset = ["TRANSSHIPMENT", "NORMAL", "STOPPING", "LOITERING"]

    try:
        spark = SparkSessionInitializer.init_spark_session("Generate_Image_Trajectory_Dataset_[Data_Processing_API]")

        is_container = ProcessDataService._is_running_in_container()

        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        output_dir = os.path.join(base_output_dir, "image_trajectory_datasets_all_behavior_types")

        ProcessDataService.generate_image_trajectory_dataset_for_all_behavior_types_with_spark(
            spark=spark,
            output_dir=output_dir,
            behavior_types_to_generate_dataset=behavior_types_to_generate_dataset,
            max_rows_per_behavior=800
        )

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "Image trajectory datasets for all behavior types generated and saved.",
            "output_path": output_dir
        }), 200

    except Exception as e:
        logger.error("Error generating image trajectory datasets", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500
    
@swag_from(path.join(path.dirname(__file__), '../docs/generate_csv_image_trajectory_and_cog_sog_timestamp_arrays_dataset_for_transshipment_events.yml'))
@preprocess_data_bp.route('/generate-csv-image-trajectory-and-cog-sog-timestamp-arrays-dataset-for-transshipment-events', methods=['POST'])
def generate_csv_image_trajectory_and_cog_sog_timestamp_arrays_dataset_for_transshipment_events():
    """
    Generates a CSV dataset containing image trajectory arrays and corresponding COG/SOG/timestamp arrays for transshipment events.
    The dataset is created using Spark and saved as a CSV file in the specified output directory.
    Returns:
        Flask Response: JSON object with status and message indicating success or failure, along with the output path of the generated CSV dataset.
    """
    logger.info("Received request at /generate-csv-image-trajectory-and-cog-sog-timestamp-arrays-dataset-for-transshipment-events")

    behavior_types_to_generate_dataset = ["TRANSSHIPMENT", "NORMAL", "STOPPING", "LOITERING"]

    try:
        spark = SparkSessionInitializer.init_spark_session("Generate_CSV_Image_Trajectory_and_COG_SOG_Timestamp_Arrays_Dataset_[Data_Processing_API]")

        is_container = ProcessDataService._is_running_in_container()

        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        output_dir = os.path.join(base_output_dir, "csv_image_trajectory_and_cog_sog_timestamp_arrays_dataset_for_transshipment_events")

        ProcessDataService.generate_csv_image_trajectory_and_cog_sog_timestamp_arrays_dataset_for_transshipment_events_with_spark(
            spark=spark,
            output_dir=output_dir,
            behavior_types_to_generate_dataset=behavior_types_to_generate_dataset,
            max_rows_per_behavior=320
        )

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "CSV dataset with image trajectory and COG/SOG/timestamp arrays for transshipment events generated and saved.",
            "output_path": output_dir
        }), 200

    except Exception as e:
        logger.error("Error generating CSV dataset with image trajectory and COG/SOG/timestamp arrays for transshipment events", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback"
                        : traceback_str}), 500

    
# @swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_DEPRECATED_MUST_BE_SKIPPED.yml'))
# @preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-DEPRECATED-MUST-BE-SKIPPED', methods=['POST'])
# def process_Pitsikalis_2019_AIS_data_DEPRECATED_MUST_BE_SKIPPED():
#     """
#     Process the Pitsikalis 2019 AIS data (DEPRECATED_MUST_BE_SKIPPED).
#     Expects a POST request. Loads data from a predefined CSV path ("Pitsikalis_2019_filtered_fluentname_data_v2"),
#     processes it using Spark, and saves the processed data as new CSV files: one for transshipment events ("ais_transshipment_events")
#     and another for non-transshipment events ("ais_loitering_non_loitering_stopping_events").
#     """
#     logger.info("Received request at /process-Pitsikalis-2019-AIS-data-DEPRECATED-MUST-BE-SKIPPED")

#     try:
#         spark = SparkSessionInitializer.init_spark_session("Pitsikalis_2019_AIS_DEPRECATED_MUST_BE_SKIPPED_[Data_Processing_API]")

#         is_container = ProcessDataService._is_running_in_container()  # log if in container or not
        
#         # input dir from env var or default
#         if is_container:
#             input_dir = "/app/processed_output"
#         else:
#             input_dir = "shared/utils/processed_output"

#         # Build paths to the CSV files
#         preprocessed_AIS_dir = os.path.join(input_dir, "ais_loitering_non_loitering_stopping_events")
#         csv_files = glob.glob(os.path.join(preprocessed_AIS_dir, "*.csv"))
#         if not csv_files:
#             raise FileNotFoundError(f"No CSV files found in directory: {preprocessed_AIS_dir}")
#         preprocessed_AIS_csv_path = csv_files[0]

#         # if is_container:
#         #     ais_csv_path = "/app/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"
#         # else:
#         #     ais_csv_path = "shared/utils/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"

#         logger.info(f"Loading pre-processed loitering, non-loitering, and stopping events labels from {preprocessed_AIS_csv_path}")

#         ais_spark_df = spark.read.option("header", True).option("inferSchema", True).csv(preprocessed_AIS_csv_path)
#         ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
#         result_df = ProcessDataService.DEPRECATED_MUST_BE_SKIPPED_convert_to_vessel_events_Pitsikalis_2019(ais_spark_df, spark)

#         # Save the processed dataframe as CSV
#         # processed output dir from env var or default
#         if is_container:
#             base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
#         else:
#             base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

#         # Write to an output directory (coalesce to 1)
#         new_file_name = os.getenv("OUTPUT_FOLDER_NAME_FOR_CLEANED_LOITER_STOP_TRAJECTORIES_SUBSET_V2", "Placeholder_folder_data_processed_by_spark")
#         cleaned_non_transshipment_output_path = os.path.join(base_output_dir, new_file_name)

#         # Call helper function (it coalesces and then promotes)
#         ProcessDataService.save_spark_df_as_csv(result_df, cleaned_non_transshipment_output_path, spark)
#         logger.info(f"CLEANED Loitering, non-loitering, and stopping events saved to '{cleaned_non_transshipment_output_path}'")

#         logger.info("Task finished. Stopping Spark session...")
#         spark.stop()

#         return jsonify({
#             "status": "success",
#             "message": "CLEANED Loitering, non-loitering, and stopping events processed and saved.",
#             "output_path": cleaned_non_transshipment_output_path
#         }), 200
#     except Exception as e:
#         logger.error("Error loading or processing events data", exc_info=True)
#         traceback_str = traceback.format_exc()
#         return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500    