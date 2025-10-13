import os
import traceback
import logging
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from os import path
from dotenv import load_dotenv
from ..services.process_data_service import Process_Data_Service
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

@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_labels_data.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-labels-data', methods=['POST'])
def process_Pitsikalis_2019_labels_data():
    """
    Process the Pitsikalis 2019 labels data (recognized composite events).
    Expects a POST request. Loads data from a predefined CSV path, processes it using Spark,
    and saves the processed data as a new CSV file.
    """
    logger.info("Received request at /process-Pitsikalis-2019-labels-data")

    try:
        spark = Process_Data_Service.init_spark_session("Pitsikalis_2019_Labels_[Data_Processing_API]")

        expected_header = ["FluentName", "MMSI", "Argument", "Value", "T_start", "T_end"]
        
        is_container = Process_Data_Service._is_running_in_container()  # log if in container or not

        # Read Spark master URL and event log dir from environment variables or use defaults
        if is_container:
            csv_path = "/app/datasets/Maritime_Composite_Events/CEs/recognised_CEs.csv"
        else:
            csv_path = "shared/utils/datasets/Maritime_Composite_Events/CEs/recognised_CEs.csv"

        logger.info(f"WARNING: Loading raw labels data from {csv_path}")

        df = Process_Data_Service.load_spark_labels_df_from_Pitsikalis_2019_csv(spark, csv_path, expected_header)

        logger.info("WARNING: Filtering and transforming labels data...")

        df_processed = Process_Data_Service.filter_and_transform_Pitsikalis_2019_labels_data(df)

        summary = Process_Data_Service.inspect_spark_labels_dataframe_Pitsikalis_2019(df_processed)

        # Save the processed dataframe as CSV
        # processed output dir from env var or default
        if is_container:
            output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        new_file_name = os.getenv("OUTPUT_FOLDER_NAME_FOR_LABELS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        output_path = os.path.join(output_dir, new_file_name)
        
        # Save processed Spark DataFrame as CSV
        Process_Data_Service.save_spark_df_as_csv(df_processed, output_path, spark)

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

@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_PART_1.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-PART-1', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_PART_1():
    """
    Process the Pitsikalis 2019 AIS data (PART 1).
    Expects a POST request. Loads data from a predefined CSV path ("Pitsikalis_2019_filtered_fluentname_data_v2"),
    processes it using Spark, and saves the processed data as new CSV files: one for transshipment events ("ais_transshipment_events")
    and another for non-transshipment events ("ais_loitering_non_loitering_stopping_events").
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-PART-1-data")

    try:
        spark = Process_Data_Service.init_spark_session("Pitsikalis_2019_AIS_PART_1_[Data_Processing_API]")

        is_container = Process_Data_Service._is_running_in_container()  # log if in container or not
        
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
        df_events = Process_Data_Service.load_events_Pitsikalis_2019(spark, events_csv_path)

        logger.info("Splitting events into transshipment and non-transshipment")
        transshipment_df, loitering_df, normal_df, stopping_df = Process_Data_Service.split_events_Pitsikalis_2019(df_events)

        logger.info("Collecting relevant transshipment MMSI pairs")
        relevant_transship_mmsi = Process_Data_Service.get_relevant_transship_mmsi_Pitsikalis_2019(transshipment_df)

        # Save the processed dataframe as CSV
        # processed output dir from env var or default
        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        new_file_name_1 = os.getenv("OUTPUT_FOLDER_NAME_FOR_TRANSSHIP_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        transshipment_output_path = os.path.join(base_output_dir, new_file_name_1)

        logger.info("Saving transshipment events (coalesced single CSV)")
        Process_Data_Service.save_spark_df_as_csv(transshipment_df, transshipment_output_path, spark)
        logger.info(f"Transshipment events saved to '{transshipment_output_path}'")

        logger.info("WARNING: Transshipment processing done. Now processing non-transshipment AIS data...")
        logger.info("Processing AIS and joining with non-transshipment events...")
        aggregated_loitering_df, aggregated_normal_df, aggregated_stopping_df = Process_Data_Service.process_ais_events_Pitsikalis_2019(spark, ais_csv_path, loitering_df, normal_df, stopping_df)
        
        # Write to an output directory (coalesce to 1)
        new_file_name_2 = os.getenv("OUTPUT_FOLDER_NAME_FOR_LOITERING_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        new_file_name_3 = os.getenv("OUTPUT_FOLDER_NAME_FOR_NORMAL_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        new_file_name_4 = os.getenv("OUTPUT_FOLDER_NAME_FOR_STOPPING_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")

        loitering_output_path = os.path.join(base_output_dir, new_file_name_2)
        normal_output_path = os.path.join(base_output_dir, new_file_name_3)
        stopping_output_path = os.path.join(base_output_dir, new_file_name_4)

        # Call helper function (it coalesces and then promotes)
        Process_Data_Service.save_spark_df_as_csv(aggregated_loitering_df, loitering_output_path, spark, allow_multiple_files=False, original_filename=new_file_name_2)
        logger.info(f"Loitering events saved to '{loitering_output_path}'")

        Process_Data_Service.save_spark_df_as_csv(aggregated_normal_df, normal_output_path, spark, allow_multiple_files=False, original_filename=new_file_name_3)
        logger.info(f"Normal events saved to '{normal_output_path}'")

        Process_Data_Service.save_spark_df_as_csv(aggregated_stopping_df, stopping_output_path, spark, allow_multiple_files=False, original_filename=new_file_name_4)
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
    
@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_PART_2.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-PART-2', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_PART_2():
    """
    Process the Pitsikalis 2019 AIS data (PART 2).
    Expects a POST request. Loads preprocessed data from predefined CSV paths ("ais_transshipment_events_Pitsikalis_2019",
    "ais_loitering_events_Pitsikalis_2019", "ais_normal_events_Pitsikalis_2019" and "ais_stopping_events_Pitsikalis_2019"),
    cross-references them with raw AIS data using Spark, and saves the processed data as new CSV files: one for transshipment events
    ("agg_transship_unique_vessel_data_pitsikalis2019_WITH_EXTRA_FEATURES"), other for loitering events
    ("agg_loitering_unique_vessel_data_pitsikalis2019_WITH_EXTRA_FEATURES"), another for normal events
    ("agg_normal_unique_vessel_data_pitsikalis2019_WITH_EXTRA_FEATURES"), and another for stopping events
    ("agg_stopping_unique_vessel_data_pitsikalis2019_WITH_EXTRA_FEATURES").

    This variant keeps init_spark_session unchanged and applies:
      - more aggressive repartitioning (smaller partitions per executor)
      - reduced chunk_bytes (default 256KB) for chunked streaming writer
      - driver disk-space pre-check (fail-fast)
      - logs choices and falls back gracefully if helpers missing
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-PART-2-data")

    try:
        # === Create Spark session (unchanged init_spark_session is used) ===
        spark = Process_Data_Service.init_spark_session("Pitsikalis_2019_AIS_PART_2_[Data_Processing_API]")

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
        is_container = Process_Data_Service._is_running_in_container()

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
        result_df = Process_Data_Service.create_aggregated_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
            ais_spark_df, ais_csv_path, spark
        )

        new_file_name_1 = os.getenv(
            "OUTPUT_FOLDER_NAME_FOR_AGG_TRANSSHIP_UNIQUE_VESSEL_DATA_PITSIKALIS_2019_WITH_EXTRA_FEATURES",
            "Placeholder_folder_data_processed_by_spark"
        )
        aggregated_transshipment_output_path = os.path.join(base_output_dir, new_file_name_1)

        # existing saver (coalesces + promotes) - leave as-is since it works for transshipment
        Process_Data_Service.save_spark_df_as_csv(result_df, aggregated_transshipment_output_path, spark)
        logger.info(f"AGGREGATED Transshipment events saved to '{aggregated_transshipment_output_path}'")

        ####

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
        result_df = Process_Data_Service.create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
            ais_spark_df, ais_csv_path, spark
        )

        new_file_name_2 = os.getenv(
            "OUTPUT_FOLDER_NAME_FOR_AGG_NORMAL_UNIQUE_VESSEL_DATA_PITSIKALIS_2019_WITH_EXTRA_FEATURES",
            "Placeholder_folder_data_processed_by_spark"
        )
        aggregated_normal_output_path = os.path.join(base_output_dir, new_file_name_2)

        Process_Data_Service.save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019(
            spark_df=result_df,
            output_dir=aggregated_normal_output_path,
            spark=spark,
            num_buckets=int(os.getenv("BUCKETS_FOR_LARGE_WRITE", "400")),
            bucket_coalesce=True,
            allow_bucket_fallback_to_chunked=True,
            progress_log_every=int(os.getenv("BUCKET_PROGRESS_LOG_EVERY", "20")),
            sample_for_bucket_size=False
        )

        logger.info(f"AGGREGATED Normal events saved to '{aggregated_normal_output_path}'")

        # Stopping events
        preprocessed_stopping_AIS_dir = os.path.join(input_dir, "ais_stopping_events_Pitsikalis_2019")
        csv_files = glob.glob(os.path.join(preprocessed_stopping_AIS_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {preprocessed_stopping_AIS_dir}")
        preprocessed_stopping_AIS_csv_path = csv_files[0]

        logger.info(f"Loading pre-processed stopping events labels from {preprocessed_stopping_AIS_csv_path}")

        ais_spark_df = spark.read.option("header", True).option("inferSchema", True).csv(preprocessed_stopping_AIS_csv_path)
        ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
        result_df = Process_Data_Service.create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
            ais_spark_df, ais_csv_path, spark
        )

        new_file_name_3 = os.getenv(
            "OUTPUT_FOLDER_NAME_FOR_AGG_STOPPING_UNIQUE_VESSEL_DATA_PITSIKALIS_2019_WITH_EXTRA_FEATURES",
            "Placeholder_folder_data_processed_by_spark"
        )
        aggregated_stopping_output_path = os.path.join(base_output_dir, new_file_name_3)

        Process_Data_Service.save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019(
            spark_df=result_df,
            output_dir=aggregated_stopping_output_path,
            spark=spark,
            num_buckets=int(os.getenv("BUCKETS_FOR_LARGE_WRITE", "400")),
            bucket_coalesce=True,
            allow_bucket_fallback_to_chunked=True,
            progress_log_every=int(os.getenv("BUCKET_PROGRESS_LOG_EVERY", "20")),
            sample_for_bucket_size=False
        )
        logger.info(f"AGGREGATED Stopping events saved to '{aggregated_stopping_output_path}'")

        # Loitering events
        preprocessed_loitering_AIS_dir = os.path.join(input_dir, "ais_loitering_events_Pitsikalis_2019")
        csv_files = glob.glob(os.path.join(preprocessed_loitering_AIS_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {preprocessed_loitering_AIS_dir}")
        preprocessed_loitering_AIS_csv_path = csv_files[0]

        logger.info(f"Loading pre-processed loitering events labels from {preprocessed_loitering_AIS_csv_path}")

        ais_spark_df = spark.read.option("header", True).option("inferSchema", True).csv(preprocessed_loitering_AIS_csv_path)
        ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
        result_df = Process_Data_Service.create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
            ais_spark_df, ais_csv_path, spark
        )

        new_file_name_4 = os.getenv(
            "OUTPUT_FOLDER_NAME_FOR_AGG_LOITERING_UNIQUE_VESSEL_DATA_PITSIKALIS_2019_WITH_EXTRA_FEATURES",
            "Placeholder_folder_data_processed_by_spark"
        )
        aggregated_loitering_output_path = os.path.join(base_output_dir, new_file_name_4)

        Process_Data_Service.save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019(
            spark_df=result_df,
            output_dir=aggregated_loitering_output_path,
            spark=spark,
            num_buckets=int(os.getenv("BUCKETS_FOR_LARGE_WRITE", "400")),
            bucket_coalesce=True,
            allow_bucket_fallback_to_chunked=True,
            progress_log_every=int(os.getenv("BUCKET_PROGRESS_LOG_EVERY", "20")),
            sample_for_bucket_size=False
        )
        logger.info(f"AGGREGATED Loitering events saved to '{aggregated_loitering_output_path}'")

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "Aggregated Transshipment, Loitering, non-loitering, and stopping events processed and saved.",
            "output_path_transshipment": aggregated_transshipment_output_path,
            "output_path_normal": aggregated_normal_output_path,
            "output_path_stopping": aggregated_stopping_output_path,
            "output_path_loitering": aggregated_loitering_output_path
        }), 200

    except Exception as e:
        logger.error("Error loading or processing events data", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500

@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_AIS_data_OPTIONAL_MUST_BE_SKIPPED.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-AIS-data-OPTIONAL-MUST-BE-SKIPPED', methods=['POST'])
def process_Pitsikalis_2019_AIS_data_OPTIONAL_MUST_BE_SKIPPED():
    """
    Process the Pitsikalis 2019 AIS data (OPTIONAL_MUST_BE_SKIPPED).
    Expects a POST request. Loads data from a predefined CSV path ("Pitsikalis_2019_filtered_fluentname_data_v2"),
    processes it using Spark, and saves the processed data as new CSV files: one for transshipment events ("ais_transshipment_events")
    and another for non-transshipment events ("ais_loitering_non_loitering_stopping_events").
    """
    logger.info("Received request at /process-Pitsikalis-2019-AIS-data-OPTIONAL-MUST-BE-SKIPPED")

    try:
        spark = Process_Data_Service.init_spark_session("Pitsikalis_2019_AIS_OPTIONAL_MUST_BE_SKIPPED_[Data_Processing_API]")

        is_container = Process_Data_Service._is_running_in_container()  # log if in container or not
        
        # input dir from env var or default
        if is_container:
            input_dir = "/app/processed_output"
        else:
            input_dir = "shared/utils/processed_output"

        # Build paths to the CSV files
        preprocessed_AIS_dir = os.path.join(input_dir, "ais_loitering_non_loitering_stopping_events")
        csv_files = glob.glob(os.path.join(preprocessed_AIS_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {preprocessed_AIS_dir}")
        preprocessed_AIS_csv_path = csv_files[0]

        # if is_container:
        #     ais_csv_path = "/app/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"
        # else:
        #     ais_csv_path = "shared/utils/datasets/ais_brest_synopses_v0.8/ais_brest_locations.csv"

        logger.info(f"Loading pre-processed loitering, non-loitering, and stopping events labels from {preprocessed_AIS_csv_path}")

        ais_spark_df = spark.read.option("header", True).option("inferSchema", True).csv(preprocessed_AIS_csv_path)
        ais_spark_df = ais_spark_df.toDF(*[c.strip() for c in ais_spark_df.columns])
        result_df = Process_Data_Service.OPTIONAL_MUST_BE_SKIPPED_convert_to_vessel_events_Pitsikalis_2019(ais_spark_df, spark)

        # Save the processed dataframe as CSV
        # processed output dir from env var or default
        if is_container:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        else:
            base_output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/tmp/processed_output")

        # Write to an output directory (coalesce to 1)
        new_file_name = os.getenv("OUTPUT_FOLDER_NAME_FOR_CLEANED_LOITER_STOP_TRAJECTORIES_SUBSET_V2", "Placeholder_folder_data_processed_by_spark")
        cleaned_non_transshipment_output_path = os.path.join(base_output_dir, new_file_name)

        # Call helper function (it coalesces and then promotes)
        Process_Data_Service.save_spark_df_as_csv(result_df, cleaned_non_transshipment_output_path, spark)
        logger.info(f"CLEANED Loitering, non-loitering, and stopping events saved to '{cleaned_non_transshipment_output_path}'")

        logger.info("Task finished. Stopping Spark session...")
        spark.stop()

        return jsonify({
            "status": "success",
            "message": "CLEANED Loitering, non-loitering, and stopping events processed and saved.",
            "output_path": cleaned_non_transshipment_output_path
        }), 200
    except Exception as e:
        logger.error("Error loading or processing events data", exc_info=True)
        traceback_str = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": traceback_str}), 500    