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
        transshipment_df, raw_non_transshipment_df = Process_Data_Service.split_events_Pitsikalis_2019(df_events)

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
        aggregated_non_transshipment_df = Process_Data_Service.process_ais_events_Pitsikalis_2019(spark, ais_csv_path, raw_non_transshipment_df)
        
        # Write to an output directory (coalesce to 1)
        new_file_name_2 = os.getenv("OUTPUT_FOLDER_NAME_FOR_NON_TRANSSHIP_AIS_DATA_SPARK_PITSIKALIS_2019", "Placeholder_folder_data_processed_by_spark")
        non_transshipment_output_path = os.path.join(base_output_dir, new_file_name_2)

        # Call helper function (it coalesces and then promotes)
        Process_Data_Service.save_spark_df_as_csv(aggregated_non_transshipment_df, non_transshipment_output_path, spark)
        logger.info(f"Loitering, non-loitering, and stopping events saved to '{non_transshipment_output_path}'")

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