import os
import traceback
import logging
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from os import path
from dotenv import load_dotenv
from ..services.process_data_service import Process_Data_Service
from domain.config.data_processing.spark_services import Spark_Services

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

@swag_from(path.join(path.dirname(__file__), '../docs/process_Pitsikalis_2019_data.yml'))
@preprocess_data_bp.route('/process-Pitsikalis-2019-data', methods=['POST'])
def process_Pitsikalis_2019_data():#REFACTOR MOVING WHAT IS POSSIBLE TO services; MUST RETURN JSON
    """
    Process the Pitsikalis 2019 data (recognized composite events).
    Expects a POST request. Loads data from a predefined CSV path, processes it using Spark,
    and saves the processed data as a new CSV file.
    """
    logger.info("Received request at /process-Pitsikalis-2019-data")

    try:
        spark = Spark_Services.init_spark_session("Pitsikalis2019DataProcessingAPI")

        expected_header = ["FluentName", "MMSI", "Argument", "Value", "T_start", "T_end"]
        csv_path = "/app/datasets/Maritime_Composite_Events/CEs/recognised_CEs.csv"


        df = Process_Data_Service.load_spark_df_from_Pitsikalis_2019_csv(spark, csv_path, expected_header)

        df_processed = Process_Data_Service.filter_and_transform_Pitsikalis_2019_data(df)

        summary = Process_Data_Service.inspect_spark_dataframe_Pitsikalis_2019(df_processed)

        # Save the processed dataframe as CSV
        # processed output dir from env var or default
        output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")

        new_file_name = os.getenv("OUTPUT_FOLDER_NAME_FOR_DATA_PROCESSED_BY_SPARK", "Placeholder_folder_data_processed_by_spark")
        output_path = f"{output_dir}/{new_file_name}"
        logger.info(f"Saving processed data to {output_path}")
        # coalesce to 1 if you want one file, else multiple part-files
        df_processed.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)

        moved = Spark_Services.spark_func_promote_csv_from_temporary(spark, output_path)
        if not moved:
            logger.warning(f"Could not promote CSV from temporary for output path {output_path}")
        else:
            logger.info("CSV file promoted from temporary.")

        # Adjust permissions on moved files
        Process_Data_Service.adjust_file_permissions(output_path)

        logger.info("ATTENTION: Data saved successfully with correct file permissions!")

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
