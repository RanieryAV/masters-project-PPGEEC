import logging
from flask import Blueprint, request, jsonify
import os
import socket
from pyspark.sql import SparkSession
from dotenv import load_dotenv

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

class SparkSessionInitializer:
    @staticmethod
    def init_spark_session(spark_session_name):
        """Initialize and return a SparkSession using environment vars."""
        spark_master_rpc_port = os.getenv("SPARK_MASTER_RPC_PORT", "7077")
        spark_master_url = os.getenv("SPARK_MASTER_URL", f"spark://spark-master:{spark_master_rpc_port}")
        eventlog_dir = os.getenv("SPARK_EVENTLOG_DIR", "/opt/spark-events")

        logger.info(f"Spark master URL: {spark_master_url}")
        logger.info(f"Event log directory: {eventlog_dir}")

        driver_host = os.getenv("SPARK_DRIVER_HOST")
        if not driver_host:
            try:
                hostname = socket.gethostname()
                driver_host = socket.gethostbyname(hostname)
            except Exception:
                # fallback razoável dentro de container
                driver_host = "0.0.0.0"

        # --- executor/driver resource tuning (read from env with sane defaults) ---
        # These values allow the master/workers to allocate larger executors instead of the
        # Spark default of 1g per executor. They are read from environment variables so
        # you can override them in docker-compose/.env without changing code.
        spark_cores_max = os.getenv("SPARK_CORES_MAX", "4")                 # total cores allowed for this app
        spark_executor_cores = os.getenv("SPARK_EXECUTOR_CORES", "2")       # cores per executor
        spark_executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", "6g")    # memory per executor
        spark_driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "7g")
        spark_driver_max_result_size = os.getenv("SPARK_DRIVER_MAX_RESULT_SIZE", "8g")

        spark_sql_debug_max_to_string_fields = os.getenv("SPARK_SQL_DEBUG_MAX_TO_STRING_FIELDS", "10000")

        # NEW: read RPC max-size in MB (user must give number in MB)
        spark_rpc_message_maxsize_mb = 2046  # default to 2046 MB (just under Spark's hard limit of 2047 MB)
        try:
            if spark_rpc_message_maxsize_mb <= 0 or spark_rpc_message_maxsize_mb == None:
                # default: keep Spark default 128 MB (do not force any change)
                logger.warning("SPARK_RPC_MESSAGE_MAXSIZE not set or invalid. Using Spark default of 128 MB.")
                spark_rpc_message_maxsize_mb = None
            else:
                spark_rpc_message_maxsize_mb = int(spark_rpc_message_maxsize_mb)
        except Exception:
            spark_rpc_message_maxsize_mb = None

        # enforce Spark hard limit (Spark will reject > 2047 MB)
        if spark_rpc_message_maxsize_mb is not None:
            if spark_rpc_message_maxsize_mb > 2046:
                logger.warning("Requested SPARK_RPC_MESSAGE_MAXSIZE=%s > 2046 MB. Clamping to 2046 MB.", spark_rpc_message_maxsize_mb)
                spark_rpc_message_maxsize_mb = 2046

        spark = (
            SparkSession.builder
            .appName(spark_session_name)
            .master(spark_master_url)
            .config("spark.eventLog.enabled", "false")
            .config("spark.eventLog.dir", eventlog_dir)
            .config("spark.sql.shuffle.partitions", "60")  # adjust as needed
            # resource-related configs (minimal additions)
            .config("spark.cores.max", spark_cores_max)
            .config("spark.executor.cores", spark_executor_cores)
            .config("spark.executor.memory", spark_executor_memory)
            .config("spark.driver.memory", spark_driver_memory)
            .config("spark.driver.maxResultSize", spark_driver_max_result_size)
            .config("spark.driver.host", driver_host)
            .config("spark.driver.bindAddress", "0.0.0.0")
            .config("spark.local.dir", "/app/processed_output/spark_tmp")
            .config("spark.executorEnv.SPARK_LOCAL_DIRS", "/app/processed_output/spark_tmp")
            .config("spark.executor.extraJavaOptions", "-Djava.io.tmpdir=/app/processed_output/spark_tmp")
            .config("spark.shuffle.compress", "true")
            .config("spark.rdd.compress", "true")
            .config("spark.shuffle.spill.compress", "true")
            .config("spark.sql.debug.maxToStringFields", spark_sql_debug_max_to_string_fields)
            # <-- important: increase RPC message max size to avoid serialized-task-too-large errors
            .config("spark.rpc.message.maxSize", spark_rpc_message_maxsize_mb)
            .getOrCreate()
        )
        return spark