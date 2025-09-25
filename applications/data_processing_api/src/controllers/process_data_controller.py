import os
import traceback
import logging
from flask import Blueprint, request, jsonify
from flasgger import swag_from
from os import path
from dotenv import load_dotenv #move to services

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

######################EVERYTHING BELOW MUST BE MOVED TO services######################
# Load environment variables#move to services
load_dotenv()#move to services

def init_spark_session(spark_session_name):#move to services
    """Initialize and return a SparkSession using environment vars."""
    spark_master_rpc_port = os.getenv("SPARK_MASTER_RPC_PORT", "7077")
    spark_master_url = os.getenv("SPARK_MASTER_URL", f"spark://spark-master:{spark_master_rpc_port}")
    eventlog_dir = os.getenv("SPARK_EVENTLOG_DIR", "/opt/spark-events")

    logger.info(f"Spark master URL: {spark_master_url}")
    logger.info(f"Event log directory: {eventlog_dir}")

    spark = (
        SparkSession.builder
        .appName(spark_session_name)
        .master(spark_master_url)
        .config("spark.eventLog.enabled", "false")
        .config("spark.eventLog.dir", eventlog_dir)
        .config("spark.sql.shuffle.partitions", "200")
        # remover spark.executorEnv.USER se estiver causando conflito
        # .config("spark.executorEnv.USER", "root")
        .config("spark.local.dir", "/app/processed_output/spark_tmp")
        .config("spark.executorEnv.SPARK_LOCAL_DIRS", "/app/processed_output/spark_tmp")
        .config("spark.executor.extraJavaOptions", "-Djava.io.tmpdir=/app/processed_output/spark_tmp")
        .getOrCreate()
    )
    return spark

def promote_csv_from_temporary(spark, output_path: str) -> bool:
    """
    Promote (move) a part-*.csv file from Spark's internal _temporary folders
    to the root of the given output directory.

    Returns True if a rename was performed, False otherwise.
    """
    jvm = spark._jvm
    Path = jvm.org.apache.hadoop.fs.Path
    FileSystem = jvm.org.apache.hadoop.fs.FileSystem
    FsPermission = jvm.org.apache.hadoop.fs.permission.FsPermission

    hadoop_conf = spark._jsc.hadoopConfiguration()
    fs = FileSystem.get(hadoop_conf)

    out_path = Path(output_path)

    try:
        entries = fs.listStatus(out_path)
    except Exception as e:
        spark._jvm.scala.Predef.println(f"[promote_csv] cannot listStatus on {output_path}: {e}")
        return False

    # Helper recursive function to search for part-*.csv within a Path
    def search_and_move(current_path):
        try:
            statuses = fs.listStatus(current_path)
        except Exception as e2:
            spark._jvm.scala.Predef.println(f"[promote_csv] cannot listStatus inside {current_path}: {e2}")
            return False

        for st in statuses:
            p = st.getPath()
            name = p.getName()
            # If this is a CSV part file
            if name.startswith("part-") and name.endswith(".csv"):
                dest = Path(output_path + "/" + name)
                spark._jvm.scala.Predef.println(f"[promote_csv] renaming {p} → {dest}")
                try:
                    moved = fs.rename(p, dest)
                except Exception as re:
                    spark._jvm.scala.Predef.println(f"[promote_csv] rename failed {p} → {dest}: {re}")
                    return False

                # Try to set permissive file permissions (rwx for user/group/others)
                try:
                    perm = FsPermission.valueOf("rwxrwxrwx")
                    fs.setPermission(dest, perm)
                    spark._jvm.scala.Predef.println(f"[promote_csv] setPermission(rwxrwxrwx) on {dest}")
                except Exception as pe:
                    # If this fails, log and attempt a best-effort local chmod/chown fallback on file and parent dir
                    spark._jvm.scala.Predef.println(f"[promote_csv] setPermission failed on {dest}: {pe}")
                    try:
                        # Try to extract a POSIX path from the Hadoop Path (dest)
                        try:
                            dest_path_str = dest.toUri().getPath()
                        except Exception:
                            # fallback to toString()
                            dest_path_str = dest.toString()

                        # If the string contains a scheme like "file:/...", normalize it
                        if isinstance(dest_path_str, str) and dest_path_str.startswith("file:"):
                            dest_path_str = dest_path_str.split(":", 1)[1]

                        import os
                        import shutil
                        import subprocess
                        from pathlib import Path as PyPath

                        parent_dir = str(PyPath(dest_path_str).parent)

                        # 1) try local chmod on file and parent
                        try:
                            os.chmod(dest_path_str, 0o777)
                            spark._jvm.scala.Predef.println(f"[promote_csv] local os.chmod(0777) applied to {dest_path_str}")
                        except Exception as e_chmod:
                            spark._jvm.scala.Predef.println(f"[promote_csv] local os.chmod failed on file {dest_path_str}: {e_chmod}")

                        try:
                            os.chmod(parent_dir, 0o777)
                            spark._jvm.scala.Predef.println(f"[promote_csv] local os.chmod(0777) applied to parent {parent_dir}")
                        except Exception as e_chmodp:
                            spark._jvm.scala.Predef.println(f"[promote_csv] local os.chmod failed on parent {parent_dir}: {e_chmodp}")

                        # 2) try shutil.chown (may require root)
                        try:
                            # attempt to chown to root:root first
                            shutil.chown(dest_path_str, user="root", group="root")
                            shutil.chown(parent_dir, user="root", group="root")
                            spark._jvm.scala.Predef.println(f"[promote_csv] shutil.chown(root:root) applied to {dest_path_str} and {parent_dir}")
                        except Exception as e_shch:
                            spark._jvm.scala.Predef.println(f"[promote_csv] shutil.chown(root:root) failed: {e_shch}")

                        # 3) fallback: try shell chown/chmod commands (non-fatal; check output)
                        try:
                            # try chown -R root:root on parent
                            proc = subprocess.run(["chown", "-R", "root:root", parent_dir], capture_output=True, text=True)
                            spark._jvm.scala.Predef.println(f"[promote_csv] chown -R root:root {parent_dir} exit={proc.returncode} out={proc.stdout} err={proc.stderr}")
                        except Exception as e_proc:
                            spark._jvm.scala.Predef.println(f"[promote_csv] subprocess chown(root) failed: {e_proc}")

                        try:
                            # try chown -R 1001:1001 on parent (give it to spark UID if root chown didn't help)
                            proc2 = subprocess.run(["chown", "-R", "1001:1001", parent_dir], capture_output=True, text=True)
                            spark._jvm.scala.Predef.println(f"[promote_csv] chown -R 1001:1001 {parent_dir} exit={proc2.returncode} out={proc2.stdout} err={proc2.stderr}")
                        except Exception as e_proc2:
                            spark._jvm.scala.Predef.println(f"[promote_csv] subprocess chown(1001) failed: {e_proc2}")

                        try:
                            # finally try recursive chmod 777 on parent
                            proc3 = subprocess.run(["chmod", "-R", "777", parent_dir], capture_output=True, text=True)
                            spark._jvm.scala.Predef.println(f"[promote_csv] chmod -R 777 {parent_dir} exit={proc3.returncode} out={proc3.stdout} err={proc3.stderr}")
                        except Exception as e_proc3:
                            spark._jvm.scala.Predef.println(f"[promote_csv] subprocess chmod -R failed: {e_proc3}")

                    except Exception as ce:
                        spark._jvm.scala.Predef.println(f"[promote_csv] local chmod/chown fallback failed on {dest}: {ce}")

                return moved  # True if rename succeeded, False otherwise
            # If this is a directory to dig into (like _temporary, 0, task_xyz)
            # Only recurse into directories (st.isDirectory)
            if st.isDirectory():
                # skip hidden directories if not needed, but here include all
                if search_and_move(p):
                    return True
        return False

    # First, see if there's a CSV directly under the root (already promoted)
    for st in entries:
        p = st.getPath()
        nm = p.getName()
        if nm.startswith("part-") and nm.endswith(".csv"):
            return True

    # Else, try to traverse into directories
    for st in entries:
        p = st.getPath()
        if st.isDirectory():
            if search_and_move(p):
                return True

    spark._jvm.scala.Predef.println(f"[promote_csv] no part-*.csv found to promote in {output_path}")
    return False

def load_df_from_csv(spark, relative_path: str, expected_header: list):
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


def filter_and_transform(df):
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


def inspect_dataframe(df):
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
######################EVERYTHING ABOVE MUST BE MOVED TO services######################

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
        spark = init_spark_session("Pitsikalis2019DataProcessingAPI")

        expected_header = ["FluentName", "MMSI", "Argument", "Value", "T_start", "T_end"]
        csv_path = "/app/datasets/Maritime_Composite_Events/CEs/recognised_CEs.csv"


        df = load_df_from_csv(spark, csv_path, expected_header)

        df_processed = filter_and_transform(df)

        summary = inspect_dataframe(df_processed)

        # Save the processed dataframe as CSV
        # processed output dir from env var or default
        output_dir = os.getenv("PROCESSED_OUTPUT_DIR", "/app/processed_output")
        #os.makedirs(output_dir, exist_ok=True)
        #os.chmod(output_dir, 0o777)

        new_file_name = os.getenv("OUTPUT_FOLDER_NAME_FOR_DATA_PROCESSED_BY_SPARK", "Placeholder_folder_data_processed_by_spark")
        output_path = f"{output_dir}/{new_file_name}"
        logger.info(f"Saving processed data to {output_path}")
        # coalesce to 1 if you want one file, else multiple part-files
        df_processed.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)

        moved = promote_csv_from_temporary(spark, output_path)
        if not moved:
            logger.warning(f"Could not promote CSV from temporary for output path {output_path}")
        else:
            logger.info("CSV file promoted from temporary.")

        # Ajustar permissões em arquivos movidos
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

        
        logger.info("Data saved successfully.")

        # After saving, set permissions to 777 for all files in the output directory
        for root, dirs, files in os.walk(output_path):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o777)

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
