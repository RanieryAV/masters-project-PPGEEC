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

# Load environment variables
load_dotenv()

class Spark_Services:
    @staticmethod
    def init_spark_session(spark_session_name):
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

    @staticmethod
    def spark_func_promote_csv_from_temporary(spark, output_path: str) -> bool:
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