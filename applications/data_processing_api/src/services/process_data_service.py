import os
import traceback
import logging
import subprocess
import socket

from flask import Blueprint, request, jsonify
from flasgger import swag_from
from os import path
from dotenv import load_dotenv
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Tuple, List, Dict

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
    ######################## DOCKER DETECTION FUNCTIONS ########################
    @staticmethod
    def _is_running_in_container() -> bool:
        """
        Heuristic to detect whether code is running inside a container (Docker / containerd / k8s).
        Checks /.dockerenv, /proc/1/cgroup and an optional environment flag IN_DOCKER.
        Returns True when running inside a container, False otherwise.
        """
        import os

        # Quick check for docker-specific file
        if os.path.exists("/.dockerenv"):
            return True

        # Check cgroup info for docker/k8s indicators
        try:
            with open("/proc/1/cgroup", "rt") as f:
                cg = f.read()
                if any(tok in cg for tok in ("docker", "kubepods", "containerd")):
                    return True
        except Exception:
            # ignore read errors; fall through
            pass

        # Explicit override via environment variable (useful for tests)
        if os.environ.get("IN_DOCKER", "").lower() in ("1", "true", "yes"):
            return True

        return False

    @staticmethod
    def detect_spark_uid_gid(default_uid=1001, default_gid=1001, container_name="spark-master"):
        """
        Best-effort detection of UID/GID of the Spark user inside the container.

        - Tries `docker exec spark-master id -u` and `id -g`.
        - If it fails, falls back to defaults (1001:1001).
        """
        logger = logging.getLogger(__name__)
        uid, gid = default_uid, default_gid
        try:
            uid_out = subprocess.run(
                ["docker", "exec", container_name, "id", "-u"],
                capture_output=True, text=True, check=True
            )
            gid_out = subprocess.run(
                ["docker", "exec", container_name, "id", "-g"],
                capture_output=True, text=True, check=True
            )
            uid = int(uid_out.stdout.strip())
            gid = int(gid_out.stdout.strip())
            logger.info(f"Detected Spark container UID={uid}, GID={gid}")
        except Exception as e:
            logger.warning(f"Could not auto-detect Spark UID/GID, using defaults {default_uid}:{default_gid} ({e})")
        return uid, gid
    
    ######################################## END ########################################

    ########################### FILESYSTEM PERMISSIONS HELPERS ##########################

    @staticmethod
    def ensure_local_processed_dirs(target_uid: int = 1001, target_gid: int = 1001):
        """
        Ensure that host-local directories used for processed output and datasets exist
        and attempt to set ownership/permissions similar to the original shell commands:

            sudo mkdir -p ./shared/utils/processed_output ./shared/utils/datasets
            sudo chown -R 1001:1001 ./shared/utils/processed_output ./shared/utils/datasets
            sudo chmod -R 0777 ./shared/utils/processed_output ./shared/utils/datasets

        Parameters:
        - target_uid: desired owner UID to chown to (default 1001)
        - target_gid: desired owner GID to chown to (default 1001)

        Behavior (best-effort):
        - creates the directories (os.makedirs)
        - attempts chmod 0o777 recursively
        - if running as root, does os.chown recursively to target_uid:target_gid
        - if not root, attempts shutil.chown where possible
        - logs clear instructions if manual sudo chown is needed
        """
        import os
        import shutil
        from pathlib import Path
        import pwd
        import grp
        import logging

        # use existing module logger if present, otherwise get a basic logger
        try:
            logger  # if logger exists in module/class scope
        except Exception:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                ch = logging.StreamHandler()
                ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                logger.addHandler(ch)
            logger.setLevel(logging.INFO)

        # Resolve base directories (relative to current working directory)
        base_paths = [
            Path.cwd() / "shared" / "utils" / "processed_output",
            Path.cwd() / "shared" / "utils" / "datasets",
        ]

        logger.info(f"ensure_local_processed_dirs: target_uid={target_uid}, target_gid={target_gid}")
        logger.info(f"ensure_local_processed_dirs: base paths = {[str(p) for p in base_paths]}")

        # 1) Ensure directories exist
        for p in base_paths:
            try:
                p.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {p}")
            except Exception as e:
                logger.warning(f"Could not create directory {p}: {e}")

        # Helper: chmod recursively (best-effort)
        def chmod_recursive(target_path: Path, mode: int):
            try:
                os.chmod(str(target_path), mode)
            except Exception as e:
                logger.debug(f"chmod on {target_path} failed: {e}")
            if target_path.is_dir():
                for root, dirs, files in os.walk(str(target_path)):
                    for name in dirs:
                        try:
                            os.chmod(os.path.join(root, name), mode)
                        except Exception:
                            pass
                    for name in files:
                        try:
                            os.chmod(os.path.join(root, name), mode)
                        except Exception:
                            pass

        # Helper: chown recursively when running as root (safe)
        def chown_recursive_as_root(target_path: Path, uid: int, gid: int):
            if os.geteuid() == 0:
                for root, dirs, files in os.walk(str(target_path)):
                    try:
                        os.chown(root, uid, gid)
                    except Exception:
                        pass
                    for d in dirs:
                        try:
                            os.chown(os.path.join(root, d), uid, gid)
                        except Exception:
                            pass
                    for f in files:
                        try:
                            os.chown(os.path.join(root, f), uid, gid)
                        except Exception:
                            pass
                return True
            return False

        # Helper: chown using shutil for non-root (best-effort)
        def chown_recursive_nonroot(target_path: Path, user: str = None, group: str = None):
            for root, dirs, files in os.walk(str(target_path)):
                for d in dirs:
                    try:
                        shutil.chown(os.path.join(root, d), user=user, group=group)
                    except Exception:
                        pass
                for f in files:
                    try:
                        shutil.chown(os.path.join(root, f), user=user, group=group)
                    except Exception:
                        pass

        # 2) Attempt chmod 0777 recursively (best-effort)
        for p in base_paths:
            try:
                chmod_recursive(p, 0o777)
                logger.info(f"Attempted chmod 0777 recursively on {p}")
            except Exception as e:
                logger.warning(f"Failed recursive chmod on {p}: {e}")

        # 3) Attempt chown: prefer root chown, otherwise try non-root best-effort
        if os.geteuid() == 0:
            try:
                for p in base_paths:
                    chown_recursive_as_root(p, target_uid, target_gid)
                logger.info(f"Chowned {', '.join(str(x) for x in base_paths)} to {target_uid}:{target_gid} (run as root)")
            except Exception as e:
                logger.warning(f"Failed chown as root: {e}")
        else:
            try:
                # attempt to resolve uid/gid to names, if available on host
                user_name = None
                group_name = None
                try:
                    user_name = pwd.getpwuid(target_uid).pw_name
                except Exception:
                    user_name = None
                try:
                    group_name = grp.getgrgid(target_gid).gr_name
                except Exception:
                    group_name = None

                if user_name or group_name:
                    chown_recursive_nonroot(base_paths[0], user=user_name, group=group_name)
                    chown_recursive_nonroot(base_paths[1], user=user_name, group=group_name)
                    logger.info(f"Attempted non-root shutil.chown to {user_name}:{group_name} (best-effort)")
                else:
                    logger.info("Cannot resolve target UID/GID to user/group names on this host; skipping non-root chown attempt.")
            except Exception as e:
                logger.debug(f"Non-root chown attempts failed: {e}")

        # 4) If not root and ownership still differs, print clear manual instructions
        if os.geteuid() != 0:
            problematic = []
            for p in base_paths:
                try:
                    st = p.stat()
                    if (st.st_uid != target_uid) or (st.st_gid != target_gid):
                        problematic.append(str(p))
                except Exception:
                    pass

            if problematic:
                logger.warning(
                    "Some created directories are not owned by the desired UID/GID. "
                    "If you need root-level ownership, run the following on the host:\n"
                    f"  sudo chown -R {target_uid}:{target_gid} {' '.join(problematic)}\n"
                    f"  sudo chmod -R 0777 {' '.join(problematic)}\n"
                )

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
    
    ############################### END ##############################

    ############### SPARK DATAFRAME PROCESSING HELPERS ###############

    @staticmethod
    def save_spark_df_as_csv(spark_df, output_path: str, spark: SparkSession):
        """Save a Spark DataFrame as a single CSV file in the specified output directory."""
        import os
        import shutil
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Saving processed data to {output_path}")
        
        # coalesce to 1 if you want one file, else multiple part-files
        spark_df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)

        moved = Process_Data_Service.spark_func_promote_csv_from_temporary(spark, output_path)
        if not moved:
            logger.warning(f"Could not promote CSV from temporary for output path {output_path}")
        else:
            logger.info("CSV file promoted from temporary.")
        
        # Adjust permissions on moved files
        Process_Data_Service.adjust_file_permissions(output_path)

        logger.info("ATTENTION: Data saved successfully with correct file permissions!")
    
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
    
    ######################## END ########################

    ############### SPARK SESSION HELPERS ###############
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
        spark_executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", "5g")    # memory per executor
        spark_driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "6g")

        spark = (
            SparkSession.builder
            .appName(spark_session_name)
            .master(spark_master_url)
            .config("spark.eventLog.enabled", "false")
            .config("spark.eventLog.dir", eventlog_dir)
            .config("spark.sql.shuffle.partitions", "12")  # adjust as needed
            # resource-related configs (minimal additions)
            .config("spark.cores.max", spark_cores_max)
            .config("spark.executor.cores", spark_executor_cores)
            .config("spark.executor.memory", spark_executor_memory)
            .config("spark.driver.memory", spark_driver_memory)
            .config("spark.driver.host", driver_host)
            .config("spark.driver.bindAddress", "0.0.0.0")
            .config("spark.local.dir", "/app/processed_output/spark_tmp")
            .config("spark.executorEnv.SPARK_LOCAL_DIRS", "/app/processed_output/spark_tmp")
            .config("spark.executor.extraJavaOptions", "-Djava.io.tmpdir=/app/processed_output/spark_tmp")
            .getOrCreate()
        )
        return spark

    ######################## END ########################

    ############### PITSIKALIS 2019 DATA HELPERS  ###############
        ########### (RECOGNIZED COMPOSITE EVENTS) ###########
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

        #### END OF (RECOGNIZED COMPOSITE EVENTS) ####

        ## CROSS-REFERENCE RAW AIS WITH (RECOGNIZED COMPOSITE EVENTS) ##
                    #### WARNING: UNTESTED!!! ####
    def load_events_Pitsikalis_2019(spark: SparkSession, events_path: str) -> DataFrame:
        """
        Load the events CSV into a Spark DataFrame and normalize column names.

        - reads CSV from `events_path` (expects header)
        - strips whitespace from column names
        - renames 'Argument' -> 'MMSI_2' if present
        - converts 'T_start' and 'T_end' to timestamp columns
        Returns the Spark DataFrame.
        """
        # read CSV (let Spark infer schema)
        events_df = spark.read.option("header", True).option("inferSchema", True).csv(events_path)

        # normalize column names (strip)
        new_cols = [c.strip() for c in events_df.columns]
        events_df = events_df.toDF(*new_cols)

        # rename 'Argument' to 'MMSI_2' if exists
        if "Argument" in events_df.columns and "MMSI_2" not in events_df.columns:
            events_df = events_df.withColumnRenamed("Argument", "MMSI_2")

        # ensure T_start and T_end are timestamps
        if "T_start" in events_df.columns:
            events_df = events_df.withColumn("T_start", F.to_timestamp(F.col("T_start")))
        if "T_end" in events_df.columns:
            events_df = events_df.withColumn("T_end", F.to_timestamp(F.col("T_end")))

        return events_df


    def split_events_Pitsikalis_2019(events_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Split events DataFrame into:
        - transshipment_df: rows where Category == 'TRANSSHIPMENT'
        - non_transshipment_df: rows where Category in ('LOITERING', 'NORMAL', 'STOPPING')

        Returns (transshipment_df, non_transshipment_df) as Spark DataFrames.
        """
        transshipment_df = events_df.filter(F.col("Category") == F.lit("TRANSSHIPMENT"))
        non_transshipment_df = events_df.filter(F.col("Category").isin("LOITERING", "NORMAL", "STOPPING"))
        return transshipment_df, non_transshipment_df


    def get_relevant_transship_mmsi_Pitsikalis_2019(transshipment_df: DataFrame) -> List[Dict]:
        """
        Build a list of dicts [{'MMSI': ..., 'MMSI_2': ...}, ...] for transshipment pairs.

        Collects the pairs into Python memory (like the original code's relevant_transship_mmsi).
        """
        # select only necessary columns and collect
        cols = []
        if "MMSI" in transshipment_df.columns:
            cols.append("MMSI")
        if "MMSI_2" in transshipment_df.columns:
            cols.append("MMSI_2")

        if not cols:
            return []

        rows = transshipment_df.select(*cols).collect()
        result = []
        for r in rows:
            entry = {}
            if "MMSI" in r.__fields__:
                entry["MMSI"] = r["MMSI"]
            else:
                entry["MMSI"] = r[0] if len(r) > 0 else None
            # prefer named access for MMSI_2
            if len(r) > 1:
                entry["MMSI_2"] = r[1]
            else:
                entry["MMSI_2"] = r["MMSI_2"] if "MMSI_2" in r.__fields__ else None
            result.append(entry)
        return result


    # def save_transshipment_Pitsikalis_2019(transshipment_df: DataFrame, spark: SparkSession, output_path: str):
    #     """
    #     Save transshipment_df as a single CSV file (coalesced). Uses Process_Data_Service.save_spark_df_as_csv
    #     to preserve your CSV promotion and permission handling.

    #     - output_path: full path to output directory (not base dir)
    #     Returns the output path (directory) used for writing.
    #     """
    #     out_dir = os.path.join(output_path, "ais_transshipment_events")
    #     # ensure parent exists
    #     Path(out_dir).mkdir(parents=True, exist_ok=True)
        
    #     # save using helper
    #     Process_Data_Service.save_spark_df_as_csv(transshipment_df, out_dir, spark)


    def process_ais_events_Pitsikalis_2019(
        spark: SparkSession,
        ais_path: str,
        non_transshipment_df: DataFrame,
        output_base_dir: str = os.path.join("..", "datasets"),
    ) -> str:
        """
        Process AIS CSV with Spark and join with non-transshipment events to produce
        loitering/non-loitering/stopping annotated AIS records.

        Steps:
        - load AIS csv into Spark DataFrame
        - normalize columns and rename 'Id' -> 'id'
        - convert AIS 'timestamp' (ms) into a proper timestamp column named 'timestamp' (timestamp type)
        - filter AIS rows by relevant MMSI set and global min/max event times to reduce data scanned
        - perform a join where (ais.id == events.MMSI) AND (ais.timestamp between events.T_start and events.T_end)
        - select and order the output columns:
        ['id','timestamp','longitude','latitude','annotation','speed','heading','turn','course','EventIndex','Category']
        - write a single CSV (coalesced) to an output dir under output_base_dir and call promotion helper
        Returns the output directory path used for writing.
        """
        # read AIS CSV
        ais_df = spark.read.option("header", True).option("inferSchema", True).csv(ais_path)

        # normalize column names
        ais_df = ais_df.toDF(*[c.strip() for c in ais_df.columns])

        # rename 'Id' -> 'id' if required
        if "Id" in ais_df.columns and "id" not in ais_df.columns:
            ais_df = ais_df.withColumnRenamed("Id", "id")

        # Convert timestamp in ms to timestamp type called 'timestamp'
        # If incoming timestamp already looks like epoch ms number, do conversion, otherwise try to cast to timestamp
        if "timestamp" in ais_df.columns:
            # create numeric_ts = timestamp / 1000 and then from_unixtime
            ais_df = ais_df.withColumn("timestamp_ms", F.col("timestamp"))
            # if timestamp is numeric (integer/long), convert from ms
            ais_df = ais_df.withColumn(
                "timestamp",
                F.when(
                    F.col("timestamp_ms").cast("long").isNotNull(),
                    F.from_unixtime((F.col("timestamp_ms").cast("double") / 1000.0)).cast(TimestampType()),
                ).otherwise(F.to_timestamp(F.col("timestamp_ms"))),
            ).drop("timestamp_ms")

        # Ensure required AIS columns exist; create null columns if absent so schema is consistent
        required_ais_cols = ["id", "timestamp", "longitude", "latitude", "annotation", "speed", "heading", "turn", "course"]
        for col in required_ais_cols:
            if col not in ais_df.columns:
                ais_df = ais_df.withColumn(col, F.lit(None))

        # Reduce events time window first (global min and max) to reduce AIS scanning
        aggs = non_transshipment_df.agg(F.min("T_start").alias("min_start"), F.max("T_end").alias("max_end")).collect()
        if aggs and len(aggs) > 0:
            min_start = aggs[0]["min_start"]
            max_end = aggs[0]["max_end"]
        else:
            min_start = None
            max_end = None

        # build relevant MMSI list (distinct)
        relevant_mmsi = [r[0] for r in non_transshipment_df.select("MMSI").distinct().rdd.map(lambda r: (r[0],)).collect()]

        # filter AIS by relevant MMSIs and time range (best-effort filter)
        ais_filtered = ais_df
        if relevant_mmsi:
            # to avoid very large IN lists in SQL, broadcast small list; if large list then use join below
            if len(relevant_mmsi) < 10_000:
                ais_filtered = ais_filtered.filter(F.col("id").isin(relevant_mmsi))
            else:
                # join approach: create df of relevant MMSIs
                mmsi_df = spark.createDataFrame([(m,) for m in relevant_mmsi], ["MMSI__tmp_"])
                ais_filtered = ais_filtered.join(mmsi_df, ais_filtered.id == F.col("MMSI__tmp_"), "inner").drop("MMSI__tmp_")

        if min_start is not None and max_end is not None:
            ais_filtered = ais_filtered.filter((F.col("timestamp") >= F.lit(min_start)) & (F.col("timestamp") <= F.lit(max_end)))

        # Join AIS with events using range condition:
        # condition: ais.id == events.MMSI AND ais.timestamp between events.T_start and events.T_end
        join_cond = (
            (ais_filtered.id == non_transshipment_df.MMSI)
            & (ais_filtered.timestamp >= non_transshipment_df.T_start)
            & (ais_filtered.timestamp <= non_transshipment_df.T_end)
        )

        joined = ais_filtered.join(non_transshipment_df, on=join_cond, how="inner")

        # Select output columns in the order expected
        output_columns = [
            "id",
            "timestamp",
            "longitude",
            "latitude",
            "annotation",
            "speed",
            "heading",
            "turn",
            "course",
            "EventIndex",
            "Category",
        ]

        # If EventIndex or Category are not present in events, ensure they exist (avoid exceptions)
        for c in ["EventIndex", "Category"]:
            if c not in joined.columns:
                joined = joined.withColumn(c, F.lit(None))

        result_df = joined.select(*output_columns)

        # # Write to an output directory (coalesce to 1)
        # out_dir = os.path.join(output_base_dir, "ais_loitering_non_loitering_stopping_events")
        # Path(out_dir).mkdir(parents=True, exist_ok=True)

        # # Use Process_Data_Service.save_spark_df_as_csv if available, to preserve promotion logic
        # try:
        #     # Attempt to leverage your helper (it coalesces and then promotes)
        #     Process_Data_Service.save_spark_df_as_csv(result_df, out_dir, spark)
        # except Exception:
        #     # fallback: basic spark write and then try to promote
        #     result_df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_dir)
        #     try:
        #         Process_Data_Service.spark_func_promote_csv_from_temporary(spark, out_dir)
        #     except Exception:
        #         pass

        return result_df


    def run_pipeline_Pitsikalis_2019(spark: SparkSession, events_path: str, ais_path: str) -> Dict:
        """
        Orchestrate the loading, splitting, and processing pipeline for Pitsikalis 2019.

        - loads events
        - splits into transshipment and non-transshipment
        - saves the transshipment events CSV (single coalesced file)
        - processes AIS and joins with non-transshipment events, saving the annotated AIS CSV (single coalesced file)
        - returns a small summary dict including paths and the relevant transshipment list
        """
        print(f"Loading events from {events_path}")
        events_df = load_events_Pitsikalis_2019(spark, events_path)

        print("Splitting events into transshipment and non-transshipment")
        transshipment_df, non_transshipment_df = split_events_Pitsikalis_2019(events_df)

        print("Collecting relevant transshipment MMSI pairs")
        relevant_transship_mmsi = get_relevant_transship_mmsi_Pitsikalis_2019(transshipment_df)

        print("Saving transshipment events (coalesced single CSV)")
        trans_out_dir = save_transshipment_Pitsikalis_2019(transshipment_df, spark)

        print("Processing AIS and joining with non-transshipment events")
        loiter_out_dir = process_ais_events_Pitsikalis_2019(spark, ais_path, non_transshipment_df)

        summary = {
            "transshipment_output_dir": trans_out_dir,
            "loitering_output_dir": loiter_out_dir,
            "relevant_transship_mmsi": relevant_transship_mmsi,
        }
        print("Pipeline finished. Summary:", summary)
        return summary


    # Example usage (uncomment when running in real module):
    # spark = Process_Data_Service.init_spark_session("Pitsikalis2019DataProcessing")
    # events_path = os.path.join('..', 'datasets', 'filtered_fluentname_data_v2.csv')
    # ais_path = os.path.join('..', 'datasets', 'ais_brest_synopses_v0.8', 'ais_brest_locations.csv')
    # run_pipeline_Pitsikalis_2019(spark, events_path, ais_path)

                    #### WARNING: UNTESTED!!! ####
    ######################## END ########################