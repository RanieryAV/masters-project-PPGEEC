import os
import traceback
import logging
import subprocess
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

    ##### ===> ALL FUNCTIONS BELOW ARE UNTESTED!!!!! <=== ######
    @staticmethod
    def load_ais_spark_df_Pitsikalis_2019(spark, ais_path: str):
        """
        Load AIS CSV into a Spark DataFrame and normalize/convert the timestamp column.
        - ais_path: path to AIS CSV (can be local path or HDFS path)
        Returns a Spark DataFrame with columns including: id, timestamp (TimestampType), longitude, latitude, annotation, speed, heading, turn, course
        """
        logger.info(f"Loading AIS CSV from: {ais_path}")
        df = (
            spark.read
            .option("header", True)
            .option("sep", ",")
            .option("inferSchema", True)
            .csv(ais_path)
        )

        # normalize column names (strip whitespace) and rename Id -> id if present
        for colname in df.columns:
            if colname.strip() != colname:
                df = df.withColumnRenamed(colname, colname.strip())
        if "Id" in df.columns and "id" not in df.columns:
            df = df.withColumnRenamed("Id", "id")

        # Convert timestamp that may be in milliseconds (numeric) or ISO strings into TimestampType
        # We attempt: if numeric epoch in ms -> from_unixtime(col/1000), else try parsing string via to_timestamp
        ts_col = F.col("timestamp")
        df = df.withColumn("timestamp_str", ts_col.cast("string"))

        df = df.withColumn(
            "timestamp",
            F.when(
                F.regexp_extract(F.col("timestamp_str"), r"^\d{10,}$", 0) != "",
                F.to_timestamp(F.from_unixtime((F.col("timestamp_str").cast("double") / 1000.0)))
            ).otherwise(
                F.to_timestamp(F.col("timestamp_str"))
            )
        ).drop("timestamp_str")

        # Ensure required columns exist (create if missing to keep schema stable)
        required_cols = ["id", "timestamp", "longitude", "latitude", "annotation", "speed", "heading", "turn", "course"]
        for c in required_cols:
            if c not in df.columns:
                df = df.withColumn(c, F.lit(None))

        return df.select(*required_cols)

    @staticmethod
    def split_events_Pitsikalis_2019(events_df):
        """
        Split events DataFrame into transshipment and non-transshipment DataFrames.
        - events_df: DataFrame containing at least columns FluentName, MMSI, Argument (or MMSI_2), T_start, T_end, EventIndex
        Returns a tuple: (transshipment_df, non_transshipment_df, relevant_transship_mmsi_list)
        where relevant_transship_mmsi_list is a list of dicts {'MMSI': <>, 'MMSI_2': <>}
        """
        logger.info("Splitting events into transshipment and non-transshipment categories")

        # Normalize columns: if 'Argument' exists rename to 'MMSI_2'
        cols = [c for c in events_df.columns]
        if "Argument" in cols and "MMSI_2" not in cols:
            events_df = events_df.withColumnRenamed("Argument", "MMSI_2")

        # Ensure T_start/T_end are timestamps; attempt string/epoch parsing
        # If they are numeric epoch (ms), convert accordingly
        events_df = events_df.withColumn("T_start_str", F.col("T_start").cast("string"))
        events_df = events_df.withColumn("T_end_str", F.col("T_end").cast("string"))

        events_df = events_df.withColumn(
            "T_start",
            F.when(
                F.regexp_extract(F.col("T_start_str"), r"^\d{10,}$", 0) != "",
                F.to_timestamp(F.from_unixtime(F.col("T_start_str").cast("double") / 1000.0))
            ).otherwise(F.to_timestamp(F.col("T_start_str")))
        ).withColumn(
            "T_end",
            F.when(
                F.regexp_extract(F.col("T_end_str"), r"^\d{10,}$", 0) != "",
                F.to_timestamp(F.from_unixtime(F.col("T_end_str").cast("double") / 1000.0))
            ).otherwise(F.to_timestamp(F.col("T_end_str")))
        ).drop("T_start_str", "T_end_str")

        # Map FluentName to Category if Category not present (attempt to reuse your earlier mapping)
        if "Category" not in events_df.columns:
            fluent_name_categories = {
                'highSpeedNC': 'NORMAL',
                'loitering': 'LOITERING',
                'tuggingSpeed': 'LOITERING',
                'stopped': 'STOPPING',
                'anchoredOrMoored': 'STOPPING',
                'rendezVous': 'TRANSSHIPMENT',
            }
            mapping_expr = F.create_map(
                *sum(([F.lit(k), F.lit(v)] for k, v in fluent_name_categories.items()), [])
            )
            events_df = events_df.withColumn("Category", mapping_expr.getItem(F.col("FluentName")))

        # Split
        transshipment_df = events_df.filter(F.col("Category") == "TRANSSHIPMENT")
        non_transshipment_df = events_df.filter(F.col("Category").isin("LOITERING", "NORMAL", "STOPPING"))

        # Collect relevant transshipment MMSIs to Python list of dicts
        trans_pairs = []
        if "MMSI" in transshipment_df.columns and "MMSI_2" in transshipment_df.columns:
            rows = transshipment_df.select("MMSI", "MMSI_2").collect()
            for r in rows:
                trans_pairs.append({"MMSI": r["MMSI"], "MMSI_2": r["MMSI_2"]})

        return transshipment_df, non_transshipment_df, trans_pairs

    @staticmethod
    def save_transshipment_Pitsikalis_2019(spark, transshipment_df, target_dir: str):
        """
        Save transshipment events DataFrame to a single CSV file.
        - transshipment_df: Spark DataFrame of transshipment events
        - target_dir: directory path where temporary part-file will be written; the promotion function will produce the single CSV file.
        Returns True if promotion succeeded (via spark_func_promote_csv_from_temporary), False otherwise.
        """
        logger.info(f"Writing transshipment DataFrame to temporary dir: {target_dir}")
        # write to directory as Spark expects; coalesce to 1 for a single part file
        transshipment_df.coalesce(1).write.mode("overwrite").option("header", True).csv(target_dir)
        # promote to single CSV using existing helper
        try:
            promoted = Process_Data_Service.spark_func_promote_csv_from_temporary(spark, target_dir)
            if promoted:
                logger.info(f"Transshipment CSV promoted for target_dir: {target_dir}")
            else:
                logger.warning(f"Transshipment CSV promotion returned False for target_dir: {target_dir}")
            return promoted
        except Exception as e:
            logger.exception(f"Error while promoting transshipment CSV: {e}")
            return False

    @staticmethod
    def process_non_transshipment_ais_Pitsikalis_2019(spark, ais_df, non_transshipment_df, target_dir: str):
        """
        For non-transshipment events:
        - filter AIS data to MMSIs present in non_transshipment_df and within the global min/max event timestamps,
        - join AIS rows to events by matching id==MMSI and timestamp in [T_start, T_end],
        - attach EventIndex and Category to the AIS rows,
        - write output to target_dir (as CSV via Spark) and promote to single CSV using the existing promote helper.
        Returns True if data was written and promoted successfully, False otherwise.
        """
        logger.info("Processing non-transshipment AIS data and joining to event ranges")

        # Guard: if no non-trans events, nothing to do
        if non_transshipment_df.rdd.isEmpty():
            logger.info("No non-transshipment events found; skipping processing.")
            return False

        # compute global min/max timestamps of events to limit AIS search range
        min_ts_row = non_transshipment_df.agg(F.min("T_start").alias("min_ts")).collect()[0]
        max_ts_row = non_transshipment_df.agg(F.max("T_end").alias("max_ts")).collect()[0]
        min_ts = min_ts_row["min_ts"]
        max_ts = max_ts_row["max_ts"]

        logger.info(f"Filtering AIS data between {min_ts} and {max_ts}")

        # Semi-join: only keep AIS rows where id appears in non_transshipment_df.MMSI
        distinct_mmsi_df = non_transshipment_df.select(F.col("MMSI").alias("MMSI")).distinct()
        # prepare ais_df: ensure id column exists, timestamp as timestamp (assume load_ais does that)
        ais_filtered = ais_df.join(distinct_mmsi_df, ais_df.id == distinct_mmsi_df.MMSI, "semi") \
                            .filter((F.col("timestamp") >= F.lit(min_ts)) & (F.col("timestamp") <= F.lit(max_ts)))

        # perform range join: ais.timestamp between event T_start and T_end for same MMSI
        a = ais_filtered.alias("a")
        e = non_transshipment_df.alias("e")

        join_cond = (F.col("a.id") == F.col("e.MMSI")) & \
                    (F.col("a.timestamp") >= F.col("e.T_start")) & \
                    (F.col("a.timestamp") <= F.col("e.T_end"))

        joined = a.join(e, join_cond, "inner")

        # select required AIS columns plus EventIndex and Category
        select_cols = [
            F.col("a.id").alias("id"),
            F.col("a.timestamp").alias("timestamp"),
            F.col("a.longitude").alias("longitude"),
            F.col("a.latitude").alias("latitude"),
            F.col("a.annotation").alias("annotation"),
            F.col("a.speed").alias("speed"),
            F.col("a.heading").alias("heading"),
            F.col("a.turn").alias("turn"),
            F.col("a.course").alias("course"),
            F.col("e.EventIndex").alias("EventIndex"),
            F.col("e.Category").alias("Category"),
        ]

        result_df = joined.select(*select_cols)

        # Write to directory (coalesce to 1 part-file to make promotion easier)
        logger.info(f"Writing joined non-transshipment AIS data to temporary dir: {target_dir}")
        result_df.coalesce(1).write.mode("overwrite").option("header", True).csv(target_dir)

        # promote to single CSV using existing helper
        try:
            promoted = Process_Data_Service.spark_func_promote_csv_from_temporary(spark, target_dir)
            if promoted:
                logger.info(f"Non-transshipment AIS CSV promoted for target_dir: {target_dir}")
            else:
                logger.warning(f"Non-transshipment AIS CSV promotion returned False for target_dir: {target_dir}")
            return promoted
        except Exception as e:
            logger.exception(f"Error while promoting non-transshipment AIS CSV: {e}")
            return False

    