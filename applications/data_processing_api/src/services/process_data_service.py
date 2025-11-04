import os
import traceback
import logging
import subprocess
import socket
import csv
import gzip
from typing import Optional

from flask import Blueprint, request, jsonify
from flasgger import swag_from
from os import path
from dotenv import load_dotenv
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, StructType, StructField, DoubleType, StringType, LongType, StructType, TimestampType, IntegerType
)
from pyspark.sql.window import Window
from typing import Any, Tuple, List, Dict

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

    def _parse_size_string(s):
        """Parse human-friendly size strings like '1G', '512M', '100K' or plain bytes into int bytes.
        Returns None on parse failure.
        """
        if s is None:
            return None
        try:
            s2 = str(s).strip().upper()
            if s2.endswith("G"):
                return int(float(s2[:-1]) * 1024 ** 3)
            if s2.endswith("M"):
                return int(float(s2[:-1]) * 1024 ** 2)
            if s2.endswith("K"):
                return int(float(s2[:-1]) * 1024)
            return int(s2)
        except Exception:
            return None

    # @staticmethod
    # def save_spark_df_as_csv(spark_df, output_path: str, spark: SparkSession):
    #     """Save a Spark DataFrame as a single CSV file in the specified output directory."""
    #     import os
    #     import shutil
    #     import logging
        
    #     logger = logging.getLogger(__name__)
    #     logger.info(f"Saving processed data to {output_path}")
        
    #     # coalesce to 1 if you want one file, else multiple part-files
    #     spark_df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)

    #     moved = Process_Data_Service.spark_func_promote_csv_from_temporary(spark, output_path)
    #     if not moved:
    #         logger.warning(f"Could not promote CSV from temporary for output path {output_path}")
    #     else:
    #         logger.info("CSV file promoted from temporary.")
        
    #     # Adjust permissions on moved files
    #     Process_Data_Service.adjust_file_permissions(output_path)

    #     logger.info("ATTENTION: Data saved successfully with correct file permissions!")

    def save_spark_df_as_csv(spark_df, output_path: str, spark: SparkSession, allow_multiple_files: bool = False, original_filename: Optional[str] = None):
        """Save a Spark DataFrame as a single CSV file in the specified output directory.

        Behavior:
        - By default (allow_multiple_files=False): Attempt the fast, standard Spark writer using coalesce(1).
        If that succeeds, try to promote the single part-*.csv to the directory root (spark_func_promote_csv_from_temporary)
        and then adjust permissions.
        - If allow_multiple_files=True: Do not call coalesce(1); write multiple part-*.csv files directly from executors.
        This avoids the memory/cost of coalescing and keeps the work distributed. Each partition writes its own file.
        This branch now has extra safety: aggressive partitioning, driver disk pre-check, optional repartitioning.
        - If the fast write (either coalesced or multi-file) fails, automatically fall back to a robust
        driver-streaming writer that emits a single CSV by receiving chunked bytes from executors
        (Process_Data_Service.save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019).
        - In all cases we attempt to adjust file permissions before returning.
        """
        import os
        import shutil
        import logging
        import traceback
        import csv
        import uuid

        logger = logging.getLogger(__name__)
        logger.info(f"Saving processed data to {output_path} (allow_multiple_files={allow_multiple_files})")

        # ----------------------------
        #  Helper: parse size strings
        # ----------------------------
        def _local_parse_size_string(s):
            if s is None:
                return None
            try:
                s2 = str(s).strip().upper()
                if s2.endswith("G"):
                    return int(float(s2[:-1]) * 1024 ** 3)
                if s2.endswith("M"):
                    return int(float(s2[:-1]) * 1024 ** 2)
                if s2.endswith("K"):
                    return int(float(s2[:-1]) * 1024)
                return int(s2)
            except Exception:
                return None

        # Prefer module parser if available
        parse_size = getattr(Process_Data_Service, "_parse_size_string", None) or _local_parse_size_string

        # ----------------------------
        #  Determine default_parallelism robustly & target partitions
        # ----------------------------
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

        try:
            multiplier = int(os.getenv("PARTITIONS_MULTIPLIER", "8"))          # default 8 (aggressive)
            min_partitions_floor = int(os.getenv("MIN_PARTITIONS_FLOOR", "400"))
            num_target_partitions = max(default_parallelism * multiplier, min_partitions_floor)
            max_cap = int(os.getenv("MAX_PARTITIONS_CAP", "10000"))
            if num_target_partitions > max_cap:
                logger.info("num_target_partitions %d capped to MAX_PARTITIONS_CAP=%d", num_target_partitions, max_cap)
                num_target_partitions = max_cap
            logger.info(
                "Computed num_target_partitions=%d (multiplier=%d, min_floor=%d, cap=%d)",
                num_target_partitions, multiplier, min_partitions_floor, max_cap
            )
        except Exception as e:
            num_target_partitions = max(default_parallelism * 8, 400)
            logger.warning("Error computing num_target_partitions, falling back to %s: %s", num_target_partitions, e)

        # ----------------------------
        #  Driver disk check (fail-fast)
        # ----------------------------
        def _find_existing_parent(p):
            cur = p
            while cur and not os.path.exists(cur):
                parent = os.path.dirname(cur)
                if parent == cur:
                    return os.path.sep
                cur = parent
            return cur or os.path.sep

        try:
            check_path = _find_existing_parent(output_path)
            try:
                driver_min_free_bytes = parse_size(os.getenv("DRIVER_MIN_FREE_BYTES", "1G"))
                if driver_min_free_bytes is None:
                    driver_min_free_bytes = 1 * 1024 ** 3
            except Exception:
                driver_min_free_bytes = 1 * 1024 ** 3

            du = shutil.disk_usage(check_path)
            free_bytes = du.free
            logger.info("Driver disk check: path=%s free=%d bytes (threshold=%d bytes)", check_path, free_bytes, driver_min_free_bytes)
            if free_bytes < driver_min_free_bytes:
                raise RuntimeError(
                    f"Insufficient free space on driver filesystem at {check_path}: free={free_bytes} bytes < threshold={driver_min_free_bytes} bytes."
                )
        except Exception as ex_disk:
            logger.exception("Driver disk-space check failed or insufficient: %s", ex_disk)
            raise

        # ----------------------------
        #  First: try the original fast path (either coalesced single-file or standard multi-file)
        # ----------------------------
        try:
            if allow_multiple_files:
                # Write multiple part-*.csv files (no coalesce) to avoid coalescing memory pressure,
                # but optionally repartition first to reduce per-partition size.
                df_to_write = spark_df
                try:
                    do_repartition = os.getenv("ALLOW_MULTI_REPARTITION", "true").lower() in ("1", "true", "yes")
                    if do_repartition:
                        logger.info("Repartitioning DataFrame to num_target_partitions=%d before executor-side multi-file write", num_target_partitions)
                        df_to_write = df_to_write.repartition(num_target_partitions)
                        logger.info("Repartition completed (to %d partitions).", num_target_partitions)
                except Exception as e_r:
                    logger.warning("Repartition before multi-file write failed or skipped: %s", e_r)

                # Ensure output dir exists (on the shared filesystem)
                try:
                    os.makedirs(output_path, exist_ok=True)
                except Exception as e:
                    logger.warning("Could not ensure output_path exists (%s): %s", output_path, e)

                cols = df_to_write.columns

                # Parameters for flushing and safety
                flush_every_n_rows = int(os.getenv("PART_FILE_FLUSH_EVERY_ROWS", "500"))  # flush every N rows
                max_rows_per_file = os.getenv("PART_FILE_MAX_ROWS")  # optional, not enforced unless set
                if max_rows_per_file:
                    try:
                        max_rows_per_file = int(max_rows_per_file)
                    except Exception:
                        max_rows_per_file = None
                else:
                    max_rows_per_file = None

                # Partition writer: runs on executors
                def _partition_writer(partition_index, iterator):
                    """
                    Write CSV rows from this partition to a local file under output_path.
                    Produces a file named part-{partition_index:05d}.csv.part and renames to .csv at end.
                    """
                    import os as _os
                    import csv as _csv
                    import logging as _logging
                    import uuid as _uuid

                    _logger = _logging.getLogger(__name__)

                    # Use partition-specific filename (includes uuid to avoid clashes across retries)
                    part_name_base = f"part-{partition_index:05d}-{_uuid.uuid4().hex[:8]}"
                    tmp_fname = _os.path.join(output_path, f"{part_name_base}.csv.part")
                    final_fname = _os.path.join(output_path, f"{original_filename}-part-{partition_index:05d}.csv")

                    try:
                        with open(tmp_fname, "w", newline="", encoding="utf-8") as fh:
                            writer = _csv.writer(fh)
                            # Write header
                            writer.writerow(cols)
                            row_counter = 0

                            for row in iterator:
                                try:
                                    rowd = row.asDict()
                                except Exception:
                                    rowd = {c: getattr(row, c, None) for c in cols}

                                out_cells = []
                                for c in cols:
                                    v = rowd.get(c) if isinstance(rowd, dict) else None
                                    if v is None:
                                        out_cells.append("")
                                    elif isinstance(v, (list, tuple)):
                                        elems = []
                                        for e in v:
                                            if e is None:
                                                elems.append("None")
                                            else:
                                                elems.append(str(e))
                                        out_cells.append("[" + ", ".join(elems) + "]")
                                    else:
                                        out_cells.append(str(v))
                                writer.writerow(out_cells)
                                row_counter += 1

                                # periodic flush to keep memory low and ensure progress on disk
                                if (flush_every_n_rows and (row_counter % flush_every_n_rows == 0)):
                                    try:
                                        fh.flush()
                                        try:
                                            os.fsync(fh.fileno())
                                        except Exception:
                                            pass
                                    except Exception as fe_flush:
                                        _logger.debug("Partition %d flush failed: %s", partition_index, fe_flush)

                                if max_rows_per_file and row_counter >= max_rows_per_file:
                                    _logger.info("Partition %d reached max_rows_per_file=%d; stopping write for this partition file.", partition_index, max_rows_per_file)
                                    break

                        # Atomic rename to final name; overwrite if exists (best-effort)
                        try:
                            if _os.path.exists(final_fname):
                                try:
                                    _os.remove(final_fname)
                                except Exception:
                                    pass
                            _os.replace(tmp_fname, final_fname)
                        except Exception as e_rename:
                            _logger.warning("Partition %d: rename tmp->final failed (%s). tmp=%s final=%s", partition_index, e_rename, tmp_fname, final_fname)
                            return iter([])

                        _logger.info("Partition %d: wrote %d rows -> %s", partition_index, row_counter, final_fname)
                    except Exception as e_part:
                        try:
                            if _os.path.exists(tmp_fname):
                                _os.remove(tmp_fname)
                        except Exception:
                            pass
                        _logger.exception("Partition %d: failed while writing partition file: %s", partition_index, e_part)
                        raise

                    return iter([])

                # Trigger the partitioned write via RDD mapPartitionsWithIndex action.
                try:
                    write_rdd = df_to_write.rdd.mapPartitionsWithIndex(_partition_writer)
                    write_rdd.count()
                except Exception as e_exec_write:
                    logger.exception("Executor-side multi-file write failed (%s). Will fall back to streaming writer.", e_exec_write)
                    raise

                # After successful partition writes, adjust permissions
                try:
                    Process_Data_Service.adjust_file_permissions(output_path)
                except Exception as e_adj:
                    logger.warning("adjust_file_permissions failed after executor-side multi-file write: %s", e_adj)

                logger.info("ATTENTION: Data saved successfully with correct file permissions (executor multi-file fast path).")
                return

            else:
                # coalesce to 1 if you want one file (original behavior)
                spark_df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
                logger.info("Spark write completed (coalesced single-file) to %s", output_path)

                # try to promote/move the part-*.csv produced by Spark to the root of output_path
                try:
                    moved = Process_Data_Service.spark_func_promote_csv_from_temporary(spark, output_path)
                    if not moved:
                        logger.warning(f"Could not promote CSV from temporary for output path {output_path}")
                    else:
                        logger.info("CSV file promoted from temporary.")
                except Exception as e_prom:
                    logger.warning("spark_func_promote_csv_from_temporary raised an exception: %s", e_prom)
                    logger.debug("Promotion traceback:\n%s", traceback.format_exc())

                # Adjust permissions on moved files (best-effort)
                try:
                    Process_Data_Service.adjust_file_permissions(output_path)
                except Exception as e_adj:
                    logger.warning("adjust_file_permissions failed after Spark coalesced write: %s", e_adj)

                logger.info("ATTENTION: Data saved successfully with correct file permissions (fast path).")
                return

        except Exception as fast_exc:
            logger.warning(
                "Primary Spark write (allow_multiple_files=%s) failed for output_path=%s. Falling back to chunked driver streaming writer. Fast-path error: %s",
                allow_multiple_files, output_path, fast_exc
            )
            logger.debug("Fast-path exception traceback:\n%s", traceback.format_exc())

        # ----------------------------
        #  Fallback: chunked streaming writer on driver (executor-side chunking & optional compression)
        # ----------------------------
        try:
            # Determine chunk_bytes using parse_size if available
            try:
                base_chunk_env = os.getenv("BASE_CHUNK_BYTES")
                if base_chunk_env:
                    env_val = parse_size(base_chunk_env)
                    if env_val:
                        chosen_chunk_bytes = env_val
                    else:
                        chosen_chunk_bytes = 128 * 1024
                else:
                    chosen_chunk_bytes = 256 * 1024
            except Exception:
                chosen_chunk_bytes = 256 * 1024

            logger.info("Using chunk_bytes=%d for chunked driver streaming", chosen_chunk_bytes)

            compress = True

            logger.info(
                "Attempting fallback writer: save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019(output_dir=%s, chunk_bytes=%d, compress=%s)",
                output_path, chosen_chunk_bytes, compress
            )

            Process_Data_Service.save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019(
                spark_df=spark_df,
                output_dir=output_path,
                spark=spark,
                final_filename=os.getenv("OUTPUT_FILENAME_FOR_AGG_NORMAL", "part-00000.csv"),
                compress=compress,
                chunk_bytes=chosen_chunk_bytes,
                progress_log_every_chunks=int(os.getenv("PROGRESS_LOG_EVERY_CHUNKS", "20"))
            )

            try:
                Process_Data_Service.adjust_file_permissions(output_path)
            except Exception as e_adj2:
                logger.warning("adjust_file_permissions failed after fallback writer: %s", e_adj2)

            logger.info("ATTENTION: Data saved successfully with correct file permissions (fallback streaming path).")
            return

        except Exception as fallback_exc:
            logger.exception(
                "Both fast Spark write and fallback streaming writer failed for output_path=%s. Fast-path exception: %s ; Fallback exception: %s",
                output_path, fast_exc, fallback_exc
            )
            raise

    
    def save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019(
        spark_df,
        output_dir: str,
        spark,
        final_filename: Optional[str] = None,
        compress: bool = False,
        chunk_bytes: int = 256 * 1024,  # <- REDUCED default chunk size: 256 KB
        progress_log_every_chunks: int = 10  # log every N chunks written
    ):
        """
        Stream a Spark DataFrame to a single CSV file from the driver, sending partitions as
        moderate-sized byte-chunks to avoid Netty 'Too large frame' errors and to keep driver memory low.

        Notes / behavior changes:
        - NO TRUNCATION: every row is preserved in full. If a single row > chunk_bytes, it will be
          emitted as its own chunk (may be large) but will not be truncated.
        - Executor-side chunk compression (gzip) remains optional; small chunks are left uncompressed.
        - Uses bytes across the network and writes binary on the driver.
        - Calls gc.collect() after yielding chunks on executors.
        """
        import os
        import gzip
        import logging
        import gc

        logger = logging.getLogger(__name__)

        if final_filename is None:
            final_filename = "part-00000.csv"
        if compress and not final_filename.endswith(".gz"):
            final_filename = final_filename + ".gz"

        # Ensure output dir exists
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.warning("Could not ensure output_dir exists (%s): %s", output_dir, e)

        tmp_file = os.path.join(output_dir, final_filename + ".part")
        final_file = os.path.join(output_dir, final_filename)

        logger.info(
            "save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019: streaming to %s (tmp=%s, compress=%s, chunk_bytes=%d)",
            final_file, tmp_file, compress, chunk_bytes
        )

        cols = spark_df.columns

        # Number of partitions for visibility
        try:
            num_partitions = spark_df.rdd.getNumPartitions()
            logger.info("save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019: input RDD num_partitions=%d", num_partitions)
        except Exception:
            num_partitions = None

        # Compression-level and thresholds via env
        try:
            gzip_level = int(os.getenv("CHUNK_GZIP_LEVEL", "1"))  # default fast, low memory
            if gzip_level < 1 or gzip_level > 9:
                gzip_level = 1
        except Exception:
            gzip_level = 1

        try:
            min_compress_bytes = int(os.getenv("MIN_COMPRESS_BYTES", str(1024)))  # don't compress tiny chunks
        except Exception:
            min_compress_bytes = 1024

        def partition_to_chunk_bytes(iterator):
            """
            Executor-side: build and yield chunk bytes (not strings).
            - Only compress if chunk_raw length >= min_compress_bytes.
            - Use gzip.compress(..., compresslevel=gzip_level) and fallback to raw if compressed bigger.
            - Call gc.collect() after yielding a chunk to free memory quickly on executor.
            - No truncation: if a single CSV line > chunk_bytes, yield it as its own chunk.
            """
            import gzip as _gzip
            import gc as _gc
            import logging as _logging

            _logger = _logging.getLogger(__name__)

            buf_lines = []
            buf_bytes = 0

            def serialize_cell(v):
                if v is None:
                    return ""
                if isinstance(v, (list, tuple)):
                    elems = []
                    for e in v:
                        if e is None:
                            elems.append("None")
                        else:
                            elems.append(str(e))
                    return "[" + ", ".join(elems) + "]"
                else:
                    return str(v)

            for row in iterator:
                try:
                    rowd = row.asDict()
                except Exception:
                    # fallback for Row object
                    rowd = {c: getattr(row, c) for c in cols}

                cell_strs = [serialize_cell(rowd.get(c)) for c in cols]
                line_str = ",".join(cell_strs) + "\n"
                encoded_line = line_str.encode("utf-8")
                line_len = len(encoded_line)

                # If adding this line would exceed the chunk_bytes and buffer not empty, yield current buffer
                if buf_lines and (buf_bytes + line_len > chunk_bytes):
                    chunk_raw = b"".join(buf_lines)
                    # Compress only if worth it and large enough
                    if compress and len(chunk_raw) >= min_compress_bytes:
                        try:
                            compressed = _gzip.compress(chunk_raw, compresslevel=gzip_level)
                            if len(compressed) < len(chunk_raw):
                                chunk_out = compressed
                            else:
                                chunk_out = chunk_raw
                        except Exception as ce:
                            _logger.warning("Executor gzip.compress failed, sending uncompressed chunk: %s", ce)
                            chunk_out = chunk_raw
                    else:
                        chunk_out = chunk_raw

                    yield chunk_out
                    # reset buffer
                    buf_lines = []
                    buf_bytes = 0

                    # hint GC on executor to free intermediate objects
                    try:
                        _gc.collect()
                    except Exception:
                        pass

                # If single line itself is larger than chunk_bytes and buffer empty -> send it as its own chunk (no truncation)
                if line_len > chunk_bytes and not buf_lines:
                    _logger.info("Executor: encountered single row larger than chunk_bytes (%d bytes). Sending as single chunk.", line_len)
                    if compress and line_len >= min_compress_bytes:
                        try:
                            compressed = _gzip.compress(encoded_line, compresslevel=gzip_level)
                            if len(compressed) < len(encoded_line):
                                chunk_out = compressed
                            else:
                                chunk_out = encoded_line
                        except Exception as ce:
                            _logger.warning("Executor gzip.compress failed for large single-line chunk, sending raw: %s", ce)
                            chunk_out = encoded_line
                    else:
                        chunk_out = encoded_line

                    yield chunk_out
                    try:
                        _gc.collect()
                    except Exception:
                        pass
                    # continue to next row
                    continue

                # Otherwise append to buffer
                buf_lines.append(encoded_line)
                buf_bytes += line_len

            # end for -> yield remainder
            if buf_lines:
                chunk_raw = b"".join(buf_lines)
                if compress and len(chunk_raw) >= min_compress_bytes:
                    try:
                        compressed = _gzip.compress(chunk_raw, compresslevel=gzip_level)
                        if len(compressed) < len(chunk_raw):
                            chunk_out = compressed
                        else:
                            chunk_out = chunk_raw
                    except Exception as ce:
                        _logger.warning("Executor gzip.compress failed for final chunk, sending raw: %s", ce)
                        chunk_out = chunk_raw
                else:
                    chunk_out = chunk_raw

                yield chunk_out
                try:
                    _gc.collect()
                except Exception:
                    pass

        # Build RDD of chunk-bytes by mapping partitions
        chunk_rdd = spark_df.rdd.mapPartitions(partition_to_chunk_bytes)

        total_chunks_written = 0

        try:
            # Write binary on driver. If compress=True the chunks are gzip members; concatenation is OK.
            with open(tmp_file, "wb") as out_f:
                # Write header as first member (compressed per policy)
                header_line = ",".join(cols) + "\n"
                header_bytes = header_line.encode("utf-8")
                if compress and len(header_bytes) >= min_compress_bytes:
                    try:
                        header_bytes_out = gzip.compress(header_bytes, compresslevel=gzip_level)
                        if len(header_bytes_out) < len(header_bytes):
                            out_f.write(header_bytes_out)
                        else:
                            out_f.write(header_bytes)
                    except Exception as e:
                        logger.warning("Driver-side gzip.compress failed for header; writing raw header: %s", e)
                        out_f.write(header_bytes)
                else:
                    out_f.write(header_bytes)

                # Iterate chunk bytes arriving from executors (streamed)
                for chunk in chunk_rdd.toLocalIterator():
                    # chunk is bytes (already compressed if compress=True)
                    if not isinstance(chunk, (bytes, bytearray)):
                        chunk = str(chunk).encode("utf-8")
                    out_f.write(chunk)
                    total_chunks_written += 1

                    if progress_log_every_chunks and (total_chunks_written % progress_log_every_chunks == 0):
                        logger.info(
                            "save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019: wrote %d chunks so far (last_chunk_size=%d bytes)",
                            total_chunks_written, len(chunk)
                        )

                out_f.flush()
                try:
                    os.fsync(out_f.fileno())
                except Exception:
                    pass

            # atomic rename
            os.replace(tmp_file, final_file)
            logger.info(
                "save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019: finished. chunks=%d -> %s",
                total_chunks_written, final_file
            )

            # adjust permissions
            try:
                Process_Data_Service.adjust_file_permissions(output_dir)
            except Exception as e:
                logger.warning("adjust_file_permissions failed after chunked streaming save: %s", e)

        except Exception as exc:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass
            logger.exception("save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019: failed while streaming to driver: %s", exc)
            raise



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
            .config("spark.sql.shuffle.partitions", "60")  # adjust as needed
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
            .config("spark.shuffle.compress", "true")
            .config("spark.rdd.compress", "true")
            .config("spark.shuffle.spill.compress", "true")
            .getOrCreate()
        )
        return spark
    
    def save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019(
        spark_df,
        output_dir: str,
        spark,
        num_buckets: int = 400,
        bucket_coalesce: bool = True,
        allow_bucket_fallback_to_chunked: bool = True,
        progress_log_every: int = 10,
        sample_for_bucket_size: bool = False,
        sample_fraction: float = 0.01,
        compress_parts: bool = True,              # NEW: if False, do not gzip promoted part files
        max_buckets_to_process: int = 0,  # NEW: if >0, limit number of buckets to process then stop
    ):
        """
        Write `spark_df` in multiple smaller files by splitting on a stable hash of EventIndex.

        Behavior notes:
        - For each bucket a part CSV is created and then compressed to part-{i:05d}.csv.gz.
        - Compression uses Python's gzip (streamed) with level from env PART_FILE_GZIP_LEVEL (1-9, default 9).
        - Returns list of produced compressed file paths.

        New parameters:
        - compress_parts: bool = True
          If False, the function will skip gzip compression and keep promoted part-{i:05d}.csv files as-is.
        - max_buckets_to_process: int = 0
          If > 0, only the first `max_buckets_to_process` buckets will be processed
          (remaining buckets are skipped) and execution will continue after the
          bucket loop as normal.
        """
        import os
        import shutil
        import glob
        import logging
        import traceback
        import time
        import gzip
        from math import ceil
        from pyspark.sql import functions as _F

        logger = logging.getLogger(__name__)
        logger.info("save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019: start (num_buckets=%d max_buckets_to_process=%d compress_parts=%s)", num_buckets, max_buckets_to_process, compress_parts)

        # ensure output dir exists
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create output_dir %s: %s", output_dir, e)

        if "EventIndex" not in spark_df.columns:
            raise ValueError("Input DataFrame must contain 'EventIndex' column")

        hash_col_expr = _F.abs(_F.hash(_F.col("EventIndex")))
        total_buckets = int(num_buckets)
        produced_files = []

        # Compression settings (streamed gzip)
        try:
            gzip_level = int(os.getenv("PART_FILE_GZIP_LEVEL", "9"))
            if gzip_level < 1 or gzip_level > 9:
                gzip_level = 9
        except Exception:
            gzip_level = 9

        # Optional lightweight bucket-size estimation (sample-based)
        bucket_row_estimates = None
        if sample_for_bucket_size:
            try:
                logger.info("Sampling %s fraction to estimate bucket sizes for ETA weighting", sample_fraction)
                sample_df = spark_df.sample(withReplacement=False, fraction=float(sample_fraction))
                sample_counts = (
                    sample_df.withColumn("_bucket", (hash_col_expr % _F.lit(total_buckets)))
                    .groupBy("_bucket")
                    .count()
                    .collect()
                )
                sample_map = {int(r["_bucket"]): int(r["count"]) for r in sample_counts}
                if sample_map:
                    bucket_row_estimates = sample_map
                    logger.info("Bucket size sampling collected (%d buckets sampled).", len(sample_map))
                else:
                    bucket_row_estimates = None
            except Exception as e:
                logger.warning("Sampling for bucket-size estimation failed (non-fatal): %s", e)
                bucket_row_estimates = None

        processed_buckets = 0
        start_time = time.perf_counter()
        bucket_durations = []  # list of per-bucket durations (seconds)

        # function to format seconds -> HH:MM:SS
        def _format_secs(secs):
            try:
                secs = int(max(0, round(secs)))
                h = secs // 3600
                m = (secs % 3600) // 60
                s = secs % 60
                return f"{h:02d}:{m:02d}:{s:02d}"
            except Exception:
                return str(secs)

        # helper: compress a file to .gz using streaming (low memory)
        def _compress_file_stream(src_path: str, compresslevel: int = 9) -> str:
            """
            Compress `src_path` to `src_path + '.gz'` using streaming copy; remove src_path on success.
            Returns path to compressed file on success; raises on fatal errors.
            """
            gz_path = src_path + ".gz"
            tmp_gz = gz_path + ".part"
            try:
                with open(src_path, "rb") as f_in, gzip.open(tmp_gz, "wb", compresslevel=compresslevel) as f_out:
                    shutil.copyfileobj(f_in, f_out, length=64 * 1024)
                # atomic replace
                try:
                    if os.path.exists(gz_path):
                        os.remove(gz_path)
                except Exception:
                    pass
                os.replace(tmp_gz, gz_path)
                # remove original file
                try:
                    os.remove(src_path)
                except Exception as e_rm:
                    logger.debug("Could not remove original file after compression %s: %s", src_path, e_rm)
                return gz_path
            except Exception as ce:
                # cleanup partial gz
                try:
                    if os.path.exists(tmp_gz):
                        os.remove(tmp_gz)
                except Exception:
                    pass
                raise

        # iterate buckets sequentially
        for i in range(total_buckets):
            bucket_num = i
            bucket_label = f"{i + 1}/{total_buckets}"
            bucket_start = time.perf_counter()
            logger.info("Bucket %s: starting (index=%d)", bucket_label, i)

            try:
                bucket_filter = (hash_col_expr % _F.lit(total_buckets)) == _F.lit(i)
                bucket_df = spark_df.filter(bucket_filter)

                # cheap emptiness check
                try:
                    is_empty = bucket_df.limit(1).count() == 0
                except Exception:
                    is_empty = False

                if is_empty:
                    processed_buckets += 1
                    bucket_elapsed = time.perf_counter() - bucket_start
                    bucket_durations.append(bucket_elapsed)
                    elapsed_total = time.perf_counter() - start_time
                    avg = sum(bucket_durations) / len(bucket_durations) if bucket_durations else 0.0
                    remaining = total_buckets - processed_buckets
                    eta = avg * remaining
                    logger.info(
                        "Bucket %s: empty -> skipped. Progress: %d/%d (%.2f%%). Avg bucket=%.2fs ETA=%s",
                        bucket_label, processed_buckets, total_buckets, (processed_buckets/total_buckets)*100.0,
                        avg, _format_secs(eta)
                    )
                    # check early-stop limit
                    if max_buckets_to_process and processed_buckets >= max_buckets_to_process:
                        logger.info("Reached processing limit %d, stopping early.", max_buckets_to_process)
                        break
                    continue

                # temp dir for bucket
                tmp_bucket_dir = os.path.join(output_dir, f"_bucket_tmp_{i:05d}")
                try:
                    if os.path.exists(tmp_bucket_dir):
                        shutil.rmtree(tmp_bucket_dir)
                    os.makedirs(tmp_bucket_dir, exist_ok=True)
                except Exception as e:
                    logger.warning("Bucket %s: could not prepare tmp dir %s: %s", bucket_label, tmp_bucket_dir, e)

                write_succeeded = False
                write_attempts = 0

                # Try per-bucket coalesce write (fast path for small bucket)
                if bucket_coalesce:
                    write_attempts += 1
                    try:
                        bucket_df.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_bucket_dir)
                        write_succeeded = True
                        logger.info("Bucket %s: coalesce(1).write succeeded to %s", bucket_label, tmp_bucket_dir)
                    except Exception as e:
                        logger.warning("Bucket %s: coalesce write failed: %s", bucket_label, e)
                        logger.debug("Bucket %s coalesce traceback:\n%s", bucket_label, traceback.format_exc())

                # If coalesce not attempted or failed, try native multi-file write on bucket
                if not write_succeeded:
                    write_attempts += 1
                    try:
                        bucket_df.write.mode("overwrite").option("header", True).csv(tmp_bucket_dir)
                        write_succeeded = True
                        logger.info("Bucket %s: native write succeeded to %s", bucket_label, tmp_bucket_dir)
                    except Exception as e:
                        logger.warning("Bucket %s: native write failed: %s", bucket_label, e)
                        logger.debug("Bucket %s native write traceback:\n%s", bucket_label, traceback.format_exc())

                # If still not succeeded and allowed, fallback to streaming single-file writer for this bucket
                if (not write_succeeded) and allow_bucket_fallback_to_chunked:
                    write_attempts += 1
                    try:
                        logger.info("Bucket %s: falling back to chunked driver streaming writer", bucket_label)
                        try:
                            small_repart = min(max(1, total_buckets // 8), 64)
                            bucket_df = bucket_df.repartition(small_repart)
                            logger.debug("Bucket %s: repartitioned to %d before streaming fallback", bucket_label, small_repart)
                        except Exception:
                            logger.debug("Bucket %s: repartition before streaming fallback failed (non-fatal)", bucket_label)

                        Process_Data_Service.save_spark_df_as_single_csv_on_driver_chunked_Pitsikalis_2019(
                            spark_df=bucket_df,
                            output_dir=tmp_bucket_dir,
                            spark=spark,
                            final_filename=f"part-{i:05d}.csv",
                            compress=False,
                            chunk_bytes=int(os.getenv("BASE_CHUNK_BYTES", str(128 * 1024))),
                            progress_log_every_chunks=int(os.getenv("PROGRESS_LOG_EVERY_CHUNKS", "20"))
                        )
                        write_succeeded = True
                        logger.info("Bucket %s: streaming fallback succeeded to %s", bucket_label, tmp_bucket_dir)
                    except Exception as e:
                        logger.exception("Bucket %s: streaming fallback failed: %s", bucket_label, e)
                        write_succeeded = False

                if not write_succeeded:
                    # final failure for this bucket -> mark processed and continue (do not abort entire job)
                    processed_buckets += 1
                    bucket_elapsed = time.perf_counter() - bucket_start
                    bucket_durations.append(bucket_elapsed)
                    remaining = total_buckets - processed_buckets
                    avg = sum(bucket_durations) / len(bucket_durations) if bucket_durations else 0.0
                    logger.error("Bucket %s: all write attempts failed (attempts=%d). Skipping bucket.", bucket_label, write_attempts)
                    logger.info("Progress: %d/%d (%.2f%%). Avg bucket=%.2fs ETA=%s", processed_buckets, total_buckets, (processed_buckets/total_buckets)*100.0, avg, _format_secs(avg * remaining))
                    # check early-stop limit
                    if max_buckets_to_process and processed_buckets >= max_buckets_to_process:
                        logger.info("Reached processing limit %d, stopping early.", max_buckets_to_process)
                        break
                    continue

                # Promote produced part file to canonical name part-{i:05d}.csv in output_dir
                try:
                    found = glob.glob(os.path.join(tmp_bucket_dir, "part-*.csv")) or glob.glob(os.path.join(tmp_bucket_dir, "**", "part-*.csv"), recursive=True)
                    promoted_dest = None
                    if not found:
                        # streaming-produced file scenario
                        streaming_file = os.path.join(tmp_bucket_dir, f"part-{i:05d}.csv")
                        if os.path.exists(streaming_file):
                            dest_path = os.path.join(output_dir, f"part-{i:05d}.csv")
                            shutil.move(streaming_file, dest_path)
                            promoted_dest = dest_path
                            logger.info("Bucket %s: moved streaming-produced file -> %s", bucket_label, dest_path)
                        else:
                            logger.warning("Bucket %s: no part-*.csv found in tmp dir %s (listing=%s).", bucket_label, tmp_bucket_dir, os.listdir(tmp_bucket_dir))
                    else:
                        src = found[0]
                        dest = os.path.join(output_dir, f"part-{i:05d}.csv")
                        try:
                            if os.path.exists(dest):
                                os.remove(dest)
                        except Exception:
                            pass
                        try:
                            shutil.move(src, dest)
                        except Exception:
                            # fallback copy & remove
                            try:
                                shutil.copy2(src, dest)
                                try:
                                    os.remove(src)
                                except Exception:
                                    pass
                            except Exception as e_copy:
                                logger.warning("Bucket %s: failed to move/copy part file %s -> %s: %s", bucket_label, src, dest, e_copy)
                                dest = None
                        if dest:
                            promoted_dest = dest
                            logger.info("Bucket %s: promoted %s -> %s", bucket_label, src, dest)

                    # If a destination was placed, compress it to .gz (streaming copy) unless compression disabled
                    if promoted_dest:
                        try:
                            if compress_parts:
                                logger.info("Bucket %s: compressing promoted file %s -> %s.gz (gzip_level=%d)", bucket_label, promoted_dest, promoted_dest, gzip_level)
                                gz_path = _compress_file_stream(promoted_dest, compresslevel=gzip_level)
                                produced_files.append(gz_path)
                                logger.info("Bucket %s: compression succeeded -> %s", bucket_label, gz_path)
                            else:
                                logger.info("Bucket %s: compression disabled, keeping promoted file %s", bucket_label, promoted_dest)
                                produced_files.append(promoted_dest)
                        except Exception as ce:
                            # If compression fails, keep the uncompressed file and include it in produced_files
                            logger.exception("Bucket %s: compression failed for %s: %s. Keeping uncompressed file.", bucket_label, promoted_dest, ce)
                            produced_files.append(promoted_dest)
                    else:
                        logger.warning("Bucket %s: nothing promoted for compression step.", bucket_label)

                    # cleanup tmp dir
                    try:
                        shutil.rmtree(tmp_bucket_dir, ignore_errors=True)
                    except Exception:
                        pass

                except Exception as e_prom:
                    logger.exception("Bucket %s: failed during promotion/compression: %s", bucket_label, e_prom)

                # Mark bucket completed and update ETA
                processed_buckets += 1
                bucket_elapsed = time.perf_counter() - bucket_start
                bucket_durations.append(bucket_elapsed)
                avg_bucket = sum(bucket_durations) / len(bucket_durations) if bucket_durations else 0.0
                remaining = total_buckets - processed_buckets
                eta_seconds = avg_bucket * remaining
                elapsed_total = time.perf_counter() - start_time

                # log progress + ETA
                logger.info(
                    "Bucket %s: finished. Progress %d/%d (%.2f%%). Produced files so far: %d. Bucket time=%.2fs Avg bucket=%.2fs ETA=%s Total elapsed=%.1fs",
                    bucket_label, processed_buckets, total_buckets, (processed_buckets/total_buckets)*100.0,
                    len(produced_files), bucket_elapsed, avg_bucket, _format_secs(eta_seconds), elapsed_total
                )

                # check early-stop limit
                if max_buckets_to_process and processed_buckets >= max_buckets_to_process:
                    logger.info("Reached processing limit %d, stopping early.", max_buckets_to_process)
                    break

                if processed_buckets % progress_log_every == 0:
                    logger.info("Overall progress: %d/%d buckets (%.2f%%). Avg bucket=%.2fs ETA=%s. Files produced=%d",
                                processed_buckets, total_buckets, (processed_buckets/total_buckets)*100.0,
                                avg_bucket, _format_secs(eta_seconds), len(produced_files))

            except Exception as e_outer:
                processed_buckets += 1
                bucket_elapsed = time.perf_counter() - bucket_start
                bucket_durations.append(bucket_elapsed)
                logger.exception("Unexpected error processing bucket %d: %s", i, e_outer)
                avg_bucket = sum(bucket_durations) / len(bucket_durations) if bucket_durations else 0.0
                eta_seconds = avg_bucket * (total_buckets - processed_buckets)
                logger.info("Progress after exception: %d/%d (%.2f%%). Avg bucket=%.2fs ETA=%s", processed_buckets, total_buckets, (processed_buckets/total_buckets)*100.0, avg_bucket, _format_secs(eta_seconds))
                # check early-stop limit
                if max_buckets_to_process and processed_buckets >= max_buckets_to_process:
                    logger.info("Reached processing limit %d, stopping early.", max_buckets_to_process)
                    break
                continue

        # final chmod/chown best-effort
        try:
            Process_Data_Service.adjust_file_permissions(output_dir)
        except Exception as e_adj:
            logger.warning("adjust_file_permissions failed after bucketed writes: %s", e_adj)

        total_elapsed = time.perf_counter() - start_time
        logger.info("save_spark_df_in_hash_partitions_and_promote_Pitsikalis_2019: finished. files_produced=%d buckets_processed=%d/%d total_elapsed=%.1fs",
                    len(produced_files), processed_buckets, total_buckets, total_elapsed)
        return produced_files


    ######################## END ########################

    ############################## PITSIKALIS 2019 DATA HELPERS  ##############################
        # (RECOGNIZED COMPOSITE EVENTS) for transshipment and non-transshipment #
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


    # def split_events_Pitsikalis_2019(events_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    #     """
    #     Split events DataFrame into:
    #     - transshipment_df: rows where Category == 'TRANSSHIPMENT'
    #     - non_transshipment_df: rows where Category in ('LOITERING', 'NORMAL', 'STOPPING')

    #     Returns (transshipment_df, non_transshipment_df) as Spark DataFrames.
    #     """
    #     transshipment_df = events_df.filter(F.col("Category") == F.lit("TRANSSHIPMENT"))
    #     non_transshipment_df = events_df.filter(F.col("Category").isin("LOITERING", "NORMAL", "STOPPING"))
    #     return transshipment_df, non_transshipment_df

    def split_events_Pitsikalis_2019(events_df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Split the input events Spark DataFrame into four DataFrames, one per category:
        - transshipment_df: rows where Category == 'TRANSSHIPMENT'
        - loitering_df:     rows where Category == 'LOITERING'
        - normal_df:        rows where Category == 'NORMAL'
        - stopping_df:      rows where Category == 'STOPPING'

        Returns:
            (transshipment_df, loitering_df, normal_df, stopping_df)

        Notes:
        - The function uses Spark filters and remains fully lazy (no actions performed).
        - Keep the returned DataFrames as-is so downstream code can cache/repartition as needed.
        """
        transshipment_df = events_df.filter(F.col("Category") == F.lit("TRANSSHIPMENT"))
        loitering_df = events_df.filter(F.col("Category") == F.lit("LOITERING"))
        normal_df = events_df.filter(F.col("Category") == F.lit("NORMAL"))
        stopping_df = events_df.filter(F.col("Category") == F.lit("STOPPING"))

        return transshipment_df, loitering_df, normal_df, stopping_df


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


    # def process_ais_events_Pitsikalis_2019(
    #     spark: SparkSession,
    #     ais_path: str,
    #     non_transshipment_df: DataFrame,
    #     output_base_dir: str = os.path.join("..", "datasets"),
    # ):
    #     """
    #     Process AIS CSV with Spark and join with non-transshipment events to produce
    #     loitering/non-loitering/stopping annotated AIS records.

    #     Steps:
    #     - load AIS csv into Spark DataFrame
    #     - normalize columns and rename 'Id' -> 'id'
    #     - convert AIS 'timestamp' (ms) into a proper timestamp column named 'timestamp' (timestamp type)
    #     - filter AIS rows by relevant MMSI set and global min/max event times to reduce data scanned
    #     - perform a join where (ais.id == events.MMSI) AND (ais.timestamp between events.T_start and events.T_end)
    #     - select and order the output columns:
    #     ['id','timestamp','longitude','latitude','annotation','speed','heading','turn','course','EventIndex','Category']
    #     Returns the Spark DataFrame with the joined results ("result_df").
    #     """
    #     # read AIS CSV
    #     ais_df = spark.read.option("header", True).option("inferSchema", True).csv(ais_path)

    #     # normalize column names
    #     ais_df = ais_df.toDF(*[c.strip() for c in ais_df.columns])

    #     # rename 'Id' -> 'id' if required
    #     if "Id" in ais_df.columns and "id" not in ais_df.columns:
    #         ais_df = ais_df.withColumnRenamed("Id", "id")

    #     # Convert timestamp in ms to timestamp type called 'timestamp'
    #     # If incoming timestamp already looks like epoch ms number, do conversion, otherwise try to cast to timestamp
    #     if "timestamp" in ais_df.columns:
    #         # create numeric_ts = timestamp / 1000 and then from_unixtime
    #         ais_df = ais_df.withColumn("timestamp_ms", F.col("timestamp"))
    #         # if timestamp is numeric (integer/long), convert from ms
    #         ais_df = ais_df.withColumn(
    #             "timestamp",
    #             F.when(
    #                 F.col("timestamp_ms").cast("long").isNotNull(),
    #                 F.from_unixtime((F.col("timestamp_ms").cast("double") / 1000.0)).cast(TimestampType()),
    #             ).otherwise(F.to_timestamp(F.col("timestamp_ms"))),
    #         ).drop("timestamp_ms")

    #     # Ensure required AIS columns exist; create null columns if absent so schema is consistent
    #     required_ais_cols = ["id", "timestamp", "longitude", "latitude", "annotation", "speed", "heading", "turn", "course"]
    #     for col in required_ais_cols:
    #         if col not in ais_df.columns:
    #             ais_df = ais_df.withColumn(col, F.lit(None))

    #     # Reduce events time window first (global min and max) to reduce AIS scanning
    #     aggs = non_transshipment_df.agg(F.min("T_start").alias("min_start"), F.max("T_end").alias("max_end")).collect()
    #     if aggs and len(aggs) > 0:
    #         min_start = aggs[0]["min_start"]
    #         max_end = aggs[0]["max_end"]
    #     else:
    #         min_start = None
    #         max_end = None

    #     # build relevant MMSI list (distinct)
    #     relevant_mmsi = [r[0] for r in non_transshipment_df.select("MMSI").distinct().rdd.map(lambda r: (r[0],)).collect()]

    #     # filter AIS by relevant MMSIs and time range (best-effort filter)
    #     ais_filtered = ais_df
    #     if relevant_mmsi:
    #         # to avoid very large IN lists in SQL, broadcast small list; if large list then use join below
    #         if len(relevant_mmsi) < 10_000:
    #             ais_filtered = ais_filtered.filter(F.col("id").isin(relevant_mmsi))
    #         else:
    #             # join approach: create df of relevant MMSIs
    #             mmsi_df = spark.createDataFrame([(m,) for m in relevant_mmsi], ["MMSI__tmp_"])
    #             ais_filtered = ais_filtered.join(mmsi_df, ais_filtered.id == F.col("MMSI__tmp_"), "inner").drop("MMSI__tmp_")

    #     if min_start is not None and max_end is not None:
    #         ais_filtered = ais_filtered.filter((F.col("timestamp") >= F.lit(min_start)) & (F.col("timestamp") <= F.lit(max_end)))

    #     # Join AIS with events using range condition:
    #     # condition: ais.id == events.MMSI AND ais.timestamp between events.T_start and events.T_end
    #     join_cond = (
    #         (ais_filtered.id == non_transshipment_df.MMSI)
    #         & (ais_filtered.timestamp >= non_transshipment_df.T_start)
    #         & (ais_filtered.timestamp <= non_transshipment_df.T_end)
    #     )

    #     joined = ais_filtered.join(non_transshipment_df, on=join_cond, how="inner")

    #     # Select output columns in the order expected
    #     output_columns = [
    #         "id",
    #         "timestamp",
    #         "longitude",
    #         "latitude",
    #         "annotation",
    #         "speed",
    #         "heading",
    #         "turn",
    #         "course",
    #         "EventIndex",
    #         "Category",
    #     ]

    #     # If EventIndex or Category are not present in events, ensure they exist (avoid exceptions)
    #     for c in ["EventIndex", "Category"]:
    #         if c not in joined.columns:
    #             joined = joined.withColumn(c, F.lit(None))

    #     result_df = joined.select(*output_columns)

    #     return result_df


    def process_ais_events_Pitsikalis_2019(
        spark: SparkSession,
        ais_path: str,
        loitering_df: DataFrame,
        normal_df: DataFrame,
        stopping_df: DataFrame,
        output_base_dir: str = os.path.join("..", "datasets"),
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Process AIS CSV with Spark and join with per-category events (loitering/normal/stopping)
        to produce annotated AIS records for each category.

        Parameters
        ----------
        spark : SparkSession
        ais_path : str
            Path to AIS CSV file.
        loitering_df, normal_df, stopping_df : DataFrame
            Spark DataFrames containing events filtered by Category already.
        output_base_dir : str
            Unused here except kept for compatibility (same default as original).

        Returns
        -------
        (loitering_result_df, normal_result_df, stopping_result_df)
            Three Spark DataFrames with the selected columns:
            ['id','timestamp','longitude','latitude','annotation','speed','heading','turn','course','EventIndex','Category']
        """
        # --- read AIS CSV ---
        ais_df = spark.read.option("header", True).option("inferSchema", True).csv(ais_path)

        # normalize column names
        ais_df = ais_df.toDF(*[c.strip() for c in ais_df.columns])

        # rename 'Id' -> 'id' if required
        if "Id" in ais_df.columns and "id" not in ais_df.columns:
            ais_df = ais_df.withColumnRenamed("Id", "id")

        # Convert timestamp in ms to timestamp type called 'timestamp'
        if "timestamp" in ais_df.columns:
            ais_df = ais_df.withColumn("timestamp_ms", F.col("timestamp"))
            # convert numeric ms -> seconds -> timestamp; fallback to to_timestamp otherwise
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

        # small helper: perform the same join/filter logic for a given event DataFrame
        def join_for_event_type(event_df: DataFrame) -> DataFrame:
            """
            Join ais_df with a specific event_df by MMSI and timestamp range and select output columns.
            Returns a Spark DataFrame with the expected output columns.
            """
            # compute global min/max for the event_df (to reduce AIS scan)
            aggs = event_df.agg(F.min("T_start").alias("min_start"), F.max("T_end").alias("max_end")).collect()
            if aggs and len(aggs) > 0:
                min_start = aggs[0]["min_start"]
                max_end = aggs[0]["max_end"]
            else:
                min_start = None
                max_end = None

            # build relevant MMSI list (distinct)
            # Note: this collects into driver memory; keep the same heuristic as original code.
            relevant_mmsi = [r[0] for r in event_df.select("MMSI").distinct().rdd.map(lambda r: (r[0],)).collect()]

            # filter AIS by relevant MMSIs and time range (best-effort filter)
            ais_filtered = ais_df
            if relevant_mmsi:
                if len(relevant_mmsi) < 10_000:
                    ais_filtered = ais_filtered.filter(F.col("id").isin(relevant_mmsi))
                else:
                    mmsi_df = spark.createDataFrame([(m,) for m in relevant_mmsi], ["MMSI__tmp_"])
                    ais_filtered = ais_filtered.join(mmsi_df, ais_filtered.id == F.col("MMSI__tmp_"), "inner").drop("MMSI__tmp_")

            if min_start is not None and max_end is not None:
                ais_filtered = ais_filtered.filter((F.col("timestamp") >= F.lit(min_start)) & (F.col("timestamp") <= F.lit(max_end)))

            # Join AIS with events using range condition:
            join_cond = (
                (ais_filtered.id == event_df.MMSI)
                & (ais_filtered.timestamp >= event_df.T_start)
                & (ais_filtered.timestamp <= event_df.T_end)
            )

            joined = ais_filtered.join(event_df, on=join_cond, how="inner")

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

            # Ensure EventIndex and Category exist in joined
            for c in ["EventIndex", "Category"]:
                if c not in joined.columns:
                    joined = joined.withColumn(c, F.lit(None))

            result_df = joined.select(*output_columns)

            return result_df

        # Apply helper for each event-type DataFrame
        loitering_result_df = join_for_event_type(loitering_df)
        normal_result_df = join_for_event_type(normal_df)
        stopping_result_df = join_for_event_type(stopping_df)

        return loitering_result_df, normal_result_df, stopping_result_df

    ################ AGGREGATION FUNCTIONS FOR PREPROCESSED AIS DATA FROM PITSIKALIS 2019 ################
        ## [OPTIONAL] BASIC DATA FOR TESTING BASIC MACHINE LEARNING (LOITERING/NON-LOITERING/STOPPING) ##
    
    # ---------- Helper: build grouped points ----------
    def build_grouped_points_Pitsikalis_2019(df: DataFrame) -> DataFrame:
        """
        Group input AIS Spark DataFrame by EventIndex and produce an array-of-structs 'points'
        sorted by timestamp.

        The resulting DataFrame has columns:
        - EventIndex
        - mmsi (first id)
        - Category (first Category)
        - points: array<struct(ts: long, lat: double, lon: double, sog: double, cog: double)>

        Timestamps attempt to be interpreted as epoch ms when numeric; otherwise unix_timestamp() is used.
        """
        if "timestamp" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'timestamp' column")

        # create epoch-second ts column using heuristic: numeric -> ms -> seconds, else unix_timestamp
        df_ts = df.withColumn(
            "_ts_seconds",
            F.when(
                F.col("timestamp").cast("double").isNotNull() & F.col("timestamp").rlike("^[0-9]+$"),
                (F.col("timestamp").cast("double") / F.lit(1000.0)).cast("long"),
            ).otherwise(F.unix_timestamp(F.col("timestamp")).cast("long")),
        )

        # point struct
        # note: use the raw columns while building the struct via SQL expression below for stable compatibility
        grouped = (
            df_ts
            .groupBy("EventIndex")
            .agg(
                F.first(F.col("id")).alias("mmsi"),
                F.first(F.col("Category")).alias("Category"),
                F.expr(
                    "array_sort(collect_list(struct(_ts_seconds as ts, latitude as lat, longitude as lon, speed as sog, heading as cog)), (x, y) -> CASE WHEN x.ts < y.ts THEN -1 WHEN x.ts > y.ts THEN 1 ELSE 0 END)"
                ).alias("points")
            )
        )

        return grouped


    # ---------- Pure-Spark haversine aggregator expression builder ----------
    def _spark_haversine_distance_expr_for_coord_pairs() -> str:
        """
        Build a Spark SQL expression string that computes the sum of haversine distances
        for an array column named 'coord_pairs' where each element has fields lat1, lon1, lat2, lon2.

        Returns a SQL expression string. This is used via F.expr(...) to produce a Double column.
        """
        # Earth radius in kilometers
        R = 6371.0088

        # Use an explicitly-typed accumulator cast(0.0 as double)
        expr = f"""
            aggregate(
                transform(coord_pairs, p ->
                    2 * ASIN(
                        LEAST(1.0,
                            SQRT(
                                POWER(SIN(RADIANS(p.lat2 - p.lat1) / 2), 2)
                                + COS(RADIANS(p.lat1)) * COS(RADIANS(p.lat2)) * POWER(SIN(RADIANS(p.lon2 - p.lon1) / 2), 2)
                            )
                        )
                    ) * {R}
                ),
                cast(0.0 as double),
                (acc, x) -> acc + x
            )
        """
        return " ".join(expr.split())


    # ---------- Compute event-level metrics using Spark ----------
    def compute_event_metrics_spark_Pitsikalis_2019(grouped: DataFrame) -> DataFrame:
        """
        Given a grouped DataFrame (output of build_grouped_points_Pitsikalis_2019),
        compute the derived columns using Spark higher-order functions:

        - timestamp_array (array of ISO strings "YYYY-MM-DD HH:MM:SS")
        - average_time_diff_between_consecutive_points (seconds, float)
        - sog_array, cog_array (arrays of doubles)
        - average_sog, min_sog, max_sog, standard_deviation_sog (population std)
        - average_cog, min_cog, max_cog, standard_deviation_cog (population std)
        - trajectory (LINESTRING string using "lon lat" pairs)
        - coord_pairs (array of structs lat1,lon1,lat2,lon2) for distance calculations
        - distance_in_kilometers computed by a pure-Spark haversine aggregator

        Returns:
        - DataFrame with final event-level columns similar to original pandas output.
        """
        # 1) ts array and ISO formatted timestamp array
        grouped2 = grouped.withColumn("ts_array", F.expr("transform(points, x -> x.ts)"))

        grouped2 = grouped2.withColumn(
            "timestamp_array",
            F.expr("transform(ts_array, t -> CASE WHEN t IS NULL THEN NULL ELSE from_unixtime(t, 'yyyy-MM-dd HH:mm:ss') END)")
        )

        # 2) time_diffs & average
        grouped2 = grouped2.withColumn(
            "time_diffs",
            F.expr(
                "CASE WHEN size(ts_array) <= 1 THEN array() "
                "ELSE transform(sequence(2, size(ts_array)), i -> element_at(ts_array, i) - element_at(ts_array, i-1)) END"
            )
        )

        grouped2 = grouped2.withColumn(
            "average_time_diff_between_consecutive_points",
            F.expr(
                "CASE WHEN size(time_diffs) = 0 THEN 0.0 ELSE (aggregate(time_diffs, cast(0 as bigint), (acc, x) -> acc + x) / size(time_diffs)) END"
            ).cast(DoubleType())
        )

        # 3) SOG / COG arrays (force double cast to avoid datatype mismatches)
        grouped2 = grouped2.withColumn("sog_array", F.expr("transform(points, x -> cast(x.sog as double))"))
        grouped2 = grouped2.withColumn("cog_array", F.expr("transform(points, x -> cast(x.cog as double))"))

        # 4) SOG stats (population stddev)
        # Use explicitly typed accumulator cast(0.0 as double) to avoid datatype mismatch
        grouped2 = grouped2.withColumn(
            "sum_sog",
            F.expr("aggregate(sog_array, cast(0.0 as double), (acc, x) -> acc + coalesce(x, cast(0.0 as double)))")
        )
        grouped2 = grouped2.withColumn(
            "sumsq_sog",
            F.expr("aggregate(sog_array, cast(0.0 as double), (acc, x) -> acc + coalesce(x, cast(0.0 as double)) * coalesce(x, cast(0.0 as double)))")
        )
        grouped2 = grouped2.withColumn(
            "count_sog",
            F.expr("aggregate(sog_array, cast(0 as int), (acc, x) -> acc + CASE WHEN x IS NULL THEN 0 ELSE 1 END)")
        )

        grouped2 = grouped2.withColumn(
            "average_sog",
            F.expr("CASE WHEN count_sog = 0 THEN NULL ELSE sum_sog / count_sog END").cast(DoubleType())
        )
        grouped2 = grouped2.withColumn("min_sog", F.expr("array_min(sog_array)").cast(DoubleType()))
        grouped2 = grouped2.withColumn("max_sog", F.expr("array_max(sog_array)").cast(DoubleType()))
        grouped2 = grouped2.withColumn(
            "standard_deviation_sog",
            F.expr("CASE WHEN count_sog = 0 THEN NULL ELSE sqrt( (sumsq_sog / count_sog) - POWER(sum_sog / count_sog, 2) ) END").cast(DoubleType())
        )

        # 5) COG stats (same treatment)
        grouped2 = grouped2.withColumn(
            "sum_cog",
            F.expr("aggregate(cog_array, cast(0.0 as double), (acc, x) -> acc + coalesce(x, cast(0.0 as double)))")
        )
        grouped2 = grouped2.withColumn(
            "sumsq_cog",
            F.expr("aggregate(cog_array, cast(0.0 as double), (acc, x) -> acc + coalesce(x, cast(0.0 as double)) * coalesce(x, cast(0.0 as double)))")
        )
        grouped2 = grouped2.withColumn(
            "count_cog",
            F.expr("aggregate(cog_array, cast(0 as int), (acc, x) -> acc + CASE WHEN x IS NULL THEN 0 ELSE 1 END)")
        )

        grouped2 = grouped2.withColumn(
            "average_cog",
            F.expr("CASE WHEN count_cog = 0 THEN NULL ELSE sum_cog / count_cog END").cast(DoubleType())
        )
        grouped2 = grouped2.withColumn("min_cog", F.expr("array_min(cog_array)").cast(DoubleType()))
        grouped2 = grouped2.withColumn("max_cog", F.expr("array_max(cog_array)").cast(DoubleType()))
        grouped2 = grouped2.withColumn(
            "standard_deviation_cog",
            F.expr("CASE WHEN count_cog = 0 THEN NULL ELSE sqrt( (sumsq_cog / count_cog) - POWER(sum_cog / count_cog, 2) ) END").cast(DoubleType())
        )

        # 6) trajectory LINESTRING
        grouped2 = grouped2.withColumn(
            "trajectory",
            F.expr("concat('LINESTRING(', array_join(transform(points, x -> concat(cast(x.lon as string), ' ', cast(x.lat as string))), ', '), ')')")
        )

        # 7) coord_pairs array for distance calculation
        grouped2 = grouped2.withColumn(
            "coord_pairs",
            F.expr(
                "CASE WHEN size(points) <= 1 THEN array() ELSE transform(sequence(1, size(points)-1), i -> struct( element_at(points, i).lat as lat1, element_at(points, i).lon as lon1, element_at(points, i+1).lat as lat2, element_at(points, i+1).lon as lon2 )) END"
            )
        )

        # 8) compute Spark haversine sum (pure Spark; fast on executors)
        haversine_expr = Process_Data_Service._spark_haversine_distance_expr_for_coord_pairs()
        grouped2 = grouped2.withColumn("distance_in_kilometers", F.expr(haversine_expr).cast(DoubleType()))

        # 9) convert timestamp_array (ARRAY<STRING>) to a STRING that looks exactly like Python's list
        # Format: ['YYYY-MM-DD HH:MM:SS', None, 'YYYY-...']
        # Use a CASE to render NULL elements as the bare word None (no quotes) and present elements with single quotes.
        ts_array_to_str_expr = (
            "concat('[', "
            "array_join(transform(timestamp_array, t -> CASE WHEN t IS NULL THEN 'None' ELSE concat(\"'\", t, \"'\") END), ', '), "
            "']')"
        )
        grouped2 = grouped2.withColumn("timestamp_array_str", F.expr(ts_array_to_str_expr))

        # 10) convert sog_array (ARRAY<DOUBLE>) to STRING like [1.23, None, 4.56]
        sog_array_to_str_expr = (
            "concat('[', "
            "array_join(transform(sog_array, x -> CASE WHEN x IS NULL THEN 'None' ELSE cast(x as string) END), ', '), "
            "']')"
        )
        grouped2 = grouped2.withColumn("sog_array_str", F.expr(sog_array_to_str_expr))

        # 11) convert cog_array (ARRAY<DOUBLE>) to STRING like [12.3, None, 45.6]
        cog_array_to_str_expr = (
            "concat('[', "
            "array_join(transform(cog_array, x -> CASE WHEN x IS NULL THEN 'None' ELSE cast(x as string) END), ', '), "
            "']')"
        )
        grouped2 = grouped2.withColumn("cog_array_str", F.expr(cog_array_to_str_expr))

        # 12) Select final columns keeping original numeric semantics/column names
        result = grouped2.select(
            F.col("mmsi").alias("mmsi"),
            F.col("distance_in_kilometers").alias("distance_in_kilometers"),
            F.col("EventIndex").alias("EventIndex"),
            F.col("trajectory").alias("trajectory"),
            F.col("timestamp_array_str").alias("timestamp_array"),  # <-- stringified version used here
            F.col("average_time_diff_between_consecutive_points").alias("average_time_diff_between_consecutive_points"),
            F.col("sog_array_str").alias("sog_array"),  # stringified
            F.col("cog_array_str").alias("cog_array"),  # stringified
            F.col("average_sog").alias("average_sog"),
            F.col("min_sog").alias("min_sog"),
            F.col("max_sog").alias("max_sog"),
            F.col("standard_deviation_sog").alias("standard_deviation_sog"),
            F.col("average_cog").alias("average_cog"),
            F.col("min_cog").alias("min_cog"),
            F.col("max_cog").alias("max_cog"),
            F.col("standard_deviation_cog").alias("standard_deviation_cog"),
            F.col("Category").alias("behavior_type_vector"),
        )

        return result

    # ---------- Top-level orchestrator ----------
    def OPTIONAL_MUST_BE_SKIPPED_convert_to_vessel_events_Pitsikalis_2019(df: DataFrame, spark: SparkSession) -> DataFrame:
        """
        Orchestrates conversion of an AIS Spark DataFrame to vessel-event summaries grouped by EventIndex.

        Steps:
        - logs partition counts and uses mapPartitionsWithIndex to emit per-partition start/finish messages
        - aggregates points per EventIndex (ordered by timestamp) using pure Spark
        - computes event-level metrics (timestamp arrays, average time diffs, SOG/COG arrays & stats, trajectory)
        - computes total distance using a pure-Spark haversine aggregator
        - logs progress and sample outputs for debugging

        Returns:
        - the final Spark DataFrame (pre-save view) for further use.
        """
        # initial stats
        try:
            total_rows = df.count()
        except Exception:
            total_rows = None
        logger.info(f"convert_to_vessel_events_Pitsikalis_2019: starting pipeline. input rows={total_rows}")

        # log partition counts for progress insight
        try:
            partition_count = df.rdd.getNumPartitions()
            logger.info(f"convert_to_vessel_events_Pitsikalis_2019: input DataFrame partitions = {partition_count}")
        except Exception:
            logger.debug("convert_to_vessel_events_Pitsikalis_2019: could not get partition count")

        # run partition-level logger to emit start/finish for each partition
        #Process_Data_Service.log_progress_partitions_Pitsikalis_2019(df)

        # aggregate points per EventIndex
        logger.info("convert_to_vessel_events_Pitsikalis_2019: aggregating points per EventIndex")
        grouped = Process_Data_Service.build_grouped_points_Pitsikalis_2019(df)

        # grouped count logging
        try:
            grouped_count = grouped.count()
        except Exception:
            grouped_count = None
        logger.info(f"convert_to_vessel_events_Pitsikalis_2019: grouped EventIndex count = {grouped_count}")

        # log partitions on grouped DataFrame as well
        try:
            grouped_partitions = grouped.rdd.getNumPartitions()
            logger.info(f"convert_to_vessel_events_Pitsikalis_2019: grouped DataFrame partitions = {grouped_partitions}")
            #Process_Data_Service.log_progress_partitions_Pitsikalis_2019(grouped)
        except Exception:
            logger.debug("convert_to_vessel_events_Pitsikalis_2019: failed grouped partition logging")

        # compute metrics (pure Spark heavy-lifting)
        logger.info("convert_to_vessel_events_Pitsikalis_2019: computing event metrics (Spark)")
        metrics_df = Process_Data_Service.compute_event_metrics_spark_Pitsikalis_2019(grouped)

        # show a sample (safe attempt)
        try:
            sample_row = metrics_df.limit(1).collect()
            if sample_row:
                logger.info("convert_to_vessel_events_Pitsikalis_2019: sample row for debugging:\n%s", [row.asDict() for row in sample_row])
            else:
                logger.info("convert_to_vessel_events_Pitsikalis_2019: no sample row available")
        except Exception:
            logger.debug("convert_to_vessel_events_Pitsikalis_2019: could not fetch sample row")

        # final counts log
        try:
            final_count = metrics_df.count()
        except Exception:
            final_count = None
        logger.info(f"convert_to_vessel_events_Pitsikalis_2019: finished. output event rows={final_count}")

        # # optionally log partition counts of final df
        # try:
        #     #Process_Data_Service.log_progress_partitions_Pitsikalis_2019(metrics_df)
        # except Exception:
        #     logger.debug("convert_to_vessel_events_Pitsikalis_2019: final partition logging failed")

        return metrics_df

        #### WARNING: UNTESTED!!! ####
    


        #### WARNING: UNTESTED!!! ####
    
    def create_aggregated_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
        events_df: DataFrame,
        ais_path: str,
        spark: SparkSession,
    ) -> DataFrame:
        """
        Pure-Spark replacement for the pandas `create_aux_dataframe_with_chunks`.

        Steps:
        - normalizes and explodes events' MMSI / MMSI_2 into one 'mmsi' column
        - reads AIS CSV with Spark and casts fields
        - joins AIS points to events by mmsi and timestamp window
        - groups points into ordered arrays per (EventIndex, mmsi)
        - computes the same metrics as the pandas implementation:
            trajectory, timestamp_array (array[str] -> then stringified),
            sog_array (array[double] -> then stringified),
            cog_array (array[double] -> then stringified),
            average_speed, min_speed, max_speed, average_heading, std_dev_heading,
            total_area_time, low_speed_percentage, stagnation_time
        - writes the final CSV using Process_Data_Service.save_spark_df_as_csv which performs coalesce(1)
            and subsequent promotion/permission adjustments.

        Parameters
        ----------
        events_df : DataFrame
            Spark DataFrame with columns ['EventIndex', 'T_start', 'T_end', 'Category', 'MMSI', 'MMSI_2'].
        ais_path : str
            Path to AIS CSV (accessible by Spark).
        spark : SparkSession
        output_dir : str
            Directory to write final CSV (passed to Process_Data_Service.save_spark_df_as_csv).

        Returns
        -------
        DataFrame
            The final Spark DataFrame (with array columns replaced by their stringified forms).
        """
        logger.info("create_aggregated_dataframe_with_spark_Pitsikalis_2019: start")

        # 1) Normalize events timestamps and explode MMSI/MMSI_2 into single 'mmsi'
        events = (
            events_df
            .withColumn("T_start", F.to_timestamp(F.col("T_start")))
            .withColumn("T_end", F.to_timestamp(F.col("T_end")))
        )

        events_exp = events.withColumn(
            "mmsi_array",
            F.array(F.col("MMSI").cast(StringType()), F.col("MMSI_2").cast(StringType()))
        ).withColumn(
            "mmsi",
            F.explode(F.expr("filter(mmsi_array, x -> x is not null)"))
        ).drop("mmsi_array")

        events_exp = events_exp.select("EventIndex", "T_start", "T_end", "Category", "mmsi")

        # 2) Read AIS CSV
        logger.info("create_aggregated_dataframe_with_spark_Pitsikalis_2019: reading AIS CSV from %s", ais_path)
        ais = (
            spark.read.option("header", True).option("inferSchema", True).csv(ais_path)
            .withColumnRenamed("Id", "id")
        )
        ais = ais.toDF(*[c.strip() for c in ais.columns])
        ais = ais.withColumn("id", F.col("id").cast(StringType()))

        # timestamp heuristic: numeric => already ms; else unix_timestamp * 1000 -> milliseconds
        ais = ais.withColumn(
            "_ts_millis",
            F.when(
                (F.col("timestamp").cast("double").isNotNull()) & F.col("timestamp").rlike("^[0-9]+$"),
                F.col("timestamp").cast("long"),
            ).otherwise((F.unix_timestamp(F.col("timestamp").cast(StringType())).cast("long") * F.lit(1000)))
        ).withColumn("latitude", F.col("latitude").cast(DoubleType())) \
        .withColumn("longitude", F.col("longitude").cast(DoubleType())) \
        .withColumn("speed", F.col("speed").cast(DoubleType())) \
        .withColumn("heading", F.col("heading").cast(DoubleType()))

        try:
            ais_parts = ais.rdd.getNumPartitions()
            logger.info("create_aggregated_dataframe_with_spark_Pitsikalis_2019: AIS partitions = %s", ais_parts)
        except Exception:
            logger.debug("could not obtain ais partitions")

        # 3) Join AIS → events by mmsi and timestamp window
        # use milliseconds for event bounds to match AIS _ts_millis
        events_bounds = events_exp.withColumn(
            "t_start_ms",
            (F.unix_timestamp(F.col("T_start")).cast(LongType()) * F.lit(1000)).cast(LongType()),
        ).withColumn(
            "t_end_ms",
            (F.unix_timestamp(F.col("T_end")).cast(LongType()) * F.lit(1000)).cast(LongType()),
        )

        join_cond = (
            (ais.id == events_bounds.mmsi) &
            (ais._ts_millis >= events_bounds.t_start_ms) &
            (ais._ts_millis <= events_bounds.t_end_ms)
        )

        logger.info("create_aggregated_dataframe_with_spark_Pitsikalis_2019: performing join (may shuffle)")
        joined = ais.join(events_bounds, on=join_cond, how="inner").select(
            events_bounds.EventIndex.alias("EventIndex"),
            events_bounds.mmsi.alias("mmsi"),
            events_bounds.Category.alias("Category"),
            ais._ts_millis.alias("ts"),
            ais.longitude.alias("lon"),
            ais.latitude.alias("lat"),
            ais.speed.alias("sog"),
            ais.heading.alias("cog")
        )

        # 4) Group into ordered 'points' array per EventIndex,mmsi
        logger.info("create_aggregated_dataframe_with_spark_Pitsikalis_2019: grouping points per (EventIndex, mmsi)")
        grouped = joined.groupBy("EventIndex", "mmsi", "Category").agg(
            F.expr(
                "array_sort(collect_list(struct(ts as ts, lat as lat, lon as lon, sog as sog, cog as cog)),"
                " (x, y) -> CASE WHEN x.ts < y.ts THEN -1 WHEN x.ts > y.ts THEN 1 ELSE 0 END)"
            ).alias("points")
        )

        # 5) Compute arrays and stats from points
        df = grouped.withColumn("ts_array", F.expr("transform(points, x -> x.ts)")) \
            .withColumn("timestamp_array", F.expr("transform(ts_array, t -> CASE WHEN t IS NULL THEN NULL ELSE from_unixtime(cast(floor(t/1000) as bigint), 'yyyy-MM-dd HH:mm:ss') END)")) \
            .withColumn("sog_array", F.expr("transform(points, x -> cast(x.sog as double))")) \
            .withColumn("cog_array", F.expr("transform(points, x -> cast(x.cog as double))"))

        # NEW: lat/lon arrays for distance computation (haversine)
        df = df.withColumn("lat_array", F.expr("transform(points, x -> cast(x.lat as double))")) \
               .withColumn("lon_array", F.expr("transform(points, x -> cast(x.lon as double))"))

        # time diffs array (milliseconds)
        df = df.withColumn(
            "time_diffs",
            F.expr(
                "CASE WHEN size(ts_array) <= 1 THEN array() "
                "ELSE transform(sequence(2, size(ts_array)), i -> element_at(ts_array, i) - element_at(ts_array, i-1)) END"
            )
        )

        # total_area_time
        df = df.withColumn("total_area_time", F.expr("aggregate(time_diffs, cast(0.0 as double), (acc,x) -> acc + x)"))

        # SOG statistics
        df = df.withColumn("sum_sog", F.expr("aggregate(sog_array, cast(0.0 as double), (acc,x) -> acc + coalesce(x, cast(0.0 as double)))")) \
            .withColumn("sumsq_sog", F.expr("aggregate(sog_array, cast(0.0 as double), (acc,x) -> acc + coalesce(x, cast(0.0 as double)) * coalesce(x, cast(0.0 as double)))")) \
            .withColumn("count_sog", F.expr("aggregate(sog_array, cast(0 as int), (acc,x) -> acc + CASE WHEN x IS NULL THEN 0 ELSE 1 END)"))

        df = df.withColumn("average_speed", F.expr("CASE WHEN count_sog = 0 THEN 0.0 ELSE sum_sog / count_sog END").cast(DoubleType())) \
            .withColumn("min_speed", F.expr("array_min(sog_array)").cast(DoubleType())) \
            .withColumn("max_speed", F.expr("array_max(sog_array)").cast(DoubleType())) \
            .withColumn("std_dev_speed", F.expr("CASE WHEN count_sog = 0 THEN 0.0 ELSE sqrt((sumsq_sog / count_sog) - POWER(sum_sog / count_sog, 2)) END").cast(DoubleType()))

        # COG statistics
        df = df.withColumn("sum_cog", F.expr("aggregate(cog_array, cast(0.0 as double), (acc,x) -> acc + coalesce(x, cast(0.0 as double)))")) \
            .withColumn("sumsq_cog", F.expr("aggregate(cog_array, cast(0.0 as double), (acc,x) -> acc + coalesce(x, cast(0.0 as double)) * coalesce(x, cast(0.0 as double)))")) \
            .withColumn("count_cog", F.expr("aggregate(cog_array, cast(0 as int), (acc,x) -> acc + CASE WHEN x IS NULL THEN 0 ELSE 1 END)"))

        df = df.withColumn("average_heading", F.expr("CASE WHEN count_cog = 0 THEN 0.0 ELSE sum_cog / count_cog END").cast(DoubleType())) \
            .withColumn("min_heading", F.expr("array_min(cog_array)").cast(DoubleType())) \
            .withColumn("max_heading", F.expr("array_max(cog_array)").cast(DoubleType())) \
            .withColumn("std_dev_heading", F.expr("CASE WHEN count_cog = 0 THEN 0.0 ELSE sqrt((sumsq_cog / count_cog) - POWER(sum_cog / count_cog, 2)) END").cast(DoubleType()))

        # 6) Build trajectory as LINESTRING(lon lat, ...)
        df = df.withColumn(
            "trajectory",
            F.expr("concat('LINESTRING(', array_join(transform(points, x -> concat(cast(x.lon as string), ' ', cast(x.lat as string))), ', '), ')')")
        )

        # 7) low_speed_percentage and stagnation_time
        df = df.withColumn("count_low_speed_2", F.expr("aggregate(sog_array, cast(0 as int), (acc,x) -> acc + CASE WHEN x < 2.0 THEN 1 ELSE 0 END)")) \
            .withColumn("count_low_speed_0_5", F.expr("aggregate(sog_array, cast(0 as int), (acc,x) -> acc + CASE WHEN x < 0.5 THEN 1 ELSE 0 END)")) \
            .withColumn("count_sog_total", F.expr("size(sog_array)"))

        df = df.withColumn(
            "low_speed_percentage",
            F.expr("CASE WHEN count_sog_total = 0 THEN 0.0 ELSE (count_low_speed_2 / cast(count_sog_total as double) * 100.0) END").cast(DoubleType())
        )

        df = df.withColumn(
            "stagnation_time",
            F.expr(
                "CASE WHEN count_sog_total = 0 OR size(time_diffs) = 0 THEN 0.0 ELSE count_low_speed_0_5 * (CASE WHEN total_area_time IS NULL THEN 0.0 ELSE total_area_time END) / cast(count_sog_total as double) END"
            ).cast(DoubleType())
        )

        # NEW: distance_in_kilometers using haversine (Spark-only math)
        # Uses element_at(lat_array, i) 1-based indexing and sums distances between consecutive pairs
        df = df.withColumn(
            "distance_in_kilometers",
            F.expr(
                "CASE WHEN size(lat_array) <= 1 THEN 0.0 ELSE aggregate(sequence(2, size(lat_array)), cast(0.0 as double), (acc, i) -> acc + ("
                "2 * 6371.0 * asin( sqrt( pow( sin( (radians(element_at(lat_array, i)) - radians(element_at(lat_array, i-1))) / 2 ), 2 ) "
                "+ cos(radians(element_at(lat_array, i-1))) * cos(radians(element_at(lat_array, i))) * pow( sin( (radians(element_at(lon_array, i)) - radians(element_at(lon_array, i-1))) / 2 ), 2 ) ) ) ) ) END"
            ).cast(DoubleType())
        )

        # NEW: average_time_diff_between_consecutive_points (milliseconds)
        df = df.withColumn(
            "average_time_diff_between_consecutive_points",
            F.expr(
                "CASE WHEN size(time_diffs) = 0 THEN 0.0 ELSE aggregate(time_diffs, cast(0.0 as double), (acc,x) -> acc + x) / cast(size(time_diffs) as double) END"
            ).cast(DoubleType())
        )

        # 8) Stringify array columns so CSV datasource accepts them:
        # timestamp_array -> [YYYY-MM-DD HH:MM:SS, None, ...]  (no quotes around timestamps)
        ts_array_to_str_expr = (
            "concat('[', array_join(transform(timestamp_array, t -> CASE WHEN t IS NULL THEN 'None' ELSE t END), ', '), ']')"
        )
        df = df.withColumn("timestamp_array_str", F.expr(ts_array_to_str_expr))

        # sog_array and cog_array stringification: numbers as strings, nulls as None
        sog_array_to_str_expr = (
            "concat('[', array_join(transform(sog_array, x -> CASE WHEN x IS NULL THEN 'None' ELSE cast(x as string) END), ', '), ']')"
        )
        cog_array_to_str_expr = (
            "concat('[', array_join(transform(cog_array, x -> CASE WHEN x IS NULL THEN 'None' ELSE cast(x as string) END), ', '), ']')"
        )
        df = df.withColumn("sog_array_str", F.expr(sog_array_to_str_expr))
        df = df.withColumn("cog_array_str", F.expr(cog_array_to_str_expr))

        # 9) Prepare final DataFrame with stringified arrays (so CSV writer won't error)
        result = df.select(
            F.col("mmsi"),
            F.col("EventIndex"),
            F.col("trajectory"),
            F.col("timestamp_array_str").alias("timestamp_array"),
            F.col("sog_array_str").alias("sog_array"),
            F.col("cog_array_str").alias("cog_array"),
            F.col("Category").alias("behavior_type_vector"),
            F.col("average_speed"),
            F.col("min_speed"),
            F.col("max_speed"),
            F.col("average_heading"),
            F.col("std_dev_heading"),
            F.col("total_area_time"),
            F.col("low_speed_percentage"),
            F.col("stagnation_time"),
            # NEW fields included:
            F.col("distance_in_kilometers"),
            F.col("average_time_diff_between_consecutive_points"),
            F.col("min_heading"),
            F.col("max_heading"),
            F.col("std_dev_speed")
        )

        # Rename mmsi to id for consistency with NON-TRANSSHIPMENT
        result = result.withColumnRenamed("mmsi", "id")
        
        return result

        
    def create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019(
        events_df: DataFrame,
        ais_path: str,
        spark: SparkSession,
    ) -> DataFrame:
        """
        Pure-Spark replacement for create_aux_dataframe_with_chunks_V2 (pandas).

        - events_df must contain: ['EventIndex', 'timestamp', 'id', 'Category'] where
        'timestamp' on events is the event timestamp (pandas used min(timestamp) per event).
        - ais_path is the path to the AIS CSV which contains ['id','timestamp','longitude','latitude','speed','heading'].
        - Uses millisecond resolution for timestamps (like the pandas code did with unit='ms').

        Returns a Spark DataFrame with columns:
        ['id','EventIndex','trajectory','timestamp_array','sog_array','cog_array','behavior_type_vector',
        'average_speed','min_speed','max_speed','average_heading','std_dev_heading',
        'total_area_time','low_speed_percentage','stagnation_time']
        """
        logger.info("create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019: start")

        # --- 1) Prepare events: compute event_time_ms (min timestamp per EventIndex,id) and keep Category ---
        # Convert events timestamp to milliseconds (robust to numeric or string input)
        events_pre = events_df.withColumn(
            "_ev_ts_millis",
            F.when(
                (F.col("timestamp").cast("double").isNotNull()) & (F.col("timestamp").rlike("^[0-9]+$")),
                F.col("timestamp").cast("long")  # assume already in ms
            ).otherwise(
                # fallback: parse string to seconds then multiply
                (F.unix_timestamp(F.col("timestamp").cast(StringType())).cast("long") * F.lit(1000))
            )
        )

        # For each (EventIndex, id) compute min timestamp (ms) and pick first Category
        events_bounds = (
            events_pre.groupBy("EventIndex", "id")
            .agg(
                F.min(F.col("_ev_ts_millis")).alias("event_time_ms"),
                F.first(F.col("Category")).alias("Category")
            )
            .withColumnRenamed("id", "vessel_id")
        )

        # --- 2) Read AIS CSV and normalize timestamp to ms (column _ts_millis) ---
        logger.info("create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019: reading AIS from %s", ais_path)
        ais = (
            spark.read.option("header", True).option("inferSchema", True).csv(ais_path)
            .withColumnRenamed("Id", "id")
        )
        ais = ais.toDF(*[c.strip() for c in ais.columns])
        ais = ais.withColumn("id", F.col("id").cast(StringType()))

        # Robustly compute milliseconds epoch for AIS timestamp:
        ais = ais.withColumn(
            "_ts_millis",
            F.when(
                (F.col("timestamp").cast("double").isNotNull()) & (F.col("timestamp").rlike("^[0-9]+$")),
                F.col("timestamp").cast("long")  # already ms
            ).otherwise(
                (F.unix_timestamp(F.col("timestamp").cast(StringType())).cast("long") * F.lit(1000))
            )
        ).withColumn("longitude", F.col("longitude").cast(DoubleType())) \
        .withColumn("latitude", F.col("latitude").cast(DoubleType())) \
        .withColumn("speed", F.col("speed").cast(DoubleType())) \
        .withColumn("heading", F.col("heading").cast(DoubleType()))

        # --- 3) Join AIS points to events where ais._ts_millis >= event_time_ms and same vessel id ---
        join_cond = (ais.id == events_bounds.vessel_id) & (ais._ts_millis >= events_bounds.event_time_ms)
        logger.info("create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019: joining ais -> events (this may shuffle)")
        joined = (
            ais.join(events_bounds, on=join_cond, how="inner")
            .select(
                events_bounds.EventIndex.alias("EventIndex"),
                events_bounds.vessel_id.alias("id"),
                events_bounds.Category.alias("Category"),
                ais._ts_millis.alias("ts"),
                ais.longitude.alias("lon"),
                ais.latitude.alias("lat"),
                ais.speed.alias("sog"),
                ais.heading.alias("cog"),
            )
        )

        #  --- 4) Build ordered 'points' per (EventIndex, id) ---
        # collect_list struct and array_sort by ts
        grouped = joined.groupBy("EventIndex", "id", "Category").agg(
            F.expr(
                "array_sort(collect_list(struct(ts as ts, lon as lon, lat as lat, sog as sog, cog as cog)), "
                "(x, y) -> CASE WHEN x.ts < y.ts THEN -1 WHEN x.ts > y.ts THEN 1 ELSE 0 END)"
            ).alias("points")
        )

        # --- 5) Extract arrays and compute time_diffs (milliseconds) ---
        df = grouped.withColumn("ts_array", F.expr("transform(points, x -> x.ts)")) \
                    .withColumn(
                        "timestamp_array",
                        # convert ms -> seconds for from_unixtime; show format 'YYYY-MM-DD HH:MM:SS'
                        F.expr("transform(ts_array, t -> CASE WHEN t IS NULL THEN NULL ELSE from_unixtime(cast(floor(t/1000) as bigint), 'yyyy-MM-dd HH:mm:ss') END)")
                    ) \
                    .withColumn("sog_array", F.expr("transform(points, x -> cast(x.sog as double))")) \
                    .withColumn("cog_array", F.expr("transform(points, x -> cast(x.cog as double))"))

        # NEW: lat/lon arrays for distance computation (haversine)
        df = df.withColumn("lat_array", F.expr("transform(points, x -> cast(x.lat as double))")) \
               .withColumn("lon_array", F.expr("transform(points, x -> cast(x.lon as double))"))

        # time_diffs in milliseconds: difference between consecutive ts entries
        df = df.withColumn(
            "time_diffs_ms",
            F.expr(
                "CASE WHEN size(ts_array) <= 1 THEN array() "
                "ELSE transform(sequence(2, size(ts_array)), i -> element_at(ts_array, i) - element_at(ts_array, i-1)) END"
            )
        )

        # total_time_ms (sum of all time diffs)
        df = df.withColumn("total_time_ms", F.expr("aggregate(time_diffs_ms, cast(0.0 as double), (acc,x) -> acc + x)"))

        # --- 6) total_area_time: sum of time_diffs for which grid cell exists ---
        # grid_x, grid_y derived from coordinates (floor(lon/0.1), floor(lat/0.1)) and trimmed to len(time_diffs_ms)
        # min_len = least(size(time_diffs_ms), size(points)-1)
        df = df.withColumn(
            "min_len_for_area",
            F.expr("CASE WHEN size(points) <= 1 THEN 0 ELSE least(size(time_diffs_ms), size(points) - 1) END")
        )

        # Build index sequence 1..min_len_for_area and sum corresponding time_diffs
        df = df.withColumn(
            "total_area_time",
            F.expr(
                "CASE WHEN min_len_for_area = 0 THEN 0.0 ELSE aggregate(sequence(1, min_len_for_area), cast(0.0 as double), (acc, i) -> acc + element_at(time_diffs_ms, i)) END"
            ).cast(DoubleType())
        )

        # --- 7) SOG / COG statistics (avg/min/max/std) and counts ---
        df = df.withColumn(
            "sum_sog",
            F.expr("aggregate(sog_array, cast(0.0 as double), (acc,x) -> acc + coalesce(x, cast(0.0 as double)))")
        ).withColumn(
            "sumsq_sog",
            F.expr("aggregate(sog_array, cast(0.0 as double), (acc,x) -> acc + coalesce(x, cast(0.0 as double)) * coalesce(x, cast(0.0 as double)))")
        ).withColumn(
            "count_sog",
            F.expr("aggregate(sog_array, cast(0 as int), (acc,x) -> acc + CASE WHEN x IS NULL THEN 0 ELSE 1 END)")
        )

        df = df.withColumn(
            "average_speed",
            F.expr("CASE WHEN count_sog = 0 THEN 0.0 ELSE sum_sog / cast(count_sog as double) END").cast(DoubleType())
        ).withColumn("min_speed", F.expr("array_min(sog_array)").cast(DoubleType())) \
        .withColumn("max_speed", F.expr("array_max(sog_array)").cast(DoubleType())) \
        .withColumn(
            "std_dev_speed",
            F.expr("CASE WHEN count_sog = 0 THEN 0.0 ELSE sqrt( (sumsq_sog / cast(count_sog as double)) - POWER(sum_sog / cast(count_sog as double), 2) ) END").cast(DoubleType())
        )

        # COG stats
        df = df.withColumn(
            "sum_cog",
            F.expr("aggregate(cog_array, cast(0.0 as double), (acc,x) -> acc + coalesce(x, cast(0.0 as double)))")
        ).withColumn(
            "sumsq_cog",
            F.expr("aggregate(cog_array, cast(0.0 as double), (acc,x) -> acc + coalesce(x, cast(0.0 as double)) * coalesce(x, cast(0.0 as double)))")
        ).withColumn(
            "count_cog",
            F.expr("aggregate(cog_array, cast(0 as int), (acc,x) -> acc + CASE WHEN x IS NULL THEN 0 ELSE 1 END)")
        )

        df = df.withColumn(
            "average_heading",
            F.expr("CASE WHEN count_cog = 0 THEN 0.0 ELSE sum_cog / cast(count_cog as double) END").cast(DoubleType())
        ).withColumn("std_dev_heading", F.expr("CASE WHEN count_cog = 0 THEN 0.0 ELSE sqrt((sumsq_cog / cast(count_cog as double)) - POWER(sum_cog / cast(count_cog as double), 2)) END").cast(DoubleType())) \
            .withColumn("min_heading", F.expr("array_min(cog_array)").cast(DoubleType())) \
            .withColumn("max_heading", F.expr("array_max(cog_array)").cast(DoubleType()))

        # --- 8) low_speed_percentage and stagnation_time (pandas formula parity) ---
        df = df.withColumn(
            "count_low_speed_2",
            F.expr("aggregate(sog_array, cast(0 as int), (acc,x) -> acc + CASE WHEN x < 2.0 THEN 1 ELSE 0 END)")
        ).withColumn(
            "count_low_speed_0_5",
            F.expr("aggregate(sog_array, cast(0 as int), (acc,x) -> acc + CASE WHEN x < 0.5 THEN 1 ELSE 0 END)")
        ).withColumn(
            "count_sog_total",
            F.expr("size(sog_array)")
        )

        df = df.withColumn(
            "low_speed_percentage",
            F.expr("CASE WHEN count_sog_total = 0 THEN 0.0 ELSE (count_low_speed_2 / cast(count_sog_total as double) * 100.0) END").cast(DoubleType())
        ).withColumn(
            "stagnation_time",
            F.expr(
                "CASE WHEN count_sog_total = 0 THEN 0.0 ELSE count_low_speed_0_5 * (CASE WHEN total_time_ms IS NULL THEN 0.0 ELSE total_time_ms END) / cast(count_sog_total as double) END"
            ).cast(DoubleType())
        )

        # NEW: distance_in_kilometers using haversine (Spark-only math)
        df = df.withColumn(
            "distance_in_kilometers",
            F.expr(
                "CASE WHEN size(lat_array) <= 1 THEN 0.0 ELSE aggregate(sequence(2, size(lat_array)), cast(0.0 as double), (acc, i) -> acc + ("
                "2 * 6371.0 * asin( sqrt( pow( sin( (radians(element_at(lat_array, i)) - radians(element_at(lat_array, i-1))) / 2 ), 2 ) "
                "+ cos(radians(element_at(lat_array, i-1))) * cos(radians(element_at(lat_array, i))) * pow( sin( (radians(element_at(lon_array, i)) - radians(element_at(lon_array, i-1))) / 2 ), 2 ) ) ) ) ) END"
            ).cast(DoubleType())
        )

        # NEW: average_time_diff_between_consecutive_points (seconds) — time_diffs_ms -> /1000.0
        df = df.withColumn(
            "average_time_diff_between_consecutive_points",
            F.expr(
                "CASE WHEN size(time_diffs_ms) = 0 THEN 0.0 ELSE (aggregate(time_diffs_ms, cast(0.0 as double), (acc,x) -> acc + x) / cast(size(time_diffs_ms) as double)) / 1000.0 END"
            ).cast(DoubleType())
        )

        # --- 9) Build trajectory LINESTRING('lon lat', ...) ---
        df = df.withColumn(
            "trajectory",
            F.expr("concat('LINESTRING(', array_join(transform(points, p -> concat(cast(p.lon as string), ' ', cast(p.lat as string))), ', '), ')')")
        )

        # --- 10) Stringify arrays for CSV output ---
        # timestamp_array: render as [YYYY-MM-DD HH:MM:SS, None, ...] (no quotes around timestamps)
        ts_array_to_str_expr = "concat('[', array_join(transform(timestamp_array, t -> CASE WHEN t IS NULL THEN 'None' ELSE t END), ', '), ']')"
        df = df.withColumn("timestamp_array_str", F.expr(ts_array_to_str_expr))

        # sog_array and cog_array stringification: numbers as strings, nulls as None
        sog_array_to_str_expr = "concat('[', array_join(transform(sog_array, x -> CASE WHEN x IS NULL THEN 'None' ELSE cast(x as string) END), ', '), ']')"
        cog_array_to_str_expr = "concat('[', array_join(transform(cog_array, x -> CASE WHEN x IS NULL THEN 'None' ELSE cast(x as string) END), ', '), ']')"
        df = df.withColumn("sog_array_str", F.expr(sog_array_to_str_expr))
        df = df.withColumn("cog_array_str", F.expr(cog_array_to_str_expr))

        # --- 11) Final select and rename to match pandas output keys ---
        result = df.select(
            F.col("id"),
            F.col("EventIndex"),
            F.col("trajectory"),
            F.col("timestamp_array_str").alias("timestamp_array"),
            F.col("sog_array_str").alias("sog_array"),
            F.col("cog_array_str").alias("cog_array"),
            F.col("Category").alias("behavior_type_vector"),
            F.col("average_speed"),
            F.col("min_speed"),
            F.col("max_speed"),
            F.col("average_heading"),
            F.col("std_dev_heading"),
            F.col("total_area_time"),
            F.col("low_speed_percentage"),
            F.col("stagnation_time"),
            # NEW fields included:
            F.col("distance_in_kilometers"),
            F.col("average_time_diff_between_consecutive_points"),
            F.col("min_heading"),
            F.col("max_heading"),
            F.col("std_dev_speed"),
        )

        logger.info("create_aggregated_NON_TRANSSHIPMENT_dataframe_with_spark_Pitsikalis_2019: finished")

        return result

    
    
    ######################## END ########################