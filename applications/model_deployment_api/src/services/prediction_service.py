import os
import socket
import logging
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)
class PredictionService:

    ### BELOW ARE DB QUERYING AND SLIDING WINDOW FUNCTIONS ###
    def query_aggregated_ais_covering_interval(
        spark,
        mmsi: str,
        start_date_str: str,
        end_date_str: str,
        schema: str = "captaima",
        table: str = "aggregated_ais_data"
    ):
        """
        Read aggregated_ais_data rows for a given mmsi from Postgres via JDBC (uses provided spark),
        compute traj_start/traj_end from the (already time-sorted) `timestamp_array` column,
        and return either:
        - The single smallest aggregated trajectory that fully contains the user interval:
                traj_start <= start_date  AND  traj_end >= end_date
            (if any rows satisfy containment), OR
        - Otherwise, ALL aggregated rows (for the mmsi) that are fully WITHIN the user interval:
                traj_start >= start_date  AND  traj_end <= end_date

        Returned DataFrame contains original DB columns plus computed:
        traj_start_ts, traj_end_ts, _duration_s, _ts_size
        """
        import os
        from pyspark.sql import functions as F

        # 2) JDBC connection props from env
        pg_host = os.getenv("POSTGRES_CONTAINER_HOST", os.getenv("POSTGRES_HOST", "localhost"))
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB")
        pg_user = os.getenv("POSTGRES_USER")
        pg_pass = os.getenv("POSTGRES_PASSWORD")

        if not (pg_db and pg_user and pg_pass):
            logger.error("Missing Postgres connection env vars (POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD).")
            raise RuntimeError("Missing Postgres connection env vars (POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD).")

        jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"
        mmsi_raw = "" if mmsi is None else str(mmsi)
        mmsi_escaped = mmsi_raw.replace("'", "''")
        # mmsi is varchar -> keep quoted
        dbtable_subquery = (
            f"(select *, ST_AsText(trajectory) as trajectory_wkt "
            f"from {schema}.{table} where mmsi = '{mmsi_escaped}') as subq"
        )

        logger.info("Reading aggregated ais data for mmsi=%s from %s.%s via JDBC (looking for covering row or fallback within-rows)", mmsi, schema, table)

        # helper: sanitize column names in a DataFrame (force strings, replace numeric/empty with col_i)
        def _sanitize_column_names(df):
            try:
                original = list(df.columns)
                safe_names = []
                for i, cname in enumerate(original):
                    s = str(cname) if cname is not None else ""
                    if s.strip() == "" or s.isdigit():
                        s = f"col_{i}"
                    # avoid duplicates
                    if s in safe_names:
                        s = f"{s}_{i}"
                    safe_names.append(s)
                if safe_names != original:
                    logger.info("Sanitizing JDBC column names: %s -> %s", original, safe_names)
                    return df.toDF(*safe_names)
                else:
                    logger.debug("JDBC column names OK: %s", original)
                    return df
            except Exception as es:
                logger.debug("Sanitization of column names failed (non-fatal): %s", es)
                return df

        # 3) Read only rows for mmsi — attempt partitioned JDBC read (memory-safe)
        try:
            # decide if partitioned read is possible by querying min/max primary_key for mmsi
            bounds_tbl = f"(select min(primary_key) as min_pk, max(primary_key) as max_pk from {schema}.{table} where mmsi = '{mmsi_escaped}') as boundsq"
            min_pk = max_pk = None
            try:
                bounds_df = (
                    spark.read
                    .format("jdbc")
                    .option("url", jdbc_url)
                    .option("dbtable", bounds_tbl)
                    .option("user", pg_user)
                    .option("password", pg_pass)
                    .option("driver", "org.postgresql.Driver")
                    .option("fetchsize", "1000")
                    .load()
                )
                # small driver-side collect is acceptable here (single-row bounds)
                row = bounds_df.limit(1).collect()
                if row:
                    row0 = row[0].asDict()
                    min_pk = row0.get("min_pk", None)
                    max_pk = row0.get("max_pk", None)
            except Exception as eb:
                # non-fatal — if it fails, fall back to single-read below
                logger.debug("Could not fetch bounds for partitioning (non-fatal): %s", eb)
                min_pk = max_pk = None

            # try interpret bounds as integers
            min_int = max_int = None
            try:
                if min_pk is not None and max_pk is not None:
                    min_int = int(min_pk)
                    max_int = int(max_pk)
            except Exception:
                min_int = max_int = None

            # if we have valid numeric bounds, use partitioned read
            if min_int is not None and max_int is not None and min_int < max_int:
                sc = spark.sparkContext
                default_parallel = 2000#getattr(sc, "defaultParallelism", None) or 2000
                # reasonable cap so we don't create thousands of tiny partitions that increase overhead
                num_partitions = min(1000, max(8, int(default_parallel) * 2))
                logger.info(
                    "Using partitioned JDBC read on primary_key [%s..%s] with %d partitions (sc.defaultParallelism=%s)",
                    min_int, max_int, num_partitions, default_parallel
                )

                df = (
                    spark.read
                    .format("jdbc")
                    .option("url", jdbc_url)
                    .option("dbtable", dbtable_subquery)
                    .option("user", pg_user)
                    .option("password", pg_pass)
                    .option("driver", "org.postgresql.Driver")
                    .option("fetchsize", "1000")
                    .option("partitionColumn", "primary_key")
                    .option("lowerBound", str(min_int))
                    .option("upperBound", str(max_int))
                    .option("numPartitions", str(num_partitions))
                    .load()
                )
            else:
                # fallback: single read but with fetchsize (cursor) to reduce memory usage
                logger.info("Using single JDBC read with fetchsize; consider enabling partitioning to avoid OOM.")
                df = (
                    spark.read
                    .format("jdbc")
                    .option("url", jdbc_url)
                    .option("dbtable", dbtable_subquery)
                    .option("user", pg_user)
                    .option("password", pg_pass)
                    .option("driver", "org.postgresql.Driver")
                    .option("fetchsize", "1000")
                    .load()
                )

            # sanitize column names immediately after read (fixes errors like Field name is '0' etc)
            df = _sanitize_column_names(df)

        except Exception as e:
            logger.exception("Failed to read aggregated_ais_data via JDBC: %s", e)
            raise

        # quick partition hint (non-fatal)
        try:
            logger.info("Read DataFrame for mmsi=%s with %d partitions and columns=%s", mmsi, df.rdd.getNumPartitions(), df.columns)
        except Exception:
            logger.debug("Could not log partitions/columns (non-fatal).")

        # 4) Normalize timestamp_array -> array and compute first/last elements
        df2 = df.withColumn(
            "_ts_array_str",
            F.regexp_replace(F.col("timestamp_array").cast("string"), r"^\s*\[|\]\s*$", "")
        ).withColumn(
            "_ts_split",
            F.split(F.col("_ts_array_str"), r"\s*,\s*")
        ).withColumn(
            "_ts_size",
            F.size(F.col("_ts_split"))
        )

        # elements and normalization
        first_elem = F.element_at(F.col("_ts_split"), F.lit(1))
        last_elem = F.element_at(F.col("_ts_split"), F.col("_ts_size"))
        first_norm = F.regexp_replace(first_elem, r"T", " ")
        last_norm = F.regexp_replace(last_elem, r"T", " ")

        traj_start_ts = F.coalesce(
            F.to_timestamp(first_norm, "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp(first_norm, "yyyy-MM-dd'T'HH:mm:ss"),
            F.to_timestamp(first_norm)
        )
        traj_end_ts = F.coalesce(
            F.to_timestamp(last_norm, "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp(last_norm, "yyyy-MM-dd'T'HH:mm:ss"),
            F.to_timestamp(last_norm)
        )

        df2 = df2.withColumn("traj_start_ts", traj_start_ts).withColumn("traj_end_ts", traj_end_ts)

        # 5) prepare user-provided timestamp literals
        start_in = start_date_str.replace("T", " ")
        end_in = end_date_str.replace("T", " ")
        start_ts_lit = F.to_timestamp(F.lit(start_in), "yyyy-MM-dd HH:mm:ss")
        end_ts_lit = F.to_timestamp(F.lit(end_in), "yyyy-MM-dd HH:mm:ss")

        # 6) First: try FULL CONTAINMENT (covering rows)
        contained = df2.filter(
            (F.col("traj_start_ts").isNotNull()) &
            (F.col("traj_end_ts").isNotNull()) &
            (F.col("traj_start_ts") <= start_ts_lit) &
            (F.col("traj_end_ts") >= end_ts_lit)
        )

        # compute duration and ts_size for ordering
        contained = contained.withColumn(
            "_duration_s",
            (F.unix_timestamp(F.col("traj_end_ts")) - F.unix_timestamp(F.col("traj_start_ts")))
        ).withColumn(
            "_ts_size_int",
            F.col("_ts_size").cast("long")
        )

        smallest_one = contained.orderBy(F.col("_duration_s").asc_nulls_last(), F.col("_ts_size_int").asc_nulls_last()).limit(1)

        # If we found one covering row -> return it
        try:
            if smallest_one.take(1):
                logger.info("Found a covering aggregated row for mmsi=%s; returning the smallest covering trajectory.", mmsi)
                out_cols = df.columns[:]  # original DB columns (sanitized)
                final_cols = out_cols + ["traj_start_ts", "traj_end_ts", "_duration_s", "_ts_size"]
                return smallest_one.select(*final_cols)
        except Exception:
            # conservative: ignore take issues and proceed to fallback
            logger.debug("take(1) on smallest_one failed or not supported; proceeding to fallback if needed.")

        # 7) Fallback: keep rows fully WITHIN the user interval
        df_within = df2.filter(
            (F.col("traj_start_ts").isNotNull()) &
            (F.col("traj_end_ts").isNotNull()) &
            (F.col("traj_start_ts") >= start_ts_lit) &
            (F.col("traj_end_ts") <= end_ts_lit)
        ).withColumn(
            "_duration_s",
            (F.unix_timestamp(F.col("traj_end_ts")) - F.unix_timestamp(F.col("traj_start_ts")))
        )

        try:
            cnt_within = df_within.count()
            logger.info("Fallback: rows fully within interval for mmsi=%s = %d", mmsi, cnt_within)
        except Exception:
            logger.debug("Skipping expensive count() on df_within.")

        if df_within.rdd.isEmpty():
            # Return an empty DF with expected columns (use original df schema plus computed cols)
            logger.info("No covering row and no rows fully within interval for mmsi=%s; returning empty DataFrame.", mmsi)
            out_cols = df.columns[:]
            final_cols = out_cols + ["traj_start_ts", "traj_end_ts", "_duration_s", "_ts_size"]
            # safe empty: df.limit(0) -> empty with original schema; add computed cols to avoid select missing
            empty = df.limit(0)
            # add missing computed cols as nulls
            for c in ["traj_start_ts", "traj_end_ts", "_duration_s", "_ts_size"]:
                empty = empty.withColumn(c, F.lit(None))
            return empty.select(*final_cols)

        # Return all rows fully-within (may be multiple)
        out_cols = df.columns[:]
        final_cols = out_cols + ["traj_start_ts", "traj_end_ts", "_duration_s", "_ts_size"]

        # Produce final DataFrame to return, then free intermediate Python references
        final_df = df_within.select(*final_cols)

        # Remove intermediate DataFrame references to help Python GC (non-fatal)
        try:
            del contained
            del smallest_one
            del df2
            del df
            del df_within
        except Exception:
            pass

        return final_df



    def sliding_window_extract_trajectory_block_for_interval(
        spark,
        mmsi: str,
        start_date_str: str,
        end_date_str: str,
        sliding_window_size: int,
        step_size_hours: int,
        *,
        schema: str = "captaima",
        table: str = "aggregated_ais_data"
    ):
        import os
        import re
        from datetime import datetime
        from pyspark.sql import functions as F
        from pyspark.sql import types as T
        from pyspark.sql import Window
        from pyspark.storagelevel import StorageLevel

        # Tunable solicitado
        MAX_POINTS_PER_WINDOW = 250

        def _canon(s: str) -> str:
            s2 = s.replace("T", " ").strip()
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s2, fmt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            try:
                dt = datetime.fromisoformat(s2)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                raise ValueError(f"Unrecognized datetime format: {s!r}")

        if sliding_window_size is None or step_size_hours is None:
            raise ValueError("Both sliding_window_size and step_size_hours (ints) must be provided.")

        start_canon = _canon(start_date_str)
        end_canon = _canon(end_date_str)
        logger.info("(1 out of 17) Sliding-window extract for mmsi=%s interval [%s, %s] window=%dh step=%dh",
                    mmsi, start_canon, end_canon, sliding_window_size, step_size_hours)

        # 1) obter aggregated rows
        agg_df = PredictionService.query_aggregated_ais_covering_interval(
            spark=spark,
            mmsi=mmsi,
            start_date_str=start_canon,
            end_date_str=end_canon,
            schema=schema,
            table=table
        )

        logger.info("(2 out of 17) returned agg_df cols: %s", agg_df.columns)

        def _is_df_empty(df):
            try:
                return df is None or df.rdd.isEmpty()
            except Exception:
                return df is None or (df.take(1) == [])

        if _is_df_empty(agg_df):
            logger.info("(3 out of 17) No aggregated rows found that cover or lie within requested interval for mmsi=%s", mmsi)
            out_schema = T.StructType([
                T.StructField("primary_key_from_aggregated_ais_data", T.ArrayType(T.LongType()), True),
                T.StructField("mmsi", T.StringType(), True),
                T.StructField("event_index", T.ArrayType(T.LongType()), True),
                T.StructField("trajectory", T.StringType(), True),
                T.StructField("timestamp_array", T.ArrayType(T.StringType()), True),
                T.StructField("sog_array", T.ArrayType(T.DoubleType()), True),
                T.StructField("cog_array", T.ArrayType(T.DoubleType()), True),
                T.StructField("behavior_type_label", T.ArrayType(T.StringType()), True),
                T.StructField("average_speed", T.DoubleType(), True),
                T.StructField("min_speed", T.DoubleType(), True),
                T.StructField("max_speed", T.DoubleType(), True),
                T.StructField("average_heading", T.DoubleType(), True),
                T.StructField("min_heading", T.DoubleType(), True),
                T.StructField("max_heading", T.DoubleType(), True),
                T.StructField("std_dev_heading", T.DoubleType(), True),
                T.StructField("std_dev_speed", T.DoubleType(), True),
                T.StructField("total_area_time", T.DoubleType(), True),
                T.StructField("low_speed_percentage", T.DoubleType(), True),
                T.StructField("stagnation_time", T.DoubleType(), True),
                T.StructField("distance_in_kilometers", T.DoubleType(), True),
                T.StructField("average_time_diff_between_consecutive_points", T.DoubleType(), True),
                T.StructField("n_points", T.LongType(), True),
                T.StructField("window_index", T.LongType(), True),
                T.StructField("window_start_ts", T.StringType(), True),
                T.StructField("window_end_ts", T.StringType(), True)
            ])
            return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=out_schema)

        # SANITIZING COLUMN NAMES
        logger.info("(4 out of 17) Sanitizing column names from JDBC read if necessary.")
        try:
            original_cols = list(agg_df.columns)
            safe_names = []
            seen = set()
            for i, cname in enumerate(original_cols):
                s = "" if cname is None else str(cname)
                s_clean = s.strip()
                if s_clean == "" or re.fullmatch(r"\d+", s_clean):
                    new = f"col_{i}"
                else:
                    new = re.sub(r"[^\w]+", "_", s_clean)
                    if not re.match(r"^[A-Za-z_]", new):
                        new = f"c_{new}"
                    if new == "":
                        new = f"col_{i}"
                if new in seen:
                    new = f"{new}_{i}"
                seen.add(new)
                safe_names.append(new)
            if safe_names != original_cols:
                logger.info("(5 out of 17) Sanitizing JDBC column names: %s -> %s", original_cols, safe_names)
                agg_df = agg_df.toDF(*safe_names)
        except Exception as es:
            logger.debug("Column sanitization failed (non-fatal): %s", es)

        # --- 2) normalizing arrays & WKT ---
        logger.info("(6 out of 17) Normalizing arrays and WKT columns.")

        traj_col = "trajectory_wkt" if "trajectory_wkt" in agg_df.columns else "trajectory"

        df2 = (
            agg_df
            .withColumn("_ts_body", F.regexp_replace(F.col("timestamp_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_ts_arr", F.split(F.col("_ts_body"), r"\s*,\s*"))
            .withColumn("_sog_body", F.regexp_replace(F.col("sog_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_sog_arr", F.split(F.col("_sog_body"), r"\s*,\s*"))
            .withColumn("_cog_body", F.regexp_replace(F.col("cog_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_cog_arr", F.split(F.col("_cog_body"), r"\s*,\s*"))
            # Use the WKT text column when available
            .withColumn("_traj_body", F.regexp_replace(F.col(traj_col).cast("string"), r'(?i)^\s*LINESTRING\s*\(\s*|\)\s*$', ""))
            # removes occurrences of POINT(...) as well, just in case, leaving only coordinate pairs
            .withColumn("_traj_body", F.regexp_replace(F.col("_traj_body"), r'(?i)POINT\s*\(\s*|\)\s*', ""))
            .withColumn("_traj_body", F.regexp_replace(F.col("_traj_body"), r"\s+", " "))
            # if _traj_body is null or empty, create empty array
            .withColumn("_pts_arr",
                        F.when((F.col("_traj_body").isNull()) | (F.col("_traj_body") == ""), F.array())
                        .otherwise(F.split(F.col("_traj_body"), r"\s*,\s*")))
        )

        logger.info("(7 out of 17) Selecting columns by name (strings).")
        wanted = [
            "primary_key", "mmsi", "event_index", "distance_in_kilometers",
            "total_area_time", "low_speed_percentage", "stagnation_time",
            "average_time_diff_between_consecutive_points", "behavior_type_label"
        ]
        existing = set(df2.columns)
        select_names = [c for c in wanted if c in existing]
        for arr_name in ("_ts_arr", "_sog_arr", "_cog_arr", "_pts_arr"):
            if arr_name not in existing:
                df2 = df2.withColumn(arr_name, F.array())
            select_names.append(arr_name)

        df2 = df2.select(*select_names)

        # arrays_zip + posexplode_outer
        df2 = df2.withColumn("_zipped", F.arrays_zip(F.col("_ts_arr"), F.col("_sog_arr"), F.col("_cog_arr"), F.col("_pts_arr")))

        parent_names = [c for c in df2.columns if c not in ("_zipped", "_ts_arr", "_sog_arr", "_cog_arr", "_pts_arr")]

        logger.info("(8 out of 17) Exploding zipped arrays with posexplode_outer and accessing struct fields directly.")
        exploded = df2.select(
            *parent_names,
            F.expr("posexplode_outer(_zipped) as (pos, elem)")
        ).select(
            *parent_names,
            F.col("pos"),
            F.col("elem._ts_arr").alias("ts_raw"),
            F.col("elem._sog_arr").alias("sog_raw"),
            F.col("elem._cog_arr").alias("cog_raw"),
            F.col("elem._pts_arr").alias("pt_raw")
        )

        logger.info("debug exploded.select for 'pt_raw': %s", exploded.select("pt_raw").limit(3).collect())

        logger.info("(9 out of 17) Parsing typed columns from raw exploded data.")
        exploded = (
            exploded
            .withColumn("ts_str", F.regexp_replace(F.col("ts_raw").cast("string"), r"^\"|\"$", ""))
            .withColumn("ts_str", F.regexp_replace(F.col("ts_str"), r"T", " "))
            .withColumn("ts_ts", F.to_timestamp(F.col("ts_str"), "yyyy-MM-dd HH:mm:ss"))
            .withColumn("ts_unix", F.unix_timestamp(F.col("ts_ts")))
            .withColumn("sog", F.when(F.col("sog_raw").isNull(), None).otherwise(F.col("sog_raw").cast("double")))
            .withColumn("cog", F.when(F.col("cog_raw").isNull(), None).otherwise(F.col("cog_raw").cast("double")))
            # --- ADJUSTED: robust pt parsing: remove POINT(...)/quotes, replace commas with spaces and normalize whitespace ---
            .withColumn("pt_clean", F.regexp_replace(F.col("pt_raw").cast("string"),
                                                    r'(?i)^\s*POINT\s*\(|\)\s*$|^\"|\"$', ""))
            .withColumn("pt_clean", F.when(F.col("pt_clean").isNull(), F.lit("")).otherwise(F.regexp_replace(F.col("pt_clean"), r",", " ")))
            .withColumn("pt_clean", F.regexp_replace(F.col("pt_clean"), r"\s+", " "))
            .withColumn("pt_clean", F.trim(F.col("pt_clean")))
            .withColumn("_pt_split", F.split(F.col("pt_clean"), r"\s+"))
            .withColumn("lon", F.when(F.size(F.col("_pt_split")) >= 2, F.col("_pt_split").getItem(0).cast("double")).otherwise(None))
            .withColumn("lat", F.when(F.size(F.col("_pt_split")) >= 2, F.col("_pt_split").getItem(1).cast("double")).otherwise(None))
            .drop("ts_raw", "sog_raw", "cog_raw", "pt_raw", "pt_clean", "_pt_split", "ts_str")
        )

        logger.info("(10 out of 17) Filtering points inside [start, end] interval.")
        start_unix_col = F.unix_timestamp(F.lit(start_canon), "yyyy-MM-dd HH:mm:ss")
        end_unix_col = F.unix_timestamp(F.lit(end_canon), "yyyy-MM-dd HH:mm:ss")

        exploded_filtered = exploded.filter(
            (F.col("ts_ts").isNotNull()) &
            (F.col("ts_unix") >= start_unix_col) &
            (F.col("ts_unix") <= end_unix_col)
        )

        try:
            if exploded_filtered.rdd.isEmpty():
                logger.info("(11 out of 17) No points inside interval after exploding and filtering for mmsi=%s", mmsi)
                return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=T.StructType([]))
        except Exception:
            if exploded_filtered.take(1) == []:
                logger.info("(11 out of 17) No points inside interval after exploding and filtering for mmsi=%s", mmsi)
                return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=T.StructType([]))

        # 4) indexes of sliding windows covering each point
        logger.info("(12 out of 17) Computing sliding window indexes for each point.")
        window_size_s = int(sliding_window_size) * 3600
        step_s = int(step_size_hours) * 3600

        kmin_expr = F.ceil((F.col("ts_unix") - F.lit(window_size_s) - start_unix_col) / F.lit(step_s)).cast("long")
        kmax_expr = F.floor((F.col("ts_unix") - start_unix_col) / F.lit(step_s)).cast("long")
        kmin_clamped = F.when(kmin_expr < 0, F.lit(0)).otherwise(kmin_expr)

        window_idx_arr = F.when(kmin_clamped <= kmax_expr, F.sequence(kmin_clamped, kmax_expr)).otherwise(F.array())
        exploded_windows = exploded_filtered.withColumn("window_idx_arr", window_idx_arr)
        exploded_windows = exploded_windows.withColumn("window_index", F.explode_outer(F.col("window_idx_arr"))).drop("window_idx_arr")

        window_start_unix = (start_unix_col + F.col("window_index") * F.lit(step_s)).cast("long")
        window_end_unix = (window_start_unix + F.lit(window_size_s)).cast("long")

        exploded_windows = exploded_windows.withColumn("window_start_unix", window_start_unix).withColumn("window_end_unix", window_end_unix)
        exploded_windows = exploded_windows.withColumn("window_start_ts", F.to_timestamp(F.from_unixtime(F.col("window_start_unix"))))
        exploded_windows = exploded_windows.withColumn("window_end_ts", F.to_timestamp(F.from_unixtime(F.col("window_end_unix"))))

        exploded_windows = exploded_windows.filter(
            (F.col("ts_unix") >= F.col("window_start_unix")) & (F.col("ts_unix") <= F.col("window_end_unix"))
        )

        # Repartition + persist (there is no RAM increase here since it is a reshuffle of existing rows)
        default_parallel = 2000
        num_partitions = min(1000, max(8, int(default_parallel) * 2))
        try:
            spark.conf.set("spark.sql.shuffle.partitions", str(num_partitions))
        except Exception:
            pass

        exploded_windows = exploded_windows.repartition(num_partitions, F.col("mmsi"), F.col("window_index"))
        exploded_windows = exploded_windows.persist(StorageLevel.MEMORY_AND_DISK)

        # trim per-window points to MAX_POINTS_PER_WINDOW
        logger.info("(13 out of 17) Trimming points per window to max %d points if necessary.", MAX_POINTS_PER_WINDOW)
        try:
            w = Window.partitionBy("mmsi", "window_index").orderBy("ts_unix")
            exploded_windows = exploded_windows.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") <= MAX_POINTS_PER_WINDOW).drop("_rn")
        except Exception:
            logger.debug("Per-window trimming failed (non-fatal).")

        # 5) group and aggregate
        logger.info("(14 out of 17) Grouping and aggregating points into sliding windows.")
        pts_struct = F.struct(
            F.col("ts_unix").alias("ts_unix"),
            F.col("ts_ts").alias("ts"),
            F.col("lon").alias("lon"),
            F.col("lat").alias("lat"),
            F.col("sog").alias("sog"),
            F.col("cog").alias("cog"),
            F.col("event_index").alias("event_index"),
            F.col("pos").alias("pos"),
            F.col("primary_key").alias("primary_key"),
            F.col("distance_in_kilometers").alias("distance_in_kilometers")
        )

        grouped = exploded_windows.groupBy("mmsi", "window_index", "window_start_unix", "window_end_unix").agg(
            F.collect_list(pts_struct).alias("pts"),
            F.collect_set(F.col("event_index").cast("long")).alias("event_index"),
            F.collect_set(F.col("primary_key").cast("long")).alias("primary_key_from_aggregated_ais_data"),
            F.count(F.lit(1)).alias("n_points"),
            # keep original sum as a fallback but we'll recompute distance below
            F.sum(F.col("distance_in_kilometers")).alias("distance_in_kilometers_raw_sum"),
            F.avg("sog").alias("average_speed"),
            F.min("sog").alias("min_speed"),
            F.max("sog").alias("max_speed"),
            F.stddev("sog").alias("std_dev_speed"),
            F.avg("cog").alias("average_heading"),
            F.min("cog").alias("min_heading"),
            F.max("cog").alias("max_heading"),
            F.stddev("cog").alias("std_dev_heading"),
            F.avg(F.col("average_time_diff_between_consecutive_points")).alias("average_time_diff_between_consecutive_points"),
            F.sum(F.col("total_area_time")).alias("total_area_time"),
            F.avg(F.col("low_speed_percentage")).alias("low_speed_percentage"),
            F.sum(F.col("stagnation_time")).alias("stagnation_time"),
            F.collect_set(F.col("behavior_type_label")).alias("behavior_type_label")
        )

        grouped = grouped.withColumn("pts_sorted", F.expr("array_sort(pts)"))

        grouped = grouped.withColumn(
            "lonlat_arr",
            F.expr(
                "transform(filter(pts_sorted, x -> x.lon is not null and x.lat is not null), x -> concat(CAST(x.lon AS STRING), ' ', CAST(x.lat AS STRING)))"
            )
        ).withColumn(
            "trajectory",
            F.when(F.size(F.col("lonlat_arr")) == 0,
                F.lit("LINESTRING()"))
            .otherwise(F.concat(F.lit("LINESTRING("), F.concat_ws(", ", F.col("lonlat_arr")), F.lit(")")))
        ).withColumn(
            "timestamp_array",
            F.expr("transform(pts_sorted, x -> date_format(x.ts, 'yyyy-MM-dd HH:mm:ss'))")
        ).withColumn(
            "sog_array",
            F.expr("transform(pts_sorted, x -> x.sog)")
        ).withColumn(
            "cog_array",
            F.expr("transform(pts_sorted, x -> x.cog)")
        ).withColumn(
            "window_start_ts",
            F.to_timestamp(F.from_unixtime(F.col("window_start_unix")))
        ).withColumn(
            "window_end_ts",
            F.to_timestamp(F.from_unixtime(F.col("window_end_unix")))
        )

        # --- 15) Compute lat/lon arrays, time diffs (ms), per-window distance (haversine) and new metrics ---
        logger.info("(15 out of 17) Computing distance_in_kilometers, displacement_ratio, cog_unit_range and cog_ratio for each window.")
        grouped = grouped.withColumn(
            "lat_array",
            F.expr("transform(pts_sorted, x -> cast(x.lat as double))")
        ).withColumn(
            "lon_array",
            F.expr("transform(pts_sorted, x -> cast(x.lon as double))")
        ).withColumn(
            "ts_array",
            F.expr("transform(pts_sorted, x -> cast(x.ts_unix as long))")  # seconds
        ).withColumn(
            # time_diffs_ms: differences (in ms) between consecutive ts entries
            "time_diffs_ms",
            F.expr(
                "CASE WHEN size(ts_array) <= 1 THEN array() "
                "ELSE transform(sequence(2, size(ts_array)), i -> (element_at(ts_array, i) - element_at(ts_array, i-1)) * 1000) END"
            )
        )

        # recompute distance using haversine across lat_array/lon_array (in km)
        grouped = grouped.withColumn(
            "distance_in_kilometers",
            F.expr(
                "CASE WHEN size(lat_array) <= 1 THEN 0.0 ELSE aggregate(sequence(2, size(lat_array)), cast(0.0 as double), (acc, i) -> acc + ("
                "2 * 6371.0 * asin( sqrt( pow( sin( (radians(element_at(lat_array, i)) - radians(element_at(lat_array, i-1))) / 2 ), 2 ) "
                "+ cos(radians(element_at(lat_array, i-1))) * cos(radians(element_at(lat_array, i))) * pow( sin( (radians(element_at(lon_array, i)) - radians(element_at(lon_array, i-1))) / 2 ), 2 ) ) ) ) ) END"
            ).cast(T.DoubleType())
        )

        # displacement_ratio: first-last straight distance divided by total window distance
        grouped = grouped.withColumn("first_lat", F.element_at(F.col("lat_array"), 1)) \
            .withColumn("first_lon", F.element_at(F.col("lon_array"), 1)) \
            .withColumn("last_lat", F.element_at(F.col("lat_array"), F.size(F.col("lat_array")))) \
            .withColumn("last_lon", F.element_at(F.col("lon_array"), F.size(F.col("lon_array"))))

        grouped = grouped.withColumn(
            "displacement_km",
            F.expr(
                "CASE WHEN first_lat IS NULL OR last_lat IS NULL THEN 0.0 ELSE "
                "2 * 6371.0 * asin( sqrt( pow( sin( (radians(last_lat) - radians(first_lat)) / 2 ), 2 ) + "
                "cos(radians(first_lat)) * cos(radians(last_lat)) * pow( sin( (radians(last_lon) - radians(first_lon)) / 2 ), 2 ) ) ) END"
            ).cast(T.DoubleType())
        ).withColumn(
            "displacement_ratio",
            F.when(F.col("distance_in_kilometers") > 0, F.col("displacement_km") / F.col("distance_in_kilometers")).otherwise(F.lit(0.0))
        )

        # cog_unit_range: time-weighted mean COG, normalized to [0,1]
        # note: atan2(deg) usage follows the formula the pandas code used (converted to SQL/Spark)
        grouped = grouped.withColumn(
            "cog_unit_numer",
            F.expr(
                "CASE WHEN size(lat_array) <= 1 THEN 0.0 ELSE aggregate(sequence(2, size(lat_array)), cast(0.0 as double), "
                "(acc, i) -> acc + ( ((degrees(atan2(element_at(lon_array, i) - element_at(lon_array, i-1), element_at(lat_array, i) - element_at(lat_array, i-1))) + 360) % 360) * element_at(time_diffs_ms, i-1) ) ) END"
            ).cast(T.DoubleType())
        ).withColumn(
            "cog_unit_denom",
            F.expr(
                "CASE WHEN size(lat_array) <= 1 THEN 0.0 ELSE aggregate(sequence(2, size(lat_array)), cast(0.0 as double), (acc,i) -> acc + element_at(time_diffs_ms, i-1)) END"
            ).cast(T.DoubleType())
        ).withColumn(
            "cog_unit_range_val",
            F.when(F.col("cog_unit_denom") == 0.0, F.lit(0.0)).otherwise(F.col("cog_unit_numer") / F.col("cog_unit_denom"))
        ).withColumn(
            "cog_unit_range",
            (F.col("cog_unit_range_val") / F.lit(360.0)).cast(T.DoubleType())
        ).drop("cog_unit_numer", "cog_unit_denom", "cog_unit_range_val")

        # cog_ratio: fraction of consecutive COG changes > 10 degrees
        grouped = grouped.withColumn(
            "_cog_change_count",
            F.expr(
                "CASE WHEN size(cog_array) <= 1 THEN 0 ELSE aggregate(sequence(2,size(cog_array)), cast(0 as int), (acc,i) -> acc + CASE WHEN abs(element_at(cog_array,i) - element_at(cog_array,i-1)) > 10.0 THEN 1 ELSE 0 END) END"
            )
        ).withColumn(
            "cog_ratio",
            F.when(F.size(F.col("cog_array")) <= 1, F.lit(0.0)).otherwise(F.col("_cog_change_count") / (F.size(F.col("cog_array")) - 1))
        ).drop("_cog_change_count")

        # drop helper columns used for computation that should not be persisted
        grouped = grouped.drop("first_lat", "first_lon", "last_lat", "last_lon", "displacement_km", "lat_array", "lon_array", "ts_array", "time_diffs_ms")

        final_cols = [
            "primary_key_from_aggregated_ais_data",
            "mmsi",
            "event_index",
            "trajectory",
            "timestamp_array",
            "sog_array",
            "cog_array",
            "behavior_type_label",
            "average_speed",
            "min_speed",
            "max_speed",
            "average_heading",
            "min_heading",
            "max_heading",
            "std_dev_heading",
            "std_dev_speed",
            "total_area_time",
            "low_speed_percentage",
            "stagnation_time",
            "distance_in_kilometers",
            "average_time_diff_between_consecutive_points",
            "n_points",
            "window_index",
            "window_start_ts",
            "window_end_ts",
            # new metrics
            "displacement_ratio",
            "cog_unit_range",
            "cog_ratio"
        ]

        final_df = grouped.select(*[c for c in final_cols if c in grouped.columns])

        # Delete unnecessary auxiliary DataFrame references to help Python GC (non-fatal)
        try:
            del agg_df
            del df2
            del exploded
            del exploded_filtered
            del grouped
        except Exception:
            pass

        # logging: try to get exact total windows (may be expensive) but fallback gracefully
        try:
            total_windows = final_df.count()
            sample_first = final_df.orderBy("window_index").limit(1).collect()
            logger.info("(16 out of 17) Sliding-window extraction produced %d windows; sample of first window: %s",
                        total_windows, sample_first)
        except Exception as e:
            # fallback: do a cheap existence/sample check
            sample_first = final_df.orderBy("window_index").limit(1).collect()
            has_any = True if sample_first else False
            logger.info("(16 out of 17) Sliding-window extraction produced %s windows (exact count skipped); sample of first window: %s",
                        ">=1" if has_any else "0", sample_first)

        try:
            exploded_windows.unpersist(blocking=False)
            logger.info("(17 out of 17) exploded_windows unpersisted (blocking=False) and function returning final_df.")
        except Exception:
            logger.info("(17 out of 17) exploded_windows unpersist attempted (non-fatal).")

        return final_df

    

    def classify_trajectory_with_loitering_equation(
        windows_df,
        earth_radius_km: float = 6371.0,
        redundancy_threshold: float = 1.0
    ):
        """
        Pure-Spark classifier that receives the windows DataFrame (as returned by
        sliding_window_extract_trajectory_block_for_interval) and returns the same
        rows/columns in the original order plus two columns:
        - trajectory_redundancy (double)
        - behavior_type_by_loitering_equation (string) -> 'LOITERING' if redundancy > redundancy_threshold else 'NON-LOITERING'

        Parameters
        ----------
        windows_df : pyspark.sql.DataFrame
            Input DataFrame containing at least a 'trajectory' or 'trajectory_wkt' column
            in LINESTRING(lon lat, ...) text format (but function is robust if missing).
        earth_radius_km : float
            Earth radius in kilometers used by the haversine formula (default 6371.0).
        redundancy_threshold : float
            Threshold to classify a window as LOITERING (default 1.0).

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with original columns preserved in order plus appended new columns
            (if they did not exist already): 'trajectory_redundancy' and
            'behavior_type_by_loitering_equation'.
        """
        # preserve original column order
        original_cols = list(windows_df.columns)

        # choose trajectory column if present
        if "trajectory" in original_cols:
            traj_col_expr = F.col("trajectory").cast("string")
        elif "trajectory_wkt" in original_cols:
            traj_col_expr = F.col("trajectory_wkt").cast("string")
        else:
            traj_col_expr = F.lit("")

        # 1) extract body of LINESTRING text
        df = windows_df.withColumn(
            "_traj_body",
            F.regexp_replace(traj_col_expr, r'(?i)^\s*LINESTRING\s*\(\s*|\)\s*$', "")
        )

        # 2) split into array of "lon lat" strings (empty array if none)
        df = df.withColumn(
            "_pt_arr",
            F.when((F.col("_traj_body").isNull()) | (F.col("_traj_body") == ""), F.array()).otherwise(
                F.split(F.col("_traj_body"), r"\s*,\s*")
            )
        )

        # 3) numeric lon/lat arrays
        df = df.withColumn(
            "lon_arr",
            F.expr("transform(_pt_arr, p -> cast(split(trim(p), '\\\\s+')[0] as double))")
        ).withColumn(
            "lat_arr",
            F.expr("transform(_pt_arr, p -> cast(split(trim(p), '\\\\s+')[1] as double))")
        )

        # 4) array size and endpoints
        df = df.withColumn("n_pts_array", F.size(F.col("lon_arr")))

        df = df.withColumn(
            "first_lon",
            F.when(F.col("n_pts_array") >= 1, F.element_at(F.col("lon_arr"), F.lit(1))).otherwise(F.lit(None))
        ).withColumn(
            "first_lat",
            F.when(F.col("n_pts_array") >= 1, F.element_at(F.col("lat_arr"), F.lit(1))).otherwise(F.lit(None))
        ).withColumn(
            "last_lon",
            F.when(F.col("n_pts_array") >= 1, F.element_at(F.col("lon_arr"), F.col("n_pts_array"))).otherwise(F.lit(None))
        ).withColumn(
            "last_lat",
            F.when(F.col("n_pts_array") >= 1, F.element_at(F.col("lat_arr"), F.col("n_pts_array"))).otherwise(F.lit(None))
        )

        # 5) haversine implemented with Spark column expressions; returns kilometers
        def haversine_km(lat1_col, lon1_col, lat2_col, lon2_col):
            lat1_rad = F.radians(lat1_col)
            lat2_rad = F.radians(lat2_col)
            lon1_rad = F.radians(lon1_col)
            lon2_rad = F.radians(lon2_col)
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = F.sin(dlat / 2) * F.sin(dlat / 2) + F.cos(lat1_rad) * F.cos(lat2_rad) * F.sin(dlon / 2) * F.sin(dlon / 2)
            c = 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))
            return F.lit(earth_radius_km) * c

        # 6) departure-to-arrival distance (numerator)
        df = df.withColumn(
            "departure_to_arrival_km",
            F.when(F.col("n_pts_array") >= 2,
                haversine_km(F.col("first_lat"), F.col("first_lon"), F.col("last_lat"), F.col("last_lon"))
            ).otherwise(F.lit(0.0))
        )

        # 7) bbox min/max lon/lat
        df = df.withColumn("min_lon", F.array_min(F.col("lon_arr"))) \
            .withColumn("max_lon", F.array_max(F.col("lon_arr"))) \
            .withColumn("min_lat", F.array_min(F.col("lat_arr"))) \
            .withColumn("max_lat", F.array_max(F.col("lat_arr")))

        # 8) bbox width/height (km); guard against nulls
        df = df.withColumn(
            "bbox_width_km",
            F.when((F.col("min_lon").isNotNull()) & (F.col("max_lon").isNotNull()) & (F.col("min_lat").isNotNull()),
                haversine_km(F.col("min_lat"), F.col("min_lon"), F.col("min_lat"), F.col("max_lon"))
            ).otherwise(F.lit(0.0))
        ).withColumn(
            "bbox_height_km",
            F.when((F.col("min_lat").isNotNull()) & (F.col("max_lat").isNotNull()) & (F.col("min_lon").isNotNull()),
                haversine_km(F.col("min_lat"), F.col("min_lon"), F.col("max_lat"), F.col("min_lon"))
            ).otherwise(F.lit(0.0))
        )

        # 9) perimeter
        df = df.withColumn("bbox_perimeter_km", 2 * (F.col("bbox_width_km") + F.col("bbox_height_km")))

        # 10) redundancy with safe division
        df = df.withColumn(
            "trajectory_redundancy",
            F.when((F.col("bbox_perimeter_km").isNotNull()) & (F.col("bbox_perimeter_km") > 0),
                (F.col("departure_to_arrival_km") / F.col("bbox_perimeter_km"))
            ).otherwise(F.lit(0.0)).cast(T.DoubleType())
        )

        # --- 11) behavior type label using redundancy_threshold (renamed as requested) ---
        df = df.withColumn(
            "behavior_type_by_loitering_equation",
            F.when(F.col("trajectory_redundancy") > F.lit(float(redundancy_threshold)), F.lit("LOITERING")).otherwise(F.lit("NON-LOITERING"))
        )

        # drop only the auxiliary columns we created earlier (defensive)
        aux_cols = [
            "_traj_body", "_pt_arr", "lon_arr", "lat_arr", "n_pts_array",
            "first_lon", "first_lat", "last_lon", "last_lat",
            "departure_to_arrival_km", "min_lon", "max_lon", "min_lat", "max_lat",
            "bbox_width_km", "bbox_height_km", "bbox_perimeter_km"
        ]
        existing_aux = [c for c in aux_cols if c in df.columns]
        df = df.drop(*existing_aux)

        # final column order: original columns (unchanged) + appended new columns if they weren't in the original
        new_cols = ["trajectory_redundancy", "behavior_type_by_loitering_equation"]
        cols_to_append = [c for c in new_cols if c not in original_cols]
        final_cols = original_cols + cols_to_append
        final_cols = [c for c in final_cols if c in df.columns]

        return df.select(*final_cols)
