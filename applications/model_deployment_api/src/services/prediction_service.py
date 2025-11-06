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
        table: str = "aggregated_ais_data",
        jdbc_jar_path: str = "infrastructure/jars/postgresql-42.7.3.jar"
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

        # 1) Ensure JDBC jar present
        try:
            jar_abspath = os.path.abspath(jdbc_jar_path)
            try:
                spark.sparkContext.addJar(jar_abspath)
                logger.info("Added JDBC jar to Spark classpath: %s", jar_abspath)
            except Exception as eadd:
                logger.debug("sparkContext.addJar non-fatal: %s", eadd)
        except Exception as e:
            logger.warning("Could not resolve JDBC jar path (%s): %s", jdbc_jar_path, e)

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
        dbtable_subquery = f"(select * from {schema}.{table} where mmsi = '{mmsi_escaped}') as subq"

        logger.info("Reading aggregated ais data for mmsi=%s from %s.%s via JDBC (looking for covering row or fallback within-rows)", mmsi, schema, table)

        # 3) Read only rows for mmsi
        try:
            df = (
                spark.read
                .format("jdbc")
                .option("url", jdbc_url)
                .option("dbtable", dbtable_subquery)
                .option("user", pg_user)
                .option("password", pg_pass)
                .option("driver", "org.postgresql.Driver")
                .load()
            )
        except Exception as e:
            logger.exception("Failed to read aggregated_ais_data via JDBC: %s", e)
            raise

        # quick partition hint
        try:
            logger.info("Read DataFrame for mmsi=%s with %d partitions", mmsi, df.rdd.getNumPartitions())
        except Exception:
            pass

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
                out_cols = df.columns[:]  # original DB columns
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
        return df_within.select(*final_cols)


    def sliding_window_extract_trajectory_block_for_interval(
        spark,
        mmsi: str,
        start_date_str: str,
        end_date_str: str,
        sliding_window_size: int,
        step_size_hours: int,
        *,
        schema: str = "captaima",
        table: str = "aggregated_ais_data",
        jdbc_jar_path: str = "infrastructure/jars/postgresql-42.7.3.jar"
    ):
        """
        Sliding-window version of extract_trajectory_block_for_interval.

        - sliding_window_size: window length in hours (int)
        - step_size_hours: sliding step in hours (int)

        Returns a Spark DataFrame where each row is a trajectory block corresponding to
        one sliding window (per mmsi). Columns emulate the aggregated_ais_data layout,
        with `primary_key` renamed to `primary_key_from_aggregated_ais_data`. Additional
        columns: window_index, window_start_ts, window_end_ts.
        """
        import os
        from datetime import datetime
        from pyspark.sql import functions as F
        from pyspark.sql import types as T

        # --- helper: canonicalize incoming datetimes (accept T or space) ---
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

        # validate sliding params
        if sliding_window_size is None or step_size_hours is None:
            raise ValueError("Both sliding_window_size and step_size_hours (ints) must be provided.")

        start_canon = _canon(start_date_str)
        end_canon = _canon(end_date_str)
        logger.info("Sliding-window extract for mmsi=%s interval [%s, %s] window=%dh step=%dh",
                    mmsi, start_canon, end_canon, sliding_window_size, step_size_hours)

        # --- 1) fetch aggregated rows that either cover or lie within the interval ---
        agg_df = PredictionService.query_aggregated_ais_covering_interval(
            spark=spark,
            mmsi=mmsi,
            start_date_str=start_canon,
            end_date_str=end_canon,
            schema=schema,
            table=table,
            jdbc_jar_path=jdbc_jar_path
        )

        # helper to test emptiness safely
        def _is_df_empty(df):
            try:
                return df is None or df.rdd.isEmpty()
            except Exception:
                return df is None or (df.take(1) == [])

        if _is_df_empty(agg_df):
            logger.info("No aggregated rows found that cover or lie within requested interval for mmsi=%s", mmsi)
            # return an empty DF with expected schema (DB-like + extras)
            out_schema = T.StructType([
                T.StructField("primary_key_from_aggregated_ais_data", T.ArrayType(T.LongType()), True),
                T.StructField("mmsi", T.StringType(), True),
                T.StructField("EventIndex", T.ArrayType(T.LongType()), True),
                T.StructField("trajectory", T.StringType(), True),
                T.StructField("timestamp_array", T.ArrayType(T.StringType()), True),
                T.StructField("sog_array", T.ArrayType(T.DoubleType()), True),
                T.StructField("cog_array", T.ArrayType(T.DoubleType()), True),
                T.StructField("behavior_type_vector", T.ArrayType(T.StringType()), True),
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

        # --- 2) normalize arrays & WKT into arrays of elements (distributed) ---
        df2 = (
            agg_df
            .withColumn("_ts_body", F.regexp_replace(F.col("timestamp_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_ts_arr", F.split(F.col("_ts_body"), r"\s*,\s*"))
            .withColumn("_sog_body", F.regexp_replace(F.col("sog_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_sog_arr", F.split(F.col("_sog_body"), r"\s*,\s*"))
            .withColumn("_cog_body", F.regexp_replace(F.col("cog_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_cog_arr", F.split(F.col("_cog_body"), r"\s*,\s*"))
            .withColumn("_traj_body", F.regexp_replace(F.col("trajectory").cast("string"), r'(?i)^\s*LINESTRING\s*\(\s*|\)\s*$', ""))
            .withColumn("_traj_body", F.regexp_replace(F.col("_traj_body"), r"\s+", " "))
            .withColumn("_pts_arr", F.when(F.col("_traj_body").isNull(), F.array()).otherwise(F.split(F.col("_traj_body"), r"\s*,\s*")))
            # select only columns we need downstream (preserve presence checks)
            .select(
                *([c for c in ["primary_key", "mmsi", "EventIndex", "distance_in_kilometers",
                            "total_area_time", "low_speed_percentage", "stagnation_time",
                            "average_time_diff_between_consecutive_points", "behavior_type_vector"]
                if c in agg_df.columns]),
                "_ts_arr", "_sog_arr", "_cog_arr", "_pts_arr"
            )
        )

        # arrays_zip to align positional elements and posexplode to produce one-row-per-point (distributed)
        df2 = df2.withColumn("_zipped", F.arrays_zip(F.col("_ts_arr"), F.col("_sog_arr"), F.col("_cog_arr"), F.col("_pts_arr")))

        exploded = df2.select(
            *([c for c in ["primary_key", "mmsi", "EventIndex", "distance_in_kilometers",
                        "total_area_time", "low_speed_percentage", "stagnation_time",
                        "average_time_diff_between_consecutive_points", "behavior_type_vector"]
            if c in df2.columns]),
            F.posexplode_outer(F.col("_zipped")).alias("pos", "elem")
        ).select(
            *([c for c in ["primary_key", "mmsi", "EventIndex", "distance_in_kilometers",
                        "total_area_time", "low_speed_percentage", "stagnation_time",
                        "average_time_diff_between_consecutive_points", "behavior_type_vector"]
            if c in df2.columns]),
            "pos",
            F.col("elem").getItem(0).alias("ts_raw"),
            F.col("elem").getItem(1).alias("sog_raw"),
            F.col("elem").getItem(2).alias("cog_raw"),
            F.col("elem").getItem(3).alias("pt_raw")
        )

        # parse typed columns
        exploded = (
            exploded
            .withColumn("ts_str", F.regexp_replace(F.col("ts_raw").cast("string"), r"^\"|\"$", ""))
            .withColumn("ts_str", F.regexp_replace(F.col("ts_str"), r"T", " "))
            .withColumn("ts_ts", F.to_timestamp(F.col("ts_str"), "yyyy-MM-dd HH:mm:ss"))
            .withColumn("ts_unix", F.unix_timestamp(F.col("ts_ts")))
            .withColumn("sog", F.when(F.col("sog_raw").isNull(), None).otherwise(F.col("sog_raw").cast("double")))
            .withColumn("cog", F.when(F.col("cog_raw").isNull(), None).otherwise(F.col("cog_raw").cast("double")))
            .withColumn("pt_clean", F.regexp_replace(F.col("pt_raw").cast("string"), r"^\s*\"|\"\s*$", ""))
            .withColumn("_pt_split", F.split(F.col("pt_clean"), r"\s+"))
            .withColumn("lon", F.when(F.size(F.col("_pt_split")) >= 2, F.col("_pt_split").getItem(0).cast("double")).otherwise(None))
            .withColumn("lat", F.when(F.size(F.col("_pt_split")) >= 2, F.col("_pt_split").getItem(1).cast("double")).otherwise(None))
            .drop("ts_raw", "sog_raw", "cog_raw", "pt_raw", "pt_clean", "_pt_split", "ts_str")
        )

        # --- 3) restrict points strictly to the overall user interval (avoid windows outside)
        start_unix_col = F.unix_timestamp(F.lit(start_canon), "yyyy-MM-dd HH:mm:ss")
        end_unix_col = F.unix_timestamp(F.lit(end_canon), "yyyy-MM-dd HH:mm:ss")

        exploded_filtered = exploded.filter(
            (F.col("ts_ts").isNotNull()) &
            (F.col("ts_unix") >= start_unix_col) &
            (F.col("ts_unix") <= end_unix_col)
        )

        # if no points -> empty
        try:
            if exploded_filtered.rdd.isEmpty():
                logger.info("No points inside interval after exploding and filtering for mmsi=%s", mmsi)
                return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=T.StructType([]))
        except Exception:
            if exploded_filtered.take(1) == []:
                logger.info("No points inside interval after exploding and filtering for mmsi=%s", mmsi)
                return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=T.StructType([]))

        # --- 4) For sliding windows: compute k-range for every point and explode to assign point to windows
        window_size_s = int(sliding_window_size) * 3600
        step_s = int(step_size_hours) * 3600

        # kmin = ceil((ts_unix - window_size_s - start_unix) / step_s)
        # kmax = floor((ts_unix - start_unix) / step_s)
        # clamp kmin to >= 0; if kmin > kmax -> no windows for that point
        kmin_expr = F.ceil((F.col("ts_unix") - F.lit(window_size_s) - start_unix_col) / F.lit(step_s)).cast("long")
        kmax_expr = F.floor((F.col("ts_unix") - start_unix_col) / F.lit(step_s)).cast("long")
        kmin_clamped = F.when(kmin_expr < 0, F.lit(0)).otherwise(kmin_expr)

        # generate integer sequence of window indices for each point (may be empty array)
        window_idx_arr = F.when(kmin_clamped <= kmax_expr, F.sequence(kmin_clamped, kmax_expr)).otherwise(F.array())

        exploded_windows = exploded_filtered.withColumn("window_idx_arr", window_idx_arr)

        # explode window indices so each row belongs to one window
        exploded_windows = exploded_windows.withColumn("window_index", F.explode_outer(F.col("window_idx_arr"))).drop("window_idx_arr")

        # compute concrete window start/end unix & timestamps
        window_start_unix = (start_unix_col + F.col("window_index") * F.lit(step_s)).cast("long")
        window_end_unix = (window_start_unix + F.lit(window_size_s)).cast("long")

        exploded_windows = exploded_windows.withColumn("window_start_unix", window_start_unix).withColumn("window_end_unix", window_end_unix)
        exploded_windows = exploded_windows.withColumn("window_start_ts", F.to_timestamp(F.from_unixtime(F.col("window_start_unix"))))
        exploded_windows = exploded_windows.withColumn("window_end_ts", F.to_timestamp(F.from_unixtime(F.col("window_end_unix"))))

        # drop points that don't actually lie in the computed window range (safety)
        exploded_windows = exploded_windows.filter(
            (F.col("ts_unix") >= F.col("window_start_unix")) & (F.col("ts_unix") <= F.col("window_end_unix"))
        )

        # --- 5) Group by mmsi + window_index -> collect points for each block and compute aggregates ---
        pts_struct = F.struct(
            F.col("ts_unix").alias("ts_unix"),
            F.col("ts_ts").alias("ts"),
            F.col("lon").alias("lon"),
            F.col("lat").alias("lat"),
            F.col("sog").alias("sog"),
            F.col("cog").alias("cog"),
            F.col("EventIndex").alias("EventIndex"),
            F.col("pos").alias("pos"),
            F.col("primary_key").alias("primary_key"),
            F.col("distance_in_kilometers").alias("distance_in_kilometers")
        )

        grouped = exploded_windows.groupBy("mmsi", "window_index", "window_start_unix", "window_end_unix").agg(
            F.collect_list(pts_struct).alias("pts"),
            F.collect_set(F.col("EventIndex").cast("long")).alias("EventIndex"),
            F.collect_set(F.col("primary_key").cast("long")).alias("primary_key_from_aggregated_ais_data"),
            F.count(F.lit(1)).alias("n_points"),
            # sum distances contributed by parent aggregated rows (approximate)
            F.sum(F.col("distance_in_kilometers")).alias("distance_in_kilometers"),
            # sog/cog stats
            F.avg("sog").alias("average_speed"),
            F.min("sog").alias("min_speed"),
            F.max("sog").alias("max_speed"),
            F.stddev("sog").alias("std_dev_speed"),
            F.avg("cog").alias("average_heading"),
            F.min("cog").alias("min_heading"),
            F.max("cog").alias("max_heading"),
            F.stddev("cog").alias("std_dev_heading"),
            # aggregate other parent-row metrics in a reasonable way (avg or sum depending on semantics)
            F.avg(F.col("average_time_diff_between_consecutive_points")).alias("average_time_diff_between_consecutive_points"),
            F.sum(F.col("total_area_time")).alias("total_area_time"),
            F.avg(F.col("low_speed_percentage")).alias("low_speed_percentage"),
            F.sum(F.col("stagnation_time")).alias("stagnation_time"),
            F.collect_set(F.col("behavior_type_vector")).alias("behavior_type_vector")
        )

        # sort pts by ts_unix
        grouped = grouped.withColumn("pts_sorted", F.expr("array_sort(pts)"))

        # build arrays, linestring and proper timestamp arrays
        grouped = grouped.withColumn(
            "lonlat_arr",
            F.expr("transform(pts_sorted, x -> concat(CAST(x.lon AS STRING), ' ', CAST(x.lat AS STRING)))")
        ).withColumn(
            "trajectory",
            F.concat(F.lit("LINESTRING("), F.concat_ws(", ", F.col("lonlat_arr")), F.lit(")"))
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

        # ensure final column ordering & names match DB columns (except primary_key renamed) and include window info
        final_cols = [
            "primary_key_from_aggregated_ais_data",
            "mmsi",
            "EventIndex",
            "trajectory",
            "timestamp_array",
            "sog_array",
            "cog_array",
            "behavior_type_vector",
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
            # sliding-window metadata
            "window_index",
            "window_start_ts",
            "window_end_ts"
        ]

        final_df = grouped.select(*final_cols)

        # LOG small stats (best-effort)
        try:
            sample = final_df.select("mmsi", "n_points", "window_index").limit(5).collect()
            logger.info("Sliding-window extraction produced %d blocks (sample): %s", final_df.count() if final_df is not None else -1, sample)
        except Exception:
            logger.debug("Skipping heavy logging of sliding-window results (non-fatal).")

        # Placeholder for later subdivision/anomaly detection function
        # e.g. final_df = apply_anomaly_subdivision(final_df, sliding_window_size=sliding_window_size, step_size_hours=step_size_hours)

        return final_df


