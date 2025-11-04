from applications.data_processing_api.src.services.process_data_service import Process_Data_Service
import os
import logging
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)
class Prediction_Service:

    def query_aggregated_ais_covering_interval(
        spark: SparkSession,
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
        and return a Spark DataFrame with only rows where the user's interval is fully contained:
            traj_start <= start_date  AND  traj_end >= end_date

        Parameters:
        - spark: SparkSession (created by your existing init_spark_session; this function won't recreate it)
        - mmsi: vessel mmsi string to filter on (string)
        - start_date_str, end_date_str: ISO-like strings, e.g. "2015-10-02 05:06:00" or "2015-10-02T05:06:00"
        - schema, table: DB schema and table name (defaults to captaima.aggregated_ais_data)
        - jdbc_jar_path: relative or absolute path to the Postgres JDBC jar (you said it's at infrastructure/jars/...)

        Returns:
        - Spark DataFrame filtered to rows that fully contain the [start_date, end_date] interval.
        """

        # --- 1) Ensure JDBC driver jar is available to executors/drivers ---
        try:
            jar_abspath = os.path.abspath(jdbc_jar_path)
            # Add jar only if not already added (addJar is idempotent-ish)
            try:
                spark.sparkContext.addJar(jar_abspath)
                logger.info("Added JDBC jar to Spark classpath: %s", jar_abspath)
            except Exception as eadd:
                logger.warning("Could not add JDBC jar via sparkContext.addJar (may already be on classpath): %s", eadd)
        except Exception as e:
            logger.warning("Could not resolve JDBC jar path (%s): %s", jdbc_jar_path, e)

        # --- 2) Build JDBC connection properties from env --------------------------------
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
        dbtable_subquery = f"(select * from {schema}.{table} where mmsi = '{mmsi_escaped}') as subq"


        logger.info("Reading aggregated ais data for mmsi=%s from %s.%s via JDBC", mmsi, schema, table)

        # --- 3) Read only the rows with the given mmsi (reduce network transfer) -----------
        try:
            df = (
                spark.read
                .format("jdbc")
                .option("url", jdbc_url)
                .option("dbtable", dbtable_subquery)
                .option("user", pg_user)
                .option("password", pg_pass)
                .option("driver", "org.postgresql.Driver")
                # optional tuning: fetch size / pushdown (comment/uncomment if needed)
                # .option("fetchsize", "1000")
                .load()
            )
        except Exception as e:
            logger.exception("Failed to read aggregated_ais_data via JDBC: %s", e)
            raise

        # Quick counts/log (avoid expensive count if DF is huge — still useful)
        try:
            approx_count = df.rdd.getNumPartitions()  # cheap hint
            logger.info("Read DataFrame with %d partitions (rows unknown until action).", approx_count)
        except Exception:
            pass

        # --- 4) Normalize timestamp_array: remove surrounding brackets and split by commas ---
        # Assumption: timestamp_array is a string like: "[2015-10-02 05:06:37, 2015-10-02 05:06:58, ...]"
        # And you said it's already time-sorted. We'll use element_at(..., 1) and last element (size).
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

        # Convert first and last elements to timestamp type (assumes format 'yyyy-MM-dd HH:mm:ss')
        # We'll be tolerant: try with provided format; fallback if parsing returns null (we keep simple here).
        df2 = df2.withColumn(
            "traj_start_ts",
            F.to_timestamp(F.element_at(F.col("_ts_split"), F.lit(1)), "yyyy-MM-dd HH:mm:ss")
        ).withColumn(
            "traj_end_ts",
            # element_at with size -> last element
            F.to_timestamp(F.element_at(F.col("_ts_split"), F.col("_ts_size")), "yyyy-MM-dd HH:mm:ss")
        )

        # If your timestamps use a different format (e.g. ISO with 'T'), normalize the input first:
        # e.g. replace("T", " ") before to_timestamp. For now we attempt the common format above.

        # --- 5) Build filter literals for user-provided start/end (make them timestamps) ---
        # Accept both "YYYY-MM-DDTHH:MM:SS" and "YYYY-MM-DD HH:MM:SS"
        start_in = start_date_str.replace("T", " ")
        end_in = end_date_str.replace("T", " ")

        # Use to_timestamp on literals so comparison is done in TimestampType domain
        start_ts_lit = F.to_timestamp(F.lit(start_in), "yyyy-MM-dd HH:mm:ss")
        end_ts_lit = F.to_timestamp(F.lit(end_in), "yyyy-MM-dd HH:mm:ss")

        # --- 6) Filter for FULL CONTAINMENT: traj_start <= start_date AND traj_end >= end_date ---
        filtered = df2.filter(
            (F.col("traj_start_ts").isNotNull()) &
            (F.col("traj_end_ts").isNotNull()) &
            (F.col("traj_start_ts") <= start_ts_lit) &
            (F.col("traj_end_ts") >= end_ts_lit)
        )

        # Log before/after counts reasonably: avoid .count() on huge DF unless you want to pay cost.
        try:
            total_rows = df.count()
            kept_rows = filtered.count()
            logger.info("mmsi=%s: rows read=%d, rows covering interval [%s, %s]=%d", mmsi, total_rows, start_in, end_in, kept_rows)
        except Exception as ecount:
            logger.info("Counts skipped (too heavy). DataFrame read; you can trigger an action to see sizes: %s", ecount)

        # --- 7) Select & return a compact set of columns (keep as Spark DF for downstream processing) ---
        # Keep everything but also expose traj_start_ts/traj_end_ts for downstream use
        out_cols = df.columns[:]  # keep original columns
        # ensure we also include traj_start_ts and traj_end_ts
        result_df = filtered.select(*(out_cols + ["traj_start_ts", "traj_end_ts"]))

        return result_df
    
    def extract_trajectory_block_for_interval(
        spark,
        mmsi: str,
        start_date_str: str,
        end_date_str: str,
        *,
        schema: str = "captaima",
        table: str = "aggregated_ais_data",
        jdbc_jar_path: str = "infrastructure/jars/postgresql-42.7.3.jar"
    ):
        """
        Return a Spark DataFrame with ONE row per mmsi containing ALL trajectory data (points)
        whose timestamps fall inside the user interval [start_date_str, end_date_str].

        Requirements:
        - query_aggregated_ais_covering_interval(...) must exist and return aggregated rows
            that satisfy traj_start <= start_date AND traj_end >= end_date (so row-level filtering is done).
        - Spark must support arrays_zip, posexplode_outer, array_sort and higher-order functions (PySpark 2.4+/3.x).
        Returns:
        Spark DataFrame with 1 row per mmsi and columns:
            mmsi,
            contributing_eventindices (array of ints),
            n_points,
            linestring (WKT for the concatenation of included points),
            timestamp_array (array of timestamp strings "yyyy-MM-dd HH:mm:ss"),
            sog_array (array double),
            cog_array (array double),
            average_sog, min_sog, max_sog, std_dev_sog,
            average_cog, min_cog, max_cog, std_dev_cog,
            distance_in_kilometers (sum of contributing rows' distance_in_kilometers where available)
        """

        # 1) canonicalize input datetimes (accept "T" or space)
        def _canon(s: str) -> str:
            s2 = s.replace("T", " ").strip()
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s2, fmt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            # last effort
            try:
                dt = datetime.fromisoformat(s2)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                raise ValueError(f"Unrecognized datetime format: {s!r}")

        start_canon = _canon(start_date_str)
        end_canon = _canon(end_date_str)
        logger.info("Extracting block for mmsi=%s interval [%s, %s]", mmsi, start_canon, end_canon)

        # 2) read aggregated rows that fully contain the interval (your helper does JDBC + filtering)
        agg_df = query_aggregated_ais_covering_interval(
            spark=spark,
            mmsi=mmsi,
            start_date_str=start_canon,
            end_date_str=end_canon,
            schema=schema,
            table=table,
            jdbc_jar_path=jdbc_jar_path
        )

        # If nothing returned -> empty DF with expected schema
        if agg_df.rdd.isEmpty():
            logger.info("No aggregated rows found that cover the requested interval for mmsi=%s", mmsi)
            out_schema = T.StructType([
                T.StructField("mmsi", T.StringType(), True),
                T.StructField("contributing_eventindices", T.ArrayType(T.LongType()), True),
                T.StructField("n_points", T.LongType(), True),
                T.StructField("linestring", T.StringType(), True),
                T.StructField("timestamp_array", T.ArrayType(T.StringType()), True),
                T.StructField("sog_array", T.ArrayType(T.DoubleType()), True),
                T.StructField("cog_array", T.ArrayType(T.DoubleType()), True),
                T.StructField("average_sog", T.DoubleType(), True),
                T.StructField("min_sog", T.DoubleType(), True),
                T.StructField("max_sog", T.DoubleType(), True),
                T.StructField("std_dev_sog", T.DoubleType(), True),
                T.StructField("average_cog", T.DoubleType(), True),
                T.StructField("min_cog", T.DoubleType(), True),
                T.StructField("max_cog", T.DoubleType(), True),
                T.StructField("std_dev_cog", T.DoubleType(), True),
                T.StructField("distance_in_kilometers", T.DoubleType(), True)
            ])
            return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=out_schema)

        # 3) Normalize arrays and trajectory into exploded rows (distributed)
        df2 = (
            agg_df
            # timestamp_array -> array of strings
            .withColumn("_ts_body", F.regexp_replace(F.col("timestamp_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_ts_arr", F.split(F.col("_ts_body"), r"\s*,\s*"))
            # sog, cog -> arrays
            .withColumn("_sog_body", F.regexp_replace(F.col("sog_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_sog_arr", F.split(F.col("_sog_body"), r"\s*,\s*"))
            .withColumn("_cog_body", F.regexp_replace(F.col("cog_array").cast("string"), r"^\s*\[|\]\s*$", ""))
            .withColumn("_cog_arr", F.split(F.col("_cog_body"), r"\s*,\s*"))
            # trajectory WKT -> inner "lon lat, lon lat, ..." string -> array of "lon lat"
            .withColumn("_traj_body", F.regexp_replace(F.col("trajectory").cast("string"), r'(?i)^\s*LINESTRING\s*\(\s*|\)\s*$', ""))
            .withColumn("_traj_body", F.regexp_replace(F.col("_traj_body"), r"\s+", " "))
            .withColumn("_pts_arr", F.when(F.col("_traj_body").isNull(), F.array()).otherwise(F.split(F.col("_traj_body"), r"\s*,\s*")))
            # Keep EventIndex and mmsi for provenance
            .select(
                "mmsi",
                "EventIndex",
                "trajectory",
                "distance_in_kilometers",
                "_ts_arr",
                "_sog_arr",
                "_cog_arr",
                "_pts_arr"
            )
        )

        # arrays_zip to align positions: zipped = array(struct(ts, sog, cog, pt))
        df2 = df2.withColumn("_zipped", F.arrays_zip(F.col("_ts_arr"), F.col("_sog_arr"), F.col("_cog_arr"), F.col("_pts_arr")))

        # posexplode_outer -> distributed rows (one per original point)
        exploded = df2.select(
            "mmsi",
            "EventIndex",
            "distance_in_kilometers",
            F.posexplode_outer(F.col("_zipped")).alias("pos", "elem")
        ).select(
            "mmsi",
            "EventIndex",
            "distance_in_kilometers",
            "pos",
            F.col("elem").getItem(0).alias("ts_raw"),
            F.col("elem").getItem(1).alias("sog_raw"),
            F.col("elem").getItem(2).alias("cog_raw"),
            F.col("elem").getItem(3).alias("pt_raw")
        )

        # parse fields into typed columns
        exploded = (
            exploded
            .withColumn("ts_str", F.regexp_replace(F.col("ts_raw").cast("string"), r"^\"|\"$", ""))
            .withColumn("ts_str", F.regexp_replace(F.col("ts_str"), r"T", " "))
            .withColumn("ts_ts", F.to_timestamp(F.col("ts_str"), "yyyy-MM-dd HH:mm:ss"))
            .withColumn("sog", F.when(F.col("sog_raw").isNull(), None).otherwise(F.col("sog_raw").cast("double")))
            .withColumn("cog", F.when(F.col("cog_raw").isNull(), None).otherwise(F.col("cog_raw").cast("double")))
            .withColumn("pt_clean", F.regexp_replace(F.col("pt_raw").cast("string"), r"^\s*\"|\"\s*$", ""))
            .withColumn("_pt_split", F.split(F.col("pt_clean"), r"\s+"))
            .withColumn("lon", F.when(F.size(F.col("_pt_split")) >= 2, F.col("_pt_split").getItem(0).cast("double")).otherwise(None))
            .withColumn("lat", F.when(F.size(F.col("_pt_split")) >= 2, F.col("_pt_split").getItem(1).cast("double")).otherwise(None))
            .drop("ts_raw", "sog_raw", "cog_raw", "pt_raw", "pt_clean", "_pt_split", "ts_str")
        )

        # 4) Filter points strictly to the requested interval
        start_ts_lit = F.to_timestamp(F.lit(start_canon), "yyyy-MM-dd HH:mm:ss")
        end_ts_lit = F.to_timestamp(F.lit(end_canon), "yyyy-MM-dd HH:mm:ss")

        exploded_filtered = exploded.filter(
            (F.col("ts_ts").isNotNull()) &
            (F.col("ts_ts") >= start_ts_lit) &
            (F.col("ts_ts") <= end_ts_lit)
        )

        # If no points included -> return empty DF same schema
        try:
            if exploded_filtered.rdd.isEmpty():
                logger.info("No points inside interval after exploding and filtering for mmsi=%s", mmsi)
                out_schema = T.StructType([
                    T.StructField("mmsi", T.StringType(), True),
                    T.StructField("contributing_eventindices", T.ArrayType(T.LongType()), True),
                    T.StructField("n_points", T.LongType(), True),
                    T.StructField("linestring", T.StringType(), True),
                    T.StructField("timestamp_array", T.ArrayType(T.StringType()), True),
                    T.StructField("sog_array", T.ArrayType(T.DoubleType()), True),
                    T.StructField("cog_array", T.ArrayType(T.DoubleType()), True),
                    T.StructField("average_sog", T.DoubleType(), True),
                    T.StructField("min_sog", T.DoubleType(), True),
                    T.StructField("max_sog", T.DoubleType(), True),
                    T.StructField("std_dev_sog", T.DoubleType(), True),
                    T.StructField("average_cog", T.DoubleType(), True),
                    T.StructField("min_cog", T.DoubleType(), True),
                    T.StructField("max_cog", T.DoubleType(), True),
                    T.StructField("std_dev_cog", T.DoubleType(), True),
                    T.StructField("distance_in_kilometers", T.DoubleType(), True)
                ])
                return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=out_schema)
        except Exception:
            # Some Spark versions don't have rdd.isEmpty; fallback to take(1)
            if exploded_filtered.take(1) == []:
                logger.info("No points inside interval after exploding and filtering for mmsi=%s", mmsi)
                out_schema = T.StructType([
                    T.StructField("mmsi", T.StringType(), True),
                    T.StructField("contributing_eventindices", T.ArrayType(T.LongType()), True),
                    T.StructField("n_points", T.LongType(), True),
                    T.StructField("linestring", T.StringType(), True),
                    T.StructField("timestamp_array", T.ArrayType(T.StringType()), True),
                    T.StructField("sog_array", T.ArrayType(T.DoubleType()), True),
                    T.StructField("cog_array", T.ArrayType(T.DoubleType()), True),
                    T.StructField("average_sog", T.DoubleType(), True),
                    T.StructField("min_sog", T.DoubleType(), True),
                    T.StructField("max_sog", T.DoubleType(), True),
                    T.StructField("std_dev_sog", T.DoubleType(), True),
                    T.StructField("average_cog", T.DoubleType(), True),
                    T.StructField("min_cog", T.DoubleType(), True),
                    T.StructField("max_cog", T.DoubleType(), True),
                    T.StructField("std_dev_cog", T.DoubleType(), True),
                    T.StructField("distance_in_kilometers", T.DoubleType(), True)
                ])
                return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=out_schema)

        # 5) Aggregate into a single row per mmsi:
        # collect structs where first element is unix timestamp (ts_unix) so array_sort sorts by that
        pts_struct = F.struct(
            F.unix_timestamp(F.col("ts_ts")).alias("ts_unix"),
            F.col("ts_ts").alias("ts"),
            F.col("lon").alias("lon"),
            F.col("lat").alias("lat"),
            F.col("sog").alias("sog"),
            F.col("cog").alias("cog"),
            F.col("EventIndex").alias("EventIndex"),
            F.col("pos").alias("pos")
        )

        grouped = exploded_filtered.groupBy("mmsi").agg(
            F.collect_list(pts_struct).alias("pts"),
            F.collect_set(F.col("EventIndex").cast("long")).alias("contributing_eventindices"),
            F.count(F.lit(1)).alias("n_points"),
            F.sum(F.col("distance_in_kilometers")).alias("distance_in_kilometers_sum"),
            F.avg("sog").alias("average_sog"),
            F.min("sog").alias("min_sog"),
            F.max("sog").alias("max_sog"),
            F.stddev("sog").alias("std_dev_sog"),
            F.avg("cog").alias("average_cog"),
            F.min("cog").alias("min_cog"),
            F.max("cog").alias("max_cog"),
            F.stddev("cog").alias("std_dev_cog")
        )

        # sort pts by ts_unix (first struct field)
        grouped = grouped.withColumn("pts_sorted", F.expr("array_sort(pts)"))

        # build arrays and linestring
        grouped = grouped.withColumn(
            "lonlat_arr",
            F.expr("transform(pts_sorted, x -> concat(CAST(x.lon AS STRING), ' ', CAST(x.lat AS STRING)))")
        ).withColumn(
            "linestring",
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
        )

        # final select / rename
        final_cols = [
            "mmsi",
            "contributing_eventindices",
            "n_points",
            "linestring",
            "timestamp_array",
            "sog_array",
            "cog_array",
            "average_sog",
            "min_sog",
            "max_sog",
            "std_dev_sog",
            "average_cog",
            "min_cog",
            "max_cog",
            "std_dev_cog",
            "distance_in_kilometers_sum"
        ]

        final_df = grouped.select(*final_cols).withColumnRenamed("distance_in_kilometers_sum", "distance_in_kilometers")

        # LOG summary
        try:
            stats = final_df.select("mmsi", "n_points", "distance_in_kilometers").collect()
            if stats:
                logger.info("Extracted block for mmsi=%s: n_points=%s distance_sum=%s", stats[0]["mmsi"], stats[0]["n_points"], stats[0]["distance_in_kilometers"])
        except Exception:
            logger.debug("Could not collect small stats for logging (non-fatal)")

        # -------------------------------
        # Placeholder: call your anomaly detection / subdivision function here
        #
        # Example:
        # final_df = apply_anomaly_subdivision(final_df, sliding_window_size=..., step_size_hours=...)
        #
        # DO NOT implement apply_anomaly_subdivision here (you said you'll implement it later).
        # Keep the placeholder so later you can insert the row->subtrajectories decomposition.
        # -------------------------------

        return final_df

