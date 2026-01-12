import os
import json
import math
import socket
import logging
from datetime import datetime
from dotenv import load_dotenv
from domain.config.data_processing.spark_session_initializer import SparkSessionInitializer
from domain.services.mlflow_service import MLFlowService

"""
Spark-based training services that read only from captaima.aggregated_ais_data,
use the numeric feature columns defined in AggregatedAISData, perform balanced sampling,
manual preprocessing (StringIndexer, VectorAssembler, StandardScaler), train models,
evaluate, and log to MLflow.
"""

import time
import tempfile
import traceback
from typing import List, Dict, Tuple, Optional
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql.functions import rand, row_number
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import LinearSVC, OneVsRest, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import mlflow
from mlflow import MlflowClient
import mlflow.spark
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JDBC config from env (kept for read_aggregated_table)
JDBC_URL = os.getenv("JDBC_URL")
JDBC_USER = os.getenv("JDBC_USER")
JDBC_PASSWORD = os.getenv("JDBC_PASSWORD")
JDBC_DRIVER = os.getenv("JDBC_DRIVER", "org.postgresql.Driver")
JDBC_TABLE = os.getenv("JDBC_TABLE", "captaima.aggregated_ais_data")

# The set of numeric feature columns defined in your AggregatedAISData model (kept exactly)
FEATURE_COLUMNS = [
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
    "displacement_ratio",
    "cog_unit_range",
    "cog_ratio"
]

# Allowed labels
ALLOWED_LABELS = ["LOITERING", "NORMAL", "STOPPING", "TRANSSHIPMENT"]

class TrainModelService:

    # --------------------------
    # MLflow run utilities
    # --------------------------
    @staticmethod
    def _ensure_no_active_mlflow_run(max_attempts: int = 5, sleep_seconds: float = 0.2) -> bool:
        """
        Try to end any active MLflow run. Returns True if no active run remains
        (safe to call start_run with nested=False). If we cannot guarantee that,
        returns False so callers can choose to use start_run(nested=True).
        """
        try:
            attempts = 0
            while mlflow.active_run() and attempts < max_attempts:
                run_obj = mlflow.active_run()
                try:
                    logger.warning("Defensive: ending active MLflow run (id=%s) attempt %d/%d",
                                run_obj.info.run_id if run_obj else "<unknown>", attempts + 1, max_attempts)
                    mlflow.end_run()  # defensive
                except Exception as e:
                    logger.exception("mlflow.end_run() attempt %d failed: %s", attempts + 1, e)
                attempts += 1
                # short pause to let internal state settle (helps with remote trackers)
                time.sleep(sleep_seconds)

            if mlflow.active_run():
                # still active after attempts
                logger.error("Active MLflow run remains after %d attempts. Caller should use nested=True.", max_attempts)
                return False
            logger.debug("No active MLflow run detected.")
            return True
        except Exception as exc:
            logger.exception("Unexpected error while ensuring no active mlflow run: %s", exc)
            return False

    @staticmethod
    def _end_mlflow_if_active():
        """End the active MLflow run if one exists (used in finally/except)."""
        try:
            if mlflow.active_run():
                run_id = mlflow.active_run().info.run_id
                logger.info("Ending MLflow active run (id=%s).", run_id)
                mlflow.end_run()
        except Exception as ex:
            logger.exception("Error ending mlflow run: %s", ex)

    # ------------------------------------------------------------------
    # Helper functions (Spark-first)
    # ------------------------------------------------------------------

    @staticmethod
    def clean_and_impute(train_df, test_df, feature_cols: List[str], impute_strategy: str = "median"):
        """
        Clean numeric features, replace Inf/NaN with null, drop zero-variance features,
        fit an Imputer on train_df and transform both train_df and test_df.

        Returns: (train_imputed_df, test_imputed_df, used_features_list, imputer_model)
        """
        logger.info("Starting clean_and_impute: impute_strategy=%s, num_features=%d", impute_strategy, len(feature_cols))

        # cast and nullify NaN/Inf
        def cast_and_nullify(df_local):
            for c in feature_cols:
                df_local = df_local.withColumn(c, F.col(c).cast(T.DoubleType()))
                df_local = df_local.withColumn(
                    c,
                    F.when(
                        (F.col(c).isNull()) |
                        (F.isnan(F.col(c))) |
                        (F.col(c) == float("inf")) |
                        (F.col(c) == float("-inf")),
                        None
                    ).otherwise(F.col(c))
                )
            return df_local

        train_clean = cast_and_nullify(train_df)
        test_clean = cast_and_nullify(test_df)

        logger.info("Computing feature stddevs to drop zero-variance features")
        stddev_row = train_clean.select([F.stddev(F.col(c)).alias(c) for c in feature_cols]).collect()[0].asDict()
        zero_var_cols = [c for c, v in stddev_row.items() if v is None or float(v) == 0.0]
        used_features = [c for c in feature_cols if c not in zero_var_cols]

        logger.info("Zero-variance features: %s", zero_var_cols)
        logger.info("Features used after dropping zero-variance: %s", used_features)

        if not used_features:
            logger.warning("No usable features after dropping zero-variance features.")
            return None, None, [], None

        # Fit imputer on train
        logger.info("Fitting Imputer on training data with strategy='%s' for %d features", impute_strategy, len(used_features))
        imputer = Imputer(inputCols=used_features, outputCols=[f"{c}_imp" for c in used_features]).setStrategy(impute_strategy)
        imputer_model = imputer.fit(train_clean)

        train_imp = imputer_model.transform(train_clean)
        test_imp = imputer_model.transform(test_clean)

        # replace original columns with imputed columns
        for c in used_features:
            train_imp = train_imp.drop(c).withColumnRenamed(f"{c}_imp", c)
            test_imp = test_imp.drop(c).withColumnRenamed(f"{c}_imp", c)

        logger.info("Completed imputation on train and test. Returning results.")
        return train_imp, test_imp, used_features, imputer_model

    @staticmethod
    def compute_metrics_from_confusion_rows(rows, label_order: List[str]) -> dict:
        """
        rows: list of Row(label=<num>, prediction=<num>, count=<int>)
        label_order: list mapping label index -> original label string
        returns a dict with per-class metrics, macro averages and confusion matrix.
        """
        # unchanged logic, but add a log
        logger.debug("Computing metrics from confusion rows; rows_count=%d, labels=%s", len(rows), label_order)
        n = len(label_order)
        cm = [[0 for _ in range(n)] for _ in range(n)]
        total = 0
        for r in rows:
            li = int(r["label"])
            pi = int(r["prediction"])
            cnt = int(r["count"])
            if 0 <= li < n and 0 <= pi < n:
                cm[li][pi] += cnt
                total += cnt

        per_class = {}
        sum_precision = sum_recall = sum_f1 = 0.0
        for i, label in enumerate(label_order):
            tp = cm[i][i]
            fn = sum(cm[i][j] for j in range(n) if j != i)
            fp = sum(cm[r][i] for r in range(n) if r != i)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1-score": f1, "support": tp + fn}
            sum_precision += prec
            sum_recall += rec
            sum_f1 += f1

        macro_precision = sum_precision / n if n > 0 else 0.0
        macro_recall = sum_recall / n if n > 0 else 0.0
        macro_f1 = sum_f1 / n if n > 0 else 0.0
        accuracy = sum(cm[i][i] for i in range(n)) / total if total > 0 else 0.0

        report = {
            "per_class": per_class,
            "macro_avg": {"precision": macro_precision, "recall": macro_recall, "f1-score": macro_f1},
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "total_samples": total,
            "labels": label_order
        }
        logger.debug("Metrics computed: accuracy=%.4f, total_samples=%d", accuracy, total)
        return report

    @staticmethod
    def log_metrics_from_report(report: dict, prefix: str, step: Optional[int]):
        """
        Log numeric metrics from report to MLflow. If step is provided, metrics appear as a line series.
        """
        labels = report["labels"]
        for lab in labels:
            cls = report["per_class"][lab]
            mlflow.log_metric(f"{prefix}_precision_{lab}", float(cls["precision"]), step=step if step is not None else None)
            mlflow.log_metric(f"{prefix}_recall_{lab}", float(cls["recall"]), step=step if step is not None else None)
            mlflow.log_metric(f"{prefix}_f1_{lab}", float(cls["f1-score"]), step=step if step is not None else None)
            mlflow.log_metric(f"{prefix}_support_{lab}", float(cls["support"]), step=step if step is not None else None)

        mlflow.log_metric(f"{prefix}_accuracy", float(report["accuracy"]), step=step if step is not None else None)
        mlflow.log_metric(f"{prefix}_precision_macro", float(report["macro_avg"]["precision"]), step=step if step is not None else None)
        mlflow.log_metric(f"{prefix}_recall_macro", float(report["macro_avg"]["recall"]), step=step if step is not None else None)
        mlflow.log_metric(f"{prefix}_f1_macro", float(report["macro_avg"]["f1-score"]), step=step if step is not None else None)
        logger.info("Logged MLflow metrics for %s (step=%s) : accuracy=%.4f", prefix, str(step), float(report["accuracy"]))

    @staticmethod
    def plot_confusion_matrix_and_log(report: dict, artifact_dir="artifacts/confusion_matrix"):
        os.makedirs(artifact_dir, exist_ok=True)
        cm = report["confusion_matrix"]
        labels = report["labels"]
        n = len(labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=list(range(n)), yticks=list(range(n)),
            xticklabels=labels, yticklabels=labels,
            ylabel="True label", xlabel="Predicted label",
            title="Confusion matrix (test set)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = max(max(row) for row in cm) / 2.0 if cm else 0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")
        plt.tight_layout()
        png_path = os.path.join(artifact_dir, "confusion_matrix.png")
        fig.savefig(png_path)
        plt.close(fig)

        json_path = os.path.join(artifact_dir, "confusion_matrix.json")
        with open(json_path, "w") as fh:
            json.dump({"labels": labels, "matrix": cm}, fh)

        mlflow.log_artifacts(artifact_dir, artifact_path="confusion_matrix")
        logger.info("Saved and logged confusion matrix artifact to MLflow.")

    @staticmethod
    def log_class_distribution_spark(train_df, test_df, label_col="behavior_type_label", artifact_dir="artifacts/class_distribution"):
        os.makedirs(artifact_dir, exist_ok=True)
        train_counts_rows = train_df.groupBy(label_col).count().collect()
        test_counts_rows = test_df.groupBy(label_col).count().collect()

        train_counts = {r[label_col]: int(r["count"]) for r in train_counts_rows}
        test_counts = {r[label_col]: int(r["count"]) for r in test_counts_rows}

        labels = sorted(list(set(list(train_counts.keys()) + list(test_counts.keys()))))
        train_vals = [train_counts.get(l, 0) for l in labels]
        test_vals = [test_counts.get(l, 0) for l in labels]

        # plot
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([xi - 0.2 for xi in x], train_vals, width=0.4, label="train")
        ax.bar([xi + 0.2 for xi in x], test_vals, width=0.4, label="test")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("count")
        ax.set_title("Class distribution (train vs test)")
        ax.legend()
        plt.tight_layout()
        fname = os.path.join(artifact_dir, "class_distribution.png")
        fig.savefig(fname)
        plt.close(fig)

        mlflow.log_artifacts(artifact_dir, artifact_path="class_distribution")
        mlflow.log_dict({"train_counts": train_counts, "test_counts": test_counts}, "class_distribution/counts.json")
        logger.info("Logged class distribution artifacts and counts to MLflow.")

    @staticmethod
    def log_feature_stats_histograms_spark(train_df, test_df, features: List[str], bins: int, histogram_sample_size: Optional[int] = None, artifact_dir="artifacts/feature_distributions"):
        """
        For each feature:
        - compute Spark summary (count/mean/std/min/max)
        - compute approxQuantile edges on train (bins+1 quantiles)
        - compute counts per bin via Spark filters (train and test), optionally sampling both DataFrames to limit driver work
        - plot the hist using matplotlib
        """
        os.makedirs(artifact_dir, exist_ok=True)
        feature_stats = []

        logger.info("Starting feature histograms: bins=%d, histogram_sample_size=%s", bins, str(histogram_sample_size))

        # optionally sample train/test before histogramming (keeps counts but approximate)
        def maybe_sample(df, sample_size):
            if sample_size is None:
                return df
            total = df.count()
            if total <= sample_size:
                return df
            fraction = float(sample_size) / float(total)
            fraction = max(min(fraction, 1.0), 0.000001)
            return df.sample(False, fraction, seed=42)

        train_for_hist = train_df
        test_for_hist = test_df

        if histogram_sample_size:
            train_for_hist = maybe_sample(train_df, histogram_sample_size)
            test_for_hist = maybe_sample(test_df, histogram_sample_size)

        for feat in features:
            logger.debug("Processing histogram for feature: %s", feat)
            train_for_hist = train_for_hist.withColumn(feat, F.col(feat).cast(T.DoubleType()))
            test_for_hist = test_for_hist.withColumn(feat, F.col(feat).cast(T.DoubleType()))
            train_df = train_df.withColumn(feat, F.col(feat).cast(T.DoubleType()))
            test_df = test_df.withColumn(feat, F.col(feat).cast(T.DoubleType()))

            agg_train = train_df.select(
                F.count(feat).alias("count"),
                F.mean(feat).alias("mean"),
                F.stddev(feat).alias("std"),
                F.min(feat).alias("min"),
                F.max(feat).alias("max")
            ).collect()[0].asDict()

            agg_test = test_df.select(
                F.count(feat).alias("count"),
                F.mean(feat).alias("mean"),
                F.stddev(feat).alias("std"),
                F.min(feat).alias("min"),
                F.max(feat).alias("max")
            ).collect()[0].asDict()

            feature_stats.append({
                "feature": feat,
                "train_count": int(agg_train.get("count") or 0),
                "train_mean": float(agg_train.get("mean")) if agg_train.get("mean") is not None else None,
                "train_std": float(agg_train.get("std")) if agg_train.get("std") is not None else None,
                "train_min": float(agg_train.get("min")) if agg_train.get("min") is not None else None,
                "train_max": float(agg_train.get("max")) if agg_train.get("max") is not None else None,
                "test_count": int(agg_test.get("count") or 0),
                "test_mean": float(agg_test.get("mean")) if agg_test.get("mean") is not None else None,
                "test_std": float(agg_test.get("std")) if agg_test.get("std") is not None else None,
                "test_min": float(agg_test.get("min")) if agg_test.get("min") is not None else None,
                "test_max": float(agg_test.get("max")) if agg_test.get("max") is not None else None,
            })

            # compute robust edges using approxQuantile on train (use full train_df for quantiles)
            try:
                quantiles = train_df.stat.approxQuantile(feat, [i / bins for i in range(bins + 1)], 0.01)
            except Exception:
                qmin = agg_train.get("min")
                qmax = agg_train.get("max")
                if qmin is None or qmax is None or qmin == qmax:
                    quantiles = [qmin or 0.0, qmax or 0.0]
                else:
                    step = (qmax - qmin) / bins
                    quantiles = [qmin + i * step for i in range(bins + 1)]

            # deduplicate edges
            edges = []
            for v in quantiles:
                if v is None:
                    continue
                if not edges or abs(v - edges[-1]) > 1e-12:
                    edges.append(v)
            if len(edges) < 2:
                logger.debug("Skipping histogram for feature %s because edges < 2", feat)
                continue

            train_counts = []
            test_counts = []
            for i in range(len(edges) - 1):
                left = edges[i]
                right = edges[i + 1]
                if i < len(edges) - 2:
                    cond = (F.col(feat) >= left) & (F.col(feat) < right)
                else:
                    cond = (F.col(feat) >= left) & (F.col(feat) <= right)
                cnt_t = train_for_hist.filter(cond).count()
                cnt_e = test_for_hist.filter(cond).count()
                train_counts.append(int(cnt_t))
                test_counts.append(int(cnt_e))

            # plot aggregated histogram
            fig, ax = plt.subplots(figsize=(6, 4))
            centers = []
            for i in range(len(edges) - 1):
                l = edges[i]
                r = edges[i + 1]
                centers.append((l + r) / 2 if (l is not None and r is not None) else 0.0)
            ax.bar([c - 0.15 for c in centers], train_counts, width=0.3, alpha=0.6, label="train")
            ax.bar([c + 0.15 for c in centers], test_counts, width=0.3, alpha=0.6, label="test")
            ax.set_title(f"Hist: {feat}")
            ax.set_xlabel(feat)
            ax.legend()
            plt.tight_layout()
            fname = os.path.join(artifact_dir, f"{feat}_hist.png")
            fig.savefig(fname)
            plt.close(fig)

        stats_path = os.path.join(artifact_dir, "feature_stats.json")
        with open(stats_path, "w") as fh:
            json.dump(feature_stats, fh, default=lambda x: None)
        mlflow.log_artifacts(artifact_dir, artifact_path="feature_distributions")
        logger.info("Logged feature distribution artifacts to MLflow for %d features", len(feature_stats))

    @staticmethod
    def build_html_summary_artifact(base_dirs: List[str], out_html: str = "artifacts/summary_report.html"):
        """
        Merge images found in base_dirs into a small HTML report and log it as an MLflow artifact.
        """
        html_parts = ["<html><head><meta charset='utf-8'><title>Training summary</title></head><body>"]
        html_parts.append(f"<h1>Training summary - {datetime.utcnow().isoformat()}Z</h1>")

        for d in base_dirs:
            if not os.path.isdir(d):
                continue
            html_parts.append(f"<h2>{os.path.basename(d)}</h2>")
            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    rel = os.path.join(d, fname)
                    html_parts.append(f"<div style='margin:8px'><img src='{rel}' style='max-width:900px'/></div>")

            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith(".json"):
                    path = os.path.join(d, fname)
                    try:
                        with open(path, "r") as fh:
                            data = json.load(fh)
                        html_parts.append(f"<pre style='max-width:900px; white-space:pre-wrap; background:#f7f7f7; padding:8px;'>")
                        html_parts.append(json.dumps(data, indent=2))
                        html_parts.append("</pre>")
                    except Exception:
                        continue

        html_parts.append("</body></html>")
        os.makedirs(os.path.dirname(out_html), exist_ok=True)
        with open(out_html, "w", encoding="utf-8") as fh:
            fh.write("\n".join(html_parts))
        mlflow.log_artifact(out_html, artifact_path="summary_report")
        logger.info("Built HTML summary artifact and logged to MLflow: %s", out_html)

    @staticmethod
    def log_readme_for_inference(out_dir="artifacts/readme"):
        """
        Write a short README explaining how to load preprocessing and model artifacts from MLflow.
        """
        os.makedirs(out_dir, exist_ok=True)
        readme_path = os.path.join(out_dir, "README.md")
        # Use indented code blocks inside README (avoid triple-backtick collisions in Markdown renderers)
        text = """# Inference README

This run stored preprocessing and model artifacts using MLflow.

Load the preprocessing and model in another environment (Python) with MLflow:

    import mlflow
    # Use the run id from the training run or models:/name@stage endpoints
    run_id = "<RUN_ID>"

    # Load imputer (Spark)
    imputer_model = mlflow.spark.load_model(f"runs:/{run_id}/preprocessing/imputer")

    # Load scaler (Spark)
    scaler_model = mlflow.spark.load_model(f"runs:/{run_id}/preprocessing/scaler")

    # Load trained model (Spark)
    model = mlflow.spark.load_model(f"runs:/{run_id}/model")

Notes:
- These are Spark ML models. Use mlflow.spark.load_model to load them back.
- If you promoted a registered model to a stage, you can load by models:/<model_name>/<stage>.
"""
        with open(readme_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        mlflow.log_artifacts(out_dir, artifact_path="readme")
        logger.info("Logged README artifact for inference to MLflow.")

    # ------------------------
    # Utility helpers
    # ------------------------
    @staticmethod
    def read_aggregated_table(
        spark: SparkSession,
        selected_cols: Optional[List[str]] = None,
        schema: str = "captaima",
        table: str = "aggregated_ais_data",
        where_clause: Optional[str] = None,
        fetchsize: int = 1000,
        partition_column: str = "primary_key"
    ) -> DataFrame:
        """
        Robust JDBC reader that mirrors the working pattern in your repository:
        - builds jdbc_url from POSTGRES_* env vars
        - optionally does partitioned JDBC read by primary_key bounds (fast & memory-safe)
        - falls back to a single JDBC read with fetchsize if partitioning not possible
        - sanitizes column names after the read
        """
        logger.info("Starting JDBC read for %s.%s (selected_cols=%s)", schema, table, selected_cols)
        # env vars (same precedence as your working function)
        pg_host = os.getenv("POSTGRES_CONTAINER_HOST", os.getenv("POSTGRES_HOST", "localhost"))
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB")
        pg_user = os.getenv("POSTGRES_USER")
        pg_pass = os.getenv("POSTGRES_PASSWORD")

        if not (pg_db and pg_user and pg_pass):
            logger.error("Missing Postgres connection env vars (POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD).")
            raise RuntimeError("Missing Postgres connection env vars (POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD).")

        jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"

        # Build the subquery used as dbtable (so we can limit the columns)
        added_partition_col = False
        if selected_cols:
            selected_cols_for_read = list(selected_cols)
            if partition_column not in selected_cols_for_read:
                selected_cols_for_read.append(partition_column)
                added_partition_col = True
            select_sql = ", ".join(selected_cols_for_read)
        else:
            select_sql = "*"
            added_partition_col = False

        where_sql = f" where {where_clause}" if where_clause else ""
        dbtable_subquery = f"(select {select_sql} from {schema}.{table}{where_sql}) as subq"

        # sanitize helper (copied from your working function)
        def sanitize_column_names(df: DataFrame) -> DataFrame:
            try:
                original = list(df.columns)
                safe_names = []
                for i, cname in enumerate(original):
                    s = str(cname) if cname is not None else ""
                    if s.strip() == "" or s.isdigit():
                        s = f"col_{i}"
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

        # Try to compute numeric primary_key bounds for partitioning (non-fatal)
        bounds_tbl = f"(select min(primary_key) as min_pk, max(primary_key) as max_pk from {schema}.{table}{where_sql}) as boundsq"
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
                .option("fetchsize", str(fetchsize))
                .load()
            )
            row = bounds_df.limit(1).collect()
            if row:
                row0 = row[0].asDict()
                min_pk = row0.get("min_pk", None)
                max_pk = row0.get("max_pk", None)
        except Exception as eb:
            logger.debug("Could not fetch bounds for partitioning (non-fatal): %s", eb)
            min_pk = max_pk = None

        # try interpret bounds as ints
        min_int = max_int = None
        try:
            if min_pk is not None and max_pk is not None:
                min_int = int(min_pk)
                max_int = int(max_pk)
        except Exception:
            min_int = max_int = None

        # perform the actual read: partitioned if possible, fallback otherwise
        try:
            logger.debug("JDBC dbtable_subquery: %s", dbtable_subquery)
            if min_int is not None and max_int is not None and min_int < max_int:
                sc = spark.sparkContext
                default_parallel = getattr(sc, "defaultParallelism", 2000) or 2000
                num_partitions = min(1000, max(8, int(default_parallel) * 2))
                logger.info(
                    "Using partitioned JDBC read on %s.%s [%s..%s] with %d partitions (sc.defaultParallelism=%s)",
                    schema, table, min_int, max_int, num_partitions, default_parallel
                )

                df = (
                    spark.read
                    .format("jdbc")
                    .option("url", jdbc_url)
                    .option("dbtable", dbtable_subquery)
                    .option("user", pg_user)
                    .option("password", pg_pass)
                    .option("driver", "org.postgresql.Driver")
                    .option("fetchsize", str(fetchsize))
                    .option("partitionColumn", partition_column)
                    .option("lowerBound", str(min_int))
                    .option("upperBound", str(max_int))
                    .option("numPartitions", str(num_partitions))
                    .load()
                )
            else:
                logger.info("Using single JDBC read with fetchsize=%d for %s.%s (no numeric bounds available)", fetchsize, schema, table)
                df = (
                    spark.read
                    .format("jdbc")
                    .option("url", jdbc_url)
                    .option("dbtable", dbtable_subquery)
                    .option("user", pg_user)
                    .option("password", pg_pass)
                    .option("driver", "org.postgresql.Driver")
                    .option("fetchsize", str(fetchsize))
                    .load()
                )

            # sanitize columns and return
            df = sanitize_column_names(df)
            try:
                logger.info("Read DataFrame from %s.%s with %d partitions and columns=%s", schema, table, df.rdd.getNumPartitions(), df.columns)
            except Exception:
                logger.debug("Could not log partitions/columns (non-fatal).")

            # If we added the partition_column only to make partitioning possible, drop it now
            if added_partition_col and (selected_cols is not None):
                if partition_column in df.columns:
                    df = df.drop(partition_column)
                else:
                    for c in df.columns:
                        if c.startswith(partition_column):
                            df = df.drop(c)
                            break

            logger.info("Completed JDBC read and sanitized columns.")
            return df

        except Exception as e:
            logger.exception("Failed to read aggregated table %s.%s via JDBC: %s", schema, table, e)
            raise

    # -------------------------
    # Label counts (simple)
    # -------------------------
    @staticmethod
    def get_label_counts(df: DataFrame, labels: List[str]) -> Dict[str, int]:
        """
        Return a dict {label: count} for the requested labels (zero if missing).
        """
        logger.debug("Computing label counts for labels: %s", labels)
        counts = {lbl: 0 for lbl in labels}
        grouped = df.filter(F.col("behavior_type_label").isin(labels)).groupBy("behavior_type_label").count()
        for row in grouped.collect():
            counts[row["behavior_type_label"]] = int(row["count"])
        del grouped
        logger.info("Label counts: %s", counts)
        return counts

    # -------------------------
    # Balanced random sampling per label
    # -------------------------
    @staticmethod
    def sample_random_balanced_spark(df: DataFrame, labels: List[str], per_label_n: int, seed: int = 42) -> DataFrame:
        """
        For each label: df.filter(label).orderBy(rand(seed)).limit(per_label_n)
        Union results and return DataFrame.
        """
        logger.info("Sampling %d rows per label for labels=%s", per_label_n, labels)
        sampled_parts = []
        for lbl in labels:
            part = df.filter(F.col("behavior_type_label") == lbl).orderBy(rand(seed)).limit(per_label_n)
            sampled_parts.append(part)
        if not sampled_parts:
            logger.warning("No sampled parts found, returning empty df")
            return df.limit(0)
        result = sampled_parts[0]
        for p in sampled_parts[1:]:
            result = result.unionByName(p, allowMissingColumns=True)
        logger.info("Balanced sampling complete; returning sampled DataFrame")
        return result

    # -------------------------
    # Deterministic stratified train/test split
    # -------------------------
    @staticmethod
    def stratified_train_test_split(sampled_df: DataFrame, label_col: str = "behavior_type_label", test_size: float = 0.28, seed: int = 42) -> Tuple[DataFrame, DataFrame]:
        """
        Deterministic stratified split by partitioning rows by label and using row_number()
        ordered by rand(seed) to split into train/test exact counts per label.
        Returns (train_df, test_df).
        """
        logger.info("Performing stratified train/test split: test_size=%.3f, seed=%d", test_size, seed)
        counts_rows = sampled_df.groupBy(label_col).count().collect()
        label_counts = {r[label_col]: int(r["count"]) for r in counts_rows}

        w = Window.partitionBy(label_col).orderBy(rand(seed))
        df_with_rownum = sampled_df.withColumn("row_num", row_number().over(w))

        # thresholds per label: floor((1-test_size) * count)
        spark = sampled_df.sparkSession
        rows = [(lbl, int((1.0 - float(test_size)) * label_counts[lbl])) for lbl in label_counts]
        thresh_schema = T.StructType([
            T.StructField(label_col, T.StringType(), False),
            T.StructField("train_count", T.IntegerType(), False)
        ])
        thresholds_df = spark.createDataFrame(rows, schema=thresh_schema)

        joined = df_with_rownum.join(thresholds_df, on=label_col, how="left")
        train_df = joined.filter(F.col("row_num") <= F.col("train_count")).drop("row_num", "train_count")
        test_df = joined.filter(F.col("row_num") > F.col("train_count")).drop("row_num", "train_count")
        logger.info("Completed stratified split; train_count=%d, test_count=%d", train_df.count(), test_df.count())
        return train_df, test_df

    # -------------------------
    # MLflow register helper
    # -------------------------
    @staticmethod
    def mlflow_register_model(run_id: str, artifact_path: str, registered_model_name: str) -> Optional[Dict]:
        """
        Create or update a model version in MLflow Model Registry. Returns dict with info or None.
        """
        client = MlflowClient()
        model_source = f"runs:/{run_id}/{artifact_path}"
        try:
            try:
                client.get_registered_model(registered_model_name)
            except Exception:
                try:
                    client.create_registered_model(registered_model_name)
                except Exception:
                    pass
            mv = client.create_model_version(
                name=registered_model_name,
                source=model_source,
                run_id=run_id
            )
            logger.info("Created model version for %s from run %s", registered_model_name, run_id)
            return {"name": mv.name, "version": mv.version, "stage": mv.current_stage}
        except Exception:
            logger.exception("Failed to register model to MLflow registry for %s", registered_model_name)
            return None

    # ------------------------
    # 1) LOITERING vs TRANSSHIPMENT SVM (renamed)
    # ------------------------
    def train_loitering_transshipment_svm(per_label_n=None, test_size=0.20, random_state=42,
                                max_iteration_steps=20, regParam=None,
                                experiment_name=None, register_model_name=None,
                                number_of_folds: int = 5,
                                impute_strategy: str = "median",
                                histogram_bins: int = 30,
                                histogram_sample_size: Optional[int] = 20000):
        """
        Train OneVsRest(LinearSVC) for LOITERING vs TRANSSHIPMENT.
        """
        logger.info("=== START train_loitering_transshipment_svm ===")
        logger.info("Parameters: per_label_n=%s, test_size=%s, random_state=%s, max_iteration_steps=%s",
                    str(per_label_n), str(test_size), str(random_state), str(max_iteration_steps))

        # Defensive: try to clear any active MLflow run (best-effort)
        TrainModelService._ensure_no_active_mlflow_run()

        run_id = None
        registration_info = None
        try:
            spark = SparkSessionInitializer.init_spark_session("training-loitering-transshipment-spark")
            logger.info("Spark session initialized for Loitering vs Transshipment training")

            cols = ["behavior_type_label"] + FEATURE_COLUMNS
            df = TrainModelService.read_aggregated_table(spark, selected_cols=cols)

            original_labels = ["LOITERING", "TRANSSHIPMENT"]
            df_filtered = df.filter(F.col("behavior_type_label").isin(original_labels))
            logger.info("Filtered DataFrame for labels: %s", original_labels)

            counts = TrainModelService.get_label_counts(df_filtered, original_labels)
            min_count = min(counts.values()) if counts else 0
            if min_count == 0:
                logger.warning("One or more classes have zero rows: %s", counts)
                return {"error": "One of the classes has zero rows", "counts": counts}

            if per_label_n is None:
                per_label_n = int(min_count)
            else:
                per_label_n = int(per_label_n)
                if per_label_n > min_count:
                    logger.warning("Requested per_label_n=%d greater than smallest class count=%d", per_label_n, min_count)
                    return {"error": "per_label_n greater than smallest class count", "smallest_label_count": min_count}

            logger.info("Sampling %d rows per class", per_label_n)
            sampled = TrainModelService.sample_random_balanced_spark(df_filtered, original_labels, per_label_n, seed=int(random_state))
            if sampled.count() == 0:
                logger.warning("No rows sampled")
                return {"error": "No rows sampled", "counts": counts}

            train_df, test_df = TrainModelService.stratified_train_test_split(
                sampled, label_col="behavior_type_label", test_size=float(test_size), seed=int(random_state)
            )

            TrainModelService.log_class_distribution_spark(train_df, test_df, label_col="behavior_type_label")

            indexer = StringIndexer(inputCol="behavior_type_label", outputCol="label", handleInvalid="keep")
            indexer_model = indexer.fit(train_df)
            train_idx = indexer_model.transform(train_df)
            test_idx = indexer_model.transform(test_df)
            label_order = indexer_model.labels
            logger.info("Indexing labels; label_order=%s", label_order)

            # CLEAN + IMPUTE (now returns imputer_model)
            logger.info("Running clean_and_impute with strategy=%s", impute_strategy)
            train_idx, test_idx, used_features, imputer_model = TrainModelService.clean_and_impute(
                train_idx, test_idx, FEATURE_COLUMNS, impute_strategy=impute_strategy
            )
            if not used_features:
                logger.warning("No usable features after cleaning.")
                return {"error": "No usable features after cleaning (all dropped or empty)."}

            mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
            logger.info("Logged used_features to MLflow: %d features", len(used_features))

            # feature stats and histograms (with sampling)
            TrainModelService.log_feature_stats_histograms_spark(
                train_idx, test_idx, used_features, bins=histogram_bins, histogram_sample_size=histogram_sample_size
            )

            # assemble + scale
            assembler = VectorAssembler(inputCols=used_features, outputCol="features_raw", handleInvalid="keep")
            train_asm = assembler.transform(train_idx)
            test_asm = assembler.transform(test_idx)

            scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
            scaler_model = scaler.fit(train_asm)
            train_scaled = scaler_model.transform(train_asm)
            test_scaled = scaler_model.transform(test_asm)
            logger.info("Assembled and scaled features. Train count=%d, Test count=%d", train_scaled.count(), test_scaled.count())

            # estimator
            svc = LinearSVC(featuresCol="features", labelCol="label", maxIter=int(max_iteration_steps),
                            regParam=float(regParam) if regParam is not None else 0.0)
            ovr = OneVsRest(classifier=svc, labelCol="label", featuresCol="features")

            # MLflow run start (defensive)
            experiment_name = experiment_name or f"SVM_Loitering_vs_Transshipment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            # try to ensure once more; use the boolean to decide nested
            ok_to_start_non_nested = TrainModelService._ensure_no_active_mlflow_run()
            mlflow.set_experiment(experiment_name)
            nested_flag = not ok_to_start_non_nested
            logger.info("Starting MLflow run under experiment '%s' (nested=%s)", experiment_name, nested_flag)

            with mlflow.start_run(run_name=f"SVM_Loitering_vs_Transshipment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                                nested=nested_flag):
                run_id = mlflow.active_run().info.run_id
                logger.info("MLflow run started: run_id=%s (nested=%s)", run_id, nested_flag)

                # log params
                mlflow.log_param("model_type", "SVM_OneVsRest_LinearSVC")
                mlflow.log_param("labels_original", ",".join(original_labels))
                mlflow.log_param("label_index_order", ",".join(label_order))
                mlflow.log_param("per_label_n", per_label_n)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("max_iteration_steps", max_iteration_steps)
                mlflow.log_param("impute_strategy", impute_strategy)
                mlflow.log_param("histogram_bins", histogram_bins)
                mlflow.log_param("histogram_sample_size", histogram_sample_size)
                mlflow.log_param("used_features_count", len(used_features))
                if regParam is not None:
                    mlflow.log_param("regParam", regParam)

                # persist imputer and scaler models as Spark MLflow artifacts (loadable via mlflow.spark.load_model)
                try:
                    mlflow.spark.log_model(imputer_model, artifact_path="preprocessing/imputer")
                    logger.info("Logged imputer_model to MLflow as preprocessing/imputer")
                except Exception as ex:
                    logger.warning("Could not log imputer_model as spark model: %s", ex)
                try:
                    mlflow.spark.log_model(scaler_model, artifact_path="preprocessing/scaler")
                    logger.info("Logged scaler_model to MLflow as preprocessing/scaler")
                except Exception as ex:
                    logger.warning("Could not log scaler_model as spark model: %s", ex)

                # also save a README explaining how to load artifacts
                TrainModelService.log_readme_for_inference(out_dir="artifacts/readme")

                # manual folds
                weights = [1.0 / number_of_folds] * number_of_folds
                folds = train_idx.randomSplit(weights, seed=int(random_state))
                logger.info("Starting manual cross-validation with %d folds", number_of_folds)

                for fold_idx in range(len(folds)):
                    logger.info("CV fold %d/%d: preparing train/validation parts", fold_idx + 1, number_of_folds)
                    val = folds[fold_idx]
                    train_parts = [folds[j] for j in range(len(folds)) if j != fold_idx]
                    train_fold = train_parts[0]
                    for extra in train_parts[1:]:
                        train_fold = train_fold.union(extra)

                    train_fold_asm = assembler.transform(train_fold)
                    val_asm = assembler.transform(val)
                    train_fold_scaled = scaler_model.transform(train_fold_asm)
                    val_scaled = scaler_model.transform(val_asm)

                    logger.info("Fitting model for CV fold %d", fold_idx + 1)
                    model_fold = ovr.fit(train_fold_scaled)
                    logger.info("Predicting validation fold %d", fold_idx + 1)
                    pred_val = model_fold.transform(val_scaled)
                    rows = pred_val.groupBy("label", "prediction").count().collect()
                    report_cv = TrainModelService.compute_metrics_from_confusion_rows(rows, label_order)
                    TrainModelService.log_metrics_from_report(report_cv, prefix="cv", step=fold_idx)
                    logger.info("Completed CV fold %d: accuracy=%.4f", fold_idx + 1, float(report_cv.get("accuracy", 0.0)))

                # final model trained on full train
                logger.info("Training final model on full train set")
                model = ovr.fit(train_scaled)

                # train aggregated metrics
                logger.info("Evaluating on train set")
                rows_train = model.transform(train_scaled).groupBy("label", "prediction").count().collect()
                report_train = TrainModelService.compute_metrics_from_confusion_rows(rows_train, label_order)
                TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                # test aggregated metrics
                logger.info("Evaluating on test set")
                rows_test = model.transform(test_scaled).groupBy("label", "prediction").count().collect()
                report_test = TrainModelService.compute_metrics_from_confusion_rows(rows_test, label_order)
                TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                # confusion matrix artifact
                TrainModelService.plot_confusion_matrix_and_log(report_test)

                # log used_features again
                mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")

                # log classifier model
                logger.info("Logging trained model to MLflow")
                mlflow.spark.log_model(spark_model=model, artifact_path="model")

                # build summary HTML and log
                base_dirs = ["artifacts/class_distribution", "artifacts/feature_distributions", "artifacts/confusion_matrix", "used_features", "artifacts/readme"]
                TrainModelService.build_html_summary_artifact(base_dirs, out_html="artifacts/summary_report.html")

                run_id = mlflow.active_run().info.run_id
                logger.info("Finished MLflow run: run_id=%s", run_id)

                if register_model_name:
                    logger.info("Registering model name=%s", register_model_name)
                    registration_info = TrainModelService.mlflow_register_model(run_id, "model", register_model_name)
                    logger.info("Registration info: %s", registration_info)

            # final summary values
            acc = float(report_test.get("accuracy", 0.0))
            f1 = float(report_test.get("macro_avg", {}).get("f1-score", 0.0))
            prec = float(report_test.get("macro_avg", {}).get("precision", 0.0))
            rec = float(report_test.get("macro_avg", {}).get("recall", 0.0))

            logger.info("=== END train_loitering_transshipment_svm: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id, "registered_model": registration_info}

        except Exception as e:
            logger.exception("Error in train_loitering_transshipment_svm: %s", e)
            # Ensure MLflow run cleaned up
            TrainModelService._end_mlflow_if_active()
            try:
                del df
                del df_filtered
                del sampled
                del train_df
                del test_df
                del train_idx
                del test_idx
                del train_asm
                del test_asm
                del train_scaled
                del test_scaled
            except Exception:
                pass
            return {"error": str(e), "trace": traceback.format_exc()}

    # ------------------------------------------------------------------
    # 2) train_loitering_stopping_model (RandomForest)
    # ------------------------------------------------------------------
    def train_loitering_stopping_model(per_label_n=None, test_size=0.20, random_state=42,
                                n_estimators=100, max_depth=None,
                                experiment_name=None, register_model_name=None,
                                number_of_folds: int = 5,
                                impute_strategy: str = "median",
                                histogram_bins: int = 30,
                                histogram_sample_size: Optional[int] = 20000):
        logger.info("=== START train_loitering_stopping_model ===")
        logger.info("Parameters: per_label_n=%s, test_size=%s, n_estimators=%s, max_depth=%s",
                    str(per_label_n), str(test_size), str(n_estimators), str(max_depth))

        # Defensive MLflow cleanup
        TrainModelService._ensure_no_active_mlflow_run()

        run_id = None
        registration_info = None
        try:
            spark = SparkSessionInitializer.init_spark_session("training-loitering-stopping-spark")
            logger.info("Spark session initialized for Loitering vs Stopping training")

            cols = ["behavior_type_label"] + FEATURE_COLUMNS
            df = TrainModelService.read_aggregated_table(spark, selected_cols=cols)

            original_labels = ["LOITERING", "STOPPING"]
            df_filtered = df.filter(F.col("behavior_type_label").isin(original_labels))

            counts = TrainModelService.get_label_counts(df_filtered, original_labels)
            min_count = min(counts.values()) if counts else 0
            if min_count == 0:
                logger.warning("One or more classes have zero rows: %s", counts)
                return {"error": "One of the classes has zero rows", "counts": counts}

            if per_label_n is None:
                per_label_n = int(min_count)
            else:
                per_label_n = int(per_label_n)
                if per_label_n > min_count:
                    logger.warning("Requested per_label_n=%d greater than smallest class count=%d", per_label_n, min_count)
                    return {"error": "per_label_n greater than smallest class count", "smallest_label_count": min_count}

            sampled = TrainModelService.sample_random_balanced_spark(df_filtered, original_labels, per_label_n, seed=int(random_state))
            if sampled.count() == 0:
                logger.warning("No rows sampled")
                return {"error": "No rows sampled", "counts": counts}

            train_df, test_df = TrainModelService.stratified_train_test_split(
                sampled, label_col="behavior_type_label", test_size=float(test_size), seed=int(random_state)
            )

            TrainModelService.log_class_distribution_spark(train_df, test_df, label_col="behavior_type_label")

            indexer = StringIndexer(inputCol="behavior_type_label", outputCol="label", handleInvalid="keep")
            indexer_model = indexer.fit(train_df)
            train_idx = indexer_model.transform(train_df)
            test_idx = indexer_model.transform(test_df)
            label_order = indexer_model.labels

            train_idx, test_idx, used_features, imputer_model = TrainModelService.clean_and_impute(
                train_idx, test_idx, FEATURE_COLUMNS, impute_strategy=impute_strategy
            )
            if not used_features:
                logger.warning("No usable features after cleaning.")
                return {"error": "No usable features after cleaning (all dropped or empty)."}

            mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
            TrainModelService.log_feature_stats_histograms_spark(
                train_idx, test_idx, used_features, bins=histogram_bins, histogram_sample_size=histogram_sample_size
            )

            assembler = VectorAssembler(inputCols=used_features, outputCol="features_raw", handleInvalid="keep")
            train_asm = assembler.transform(train_idx)
            test_asm = assembler.transform(test_idx)

            scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
            scaler_model = scaler.fit(train_asm)
            train_scaled = scaler_model.transform(train_asm)
            test_scaled = scaler_model.transform(test_asm)
            logger.info("Assembled and scaled features for RF. Train count=%d, Test count=%d", train_scaled.count(), test_scaled.count())

            numTrees = int(n_estimators)
            depth = int(max_depth) if max_depth is not None else 5
            rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=numTrees, maxDepth=depth, seed=int(random_state))

            experiment_name = experiment_name or f"RF_LoiteringStopping_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            # defensive check and nested decision
            ok_to_start_non_nested = TrainModelService._ensure_no_active_mlflow_run()
            mlflow.set_experiment(experiment_name)
            nested_flag = not ok_to_start_non_nested
            logger.info("Starting MLflow run under experiment '%s' (nested=%s)", experiment_name, nested_flag)

            with mlflow.start_run(run_name=f"RF_LoiteringStopping_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                                nested=nested_flag):
                run_id = mlflow.active_run().info.run_id
                logger.info("MLflow run started: run_id=%s (nested=%s)", run_id, nested_flag)

                mlflow.log_param("model_type", "RandomForest")
                mlflow.log_param("labels_original", ",".join(original_labels))
                mlflow.log_param("label_index_order", ",".join(label_order))
                mlflow.log_param("per_label_n", per_label_n)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("numTrees", numTrees)
                mlflow.log_param("maxDepth", depth)
                mlflow.log_param("impute_strategy", impute_strategy)
                mlflow.log_param("histogram_bins", histogram_bins)
                mlflow.log_param("histogram_sample_size", histogram_sample_size)
                mlflow.log_param("used_features_count", len(used_features))

                # log preprocessing models (imputer + scaler) as spark models
                try:
                    mlflow.spark.log_model(imputer_model, artifact_path="preprocessing/imputer")
                    logger.info("Logged imputer_model to MLflow")
                except Exception as ex:
                    logger.warning("Could not log imputer_model: %s", ex)
                try:
                    mlflow.spark.log_model(scaler_model, artifact_path="preprocessing/scaler")
                    logger.info("Logged scaler_model to MLflow")
                except Exception as ex:
                    logger.warning("Could not log scaler_model: %s", ex)

                TrainModelService.log_readme_for_inference(out_dir="artifacts/readme")

                # folds
                weights = [1.0 / number_of_folds] * number_of_folds
                folds = train_idx.randomSplit(weights, seed=int(random_state))
                logger.info("Starting manual cross-validation with %d folds", number_of_folds)
                for fold_idx in range(len(folds)):
                    logger.info("CV fold %d/%d: preparing train/validation parts", fold_idx + 1, number_of_folds)
                    val = folds[fold_idx]
                    train_parts = [folds[j] for j in range(len(folds)) if j != fold_idx]
                    train_fold = train_parts[0]
                    for extra in train_parts[1:]:
                        train_fold = train_fold.union(extra)

                    train_fold_asm = assembler.transform(train_fold)
                    val_asm = assembler.transform(val)
                    train_fold_scaled = scaler_model.transform(train_fold_asm)
                    val_scaled = scaler_model.transform(val_asm)

                    model_fold = rf.fit(train_fold_scaled)
                    pred_val = model_fold.transform(val_scaled)
                    rows_val = pred_val.groupBy("label", "prediction").count().collect()
                    report_cv = TrainModelService.compute_metrics_from_confusion_rows(rows_val, label_order)
                    TrainModelService.log_metrics_from_report(report_cv, prefix="cv", step=fold_idx)
                    logger.info("Completed CV fold %d: accuracy=%.4f", fold_idx + 1, float(report_cv.get("accuracy", 0.0)))

                model = rf.fit(train_scaled)

                rows_train = model.transform(train_scaled).groupBy("label", "prediction").count().collect()
                report_train = TrainModelService.compute_metrics_from_confusion_rows(rows_train, label_order)
                TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                rows_test = model.transform(test_scaled).groupBy("label", "prediction").count().collect()
                report_test = TrainModelService.compute_metrics_from_confusion_rows(rows_test, label_order)
                TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                TrainModelService.plot_confusion_matrix_and_log(report_test)

                mlflow.spark.log_model(spark_model=model, artifact_path="model")
                mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")

                base_dirs = ["artifacts/class_distribution", "artifacts/feature_distributions", "artifacts/confusion_matrix", "used_features", "artifacts/readme"]
                TrainModelService.build_html_summary_artifact(base_dirs, out_html="artifacts/summary_report.html")

                run_id = mlflow.active_run().info.run_id
                logger.info("Finished MLflow run: run_id=%s", run_id)
                if register_model_name:
                    logger.info("Registering model name=%s", register_model_name)
                    registration_info = TrainModelService.mlflow_register_model(run_id, "model", register_model_name)
                    logger.info("Registration info: %s", registration_info)

            acc = float(report_test.get("accuracy", 0.0))
            prec = float(report_test.get("macro_avg", {}).get("precision", 0.0))
            rec = float(report_test.get("macro_avg", {}).get("recall", 0.0))
            f1 = float(report_test.get("macro_avg", {}).get("f1-score", 0.0))

            logger.info("=== END train_loitering_stopping_model: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id, "registered_model": registration_info}

        except Exception as e:
            logger.exception("Error in train_loitering_stopping_model: %s", e)
            TrainModelService._end_mlflow_if_active()
            try:
                del df
                del df_filtered
                del sampled
                del train_df
                del test_df
                del train_idx
                del test_idx
                del train_asm
                del test_asm
                del train_scaled
                del test_scaled
            except Exception:
                pass
            return {"error": str(e), "trace": traceback.format_exc()}


    # ------------------------
    # 3) train_transhipment_model: multiclass RandomForest
    # ------------------------
    def train_transhipment_model(per_label_n=None, test_size=0.20, random_state=42,
                            n_estimators=100, max_depth=None,
                            experiment_name=None, register_model_name=None,
                            number_of_folds: int = 5,
                            impute_strategy: str = "median",
                            histogram_bins: int = 30,
                            histogram_sample_size: Optional[int] = 20000):
        logger.info("=== START train_transhipment_model ===")
        logger.info("Parameters: per_label_n=%s, test_size=%s, n_estimators=%s, max_depth=%s",
                    str(per_label_n), str(test_size), str(n_estimators), str(max_depth))

        # Defensive MLflow cleanup
        TrainModelService._ensure_no_active_mlflow_run()

        run_id = None
        registration_info = None
        try:
            spark = SparkSessionInitializer.init_spark_session("training-transshipment-spark")
            logger.info("Spark session initialized for Transshipment multiclass training")

            cols = ["behavior_type_label"] + FEATURE_COLUMNS
            df = TrainModelService.read_aggregated_table(spark, selected_cols=cols)

            original_labels = ALLOWED_LABELS
            df_filtered = df.filter(F.col("behavior_type_label").isin(original_labels))

            counts = TrainModelService.get_label_counts(df_filtered, original_labels)
            min_count = min(counts.values()) if counts else 0
            if min_count == 0:
                logger.warning("At least one class has zero rows: %s", counts)
                return {"error": "At least one class has zero rows", "counts": counts}

            if per_label_n is None:
                per_label_n = int(min_count)
            else:
                per_label_n = int(per_label_n)
                if per_label_n > min_count:
                    logger.warning("Requested per_label_n=%d greater than smallest class count=%d", per_label_n, min_count)
                    return {"error": "per_label_n greater than smallest class count", "smallest_label_count": min_count}

            sampled = TrainModelService.sample_random_balanced_spark(df_filtered, original_labels, per_label_n, seed=int(random_state))
            if sampled.count() == 0:
                logger.warning("No rows sampled")
                return {"error": "No rows sampled", "counts": counts}

            train_df, test_df = TrainModelService.stratified_train_test_split(
                sampled, label_col="behavior_type_label", test_size=float(test_size), seed=int(random_state)
            )

            TrainModelService.log_class_distribution_spark(train_df, test_df, label_col="behavior_type_label")

            indexer = StringIndexer(inputCol="behavior_type_label", outputCol="label", handleInvalid="keep")
            indexer_model = indexer.fit(train_df)
            train_idx = indexer_model.transform(train_df)
            test_idx = indexer_model.transform(test_df)
            label_order = indexer_model.labels

            train_idx, test_idx, used_features, imputer_model = TrainModelService.clean_and_impute(
                train_idx, test_idx, FEATURE_COLUMNS, impute_strategy=impute_strategy
            )
            if not used_features:
                logger.warning("No usable features after cleaning.")
                return {"error": "No usable features after cleaning (all dropped or empty)."}

            mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
            TrainModelService.log_feature_stats_histograms_spark(
                train_idx, test_idx, used_features, bins=histogram_bins, histogram_sample_size=histogram_sample_size
            )

            assembler = VectorAssembler(inputCols=used_features, outputCol="features_raw", handleInvalid="keep")
            train_asm = assembler.transform(train_idx)
            test_asm = assembler.transform(test_idx)

            scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
            scaler_model = scaler.fit(train_asm)
            train_scaled = scaler_model.transform(train_asm)
            test_scaled = scaler_model.transform(test_asm)
            logger.info("Assembled and scaled features for multiclass RF. Train count=%d, Test count=%d", train_scaled.count(), test_scaled.count())

            numTrees = int(n_estimators)
            depth = int(max_depth) if max_depth is not None else 5
            rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=numTrees, maxDepth=depth, seed=int(random_state))

            experiment_name = experiment_name or f"RF_Transshipment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            ok_to_start_non_nested = TrainModelService._ensure_no_active_mlflow_run()
            mlflow.set_experiment(experiment_name)
            nested_flag = not ok_to_start_non_nested
            logger.info("Starting MLflow run under experiment '%s' (nested=%s)", experiment_name, nested_flag)

            with mlflow.start_run(run_name=f"RF_Transshipment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                                nested=nested_flag):
                run_id = mlflow.active_run().info.run_id
                logger.info("MLflow run started: run_id=%s (nested=%s)", run_id, nested_flag)

                mlflow.log_param("model_type", "RandomForest_multiclass")
                mlflow.log_param("labels_original", ",".join(original_labels))
                mlflow.log_param("label_index_order", ",".join(label_order))
                mlflow.log_param("per_label_n", per_label_n)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("numTrees", numTrees)
                mlflow.log_param("maxDepth", depth)
                mlflow.log_param("impute_strategy", impute_strategy)
                mlflow.log_param("histogram_bins", histogram_bins)
                mlflow.log_param("histogram_sample_size", histogram_sample_size)
                mlflow.log_param("used_features_count", len(used_features))

                # persist imputer and scaler models
                try:
                    mlflow.spark.log_model(imputer_model, artifact_path="preprocessing/imputer")
                    logger.info("Logged imputer_model to MLflow")
                except Exception as ex:
                    logger.warning("Could not log imputer_model: %s", ex)
                try:
                    mlflow.spark.log_model(scaler_model, artifact_path="preprocessing/scaler")
                    logger.info("Logged scaler_model to MLflow")
                except Exception as ex:
                    logger.warning("Could not log scaler_model: %s", ex)

                TrainModelService.log_readme_for_inference(out_dir="artifacts/readme")

                # folds
                weights = [1.0 / number_of_folds] * number_of_folds
                folds = train_idx.randomSplit(weights, seed=int(random_state))
                logger.info("Starting manual cross-validation with %d folds", number_of_folds)
                for fold_idx in range(len(folds)):
                    logger.info("CV fold %d/%d: preparing train/validation parts", fold_idx + 1, number_of_folds)
                    val = folds[fold_idx]
                    train_parts = [folds[j] for j in range(len(folds)) if j != fold_idx]
                    train_fold = train_parts[0]
                    for extra in train_parts[1:]:
                        train_fold = train_fold.union(extra)

                    train_fold_asm = assembler.transform(train_fold)
                    val_asm = assembler.transform(val)
                    train_fold_scaled = scaler_model.transform(train_fold_asm)
                    val_scaled = scaler_model.transform(val_asm)

                    model_fold = rf.fit(train_fold_scaled)
                    pred_val = model_fold.transform(val_scaled)
                    rows_val = pred_val.groupBy("label", "prediction").count().collect()
                    report_cv = TrainModelService.compute_metrics_from_confusion_rows(rows_val, label_order)
                    TrainModelService.log_metrics_from_report(report_cv, prefix="cv", step=fold_idx)
                    logger.info("Completed CV fold %d: accuracy=%.4f", fold_idx + 1, float(report_cv.get("accuracy", 0.0)))

                    model = rf.fit(train_scaled)

                    rows_train = model.transform(train_scaled).groupBy("label", "prediction").count().collect()
                    report_train = TrainModelService.compute_metrics_from_confusion_rows(rows_train, label_order)
                    TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                    rows_test = model.transform(test_scaled).groupBy("label", "prediction").count().collect()
                    report_test = TrainModelService.compute_metrics_from_confusion_rows(rows_test, label_order)
                    TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                    TrainModelService.plot_confusion_matrix_and_log(report_test)

                    mlflow.spark.log_model(spark_model=model, artifact_path="model")
                    mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")

                    base_dirs = ["artifacts/class_distribution", "artifacts/feature_distributions", "artifacts/confusion_matrix", "used_features", "artifacts/readme"]
                    TrainModelService.build_html_summary_artifact(base_dirs, out_html="artifacts/summary_report.html")

                    run_id = mlflow.active_run().info.run_id
                    logger.info("Finished MLflow run: run_id=%s", run_id)
                    if register_model_name:
                        logger.info("Registering model name=%s", register_model_name)
                        registration_info = TrainModelService.mlflow_register_model(run_id, "model", register_model_name)
                        logger.info("Registration info: %s", registration_info)

            acc = float(report_test.get("accuracy", 0.0))
            prec = float(report_test.get("macro_avg", {}).get("precision", 0.0))
            rec = float(report_test.get("macro_avg", {}).get("recall", 0.0))
            f1 = float(report_test.get("macro_avg", {}).get("f1-score", 0.0))

            logger.info("=== END train_transhipment_model: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id, "registered_model": registration_info}

        except Exception as e:
            logger.exception("Error in train_transhipment_model: %s", e)
            TrainModelService._end_mlflow_if_active()
            try:
                del df
                del df_filtered
                del sampled
                del train_df
                del test_df
                del train_idx
                del test_idx
                del train_asm
                del test_asm
                del train_scaled
                del test_scaled
            except Exception:
                pass
            return {"error": str(e), "trace": traceback.format_exc()}
