# ------------------------
# Pandas and Sklearn Imports
# ------------------------
import pandas as pd
import numpy as np
import joblib
import tempfile
import pathlib
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.svm import LinearSVC as SklearnLinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
import mlflow
import mlflow.sklearn
from domain.config.database_config import db  # Flask-SQLAlchemy db

# ------------------------
# Spark Imports
# ------------------------
import os
import json
import math
import socket
import logging
from datetime import datetime
from dotenv import load_dotenv
from domain.config.data_processing.spark_session_initializer import SparkSessionInitializer
from domain.services.mlflow_service import MLFlowService

# ------------------------
# Tensorflow Imports
# ------------------------
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import csv

# ------------------------
# Misc imports
import glob
import ast
import gc
from typing import Callable, Iterable, Tuple, List, Dict, Any
# ------------------------

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

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTTING = True
except Exception:
    _HAS_PLOTTING = False

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
        Changed: use 'support_count' instead of 'support' in metric names.
        """
        labels = report["labels"]
        for lab in labels:
            cls = report["per_class"][lab]
            mlflow.log_metric(f"{prefix}_precision_{lab}", float(cls["precision"]), step=step if step is not None else None)
            mlflow.log_metric(f"{prefix}_recall_{lab}", float(cls["recall"]), step=step if step is not None else None)
            mlflow.log_metric(f"{prefix}_f1_{lab}", float(cls["f1-score"]), step=step if step is not None else None)
            # clearer name: support_count (number of true samples for that label in this evaluated set)
            mlflow.log_metric(f"{prefix}_support_count_{lab}", float(cls["support"]), step=step if step is not None else None)

        mlflow.log_metric(f"{prefix}_accuracy", float(report["accuracy"]), step=step if step is not None else None)
        mlflow.log_metric(f"{prefix}_precision_macro", float(report["macro_avg"]["precision"]), step=step if step is not None else None)
        mlflow.log_metric(f"{prefix}_recall_macro", float(report["macro_avg"]["recall"]), step=step if step is not None else None)
        mlflow.log_metric(f"{prefix}_f1_macro", float(report["macro_avg"]["f1-score"]), step=step if step is not None else None)
        logger.info("Logged MLflow metrics for %s (step=%s) : accuracy=%.4f", prefix, str(step), float(report["accuracy"]))
        
    @staticmethod
    def plot_confusion_matrix_and_log_spark(report: dict, artifact_dir="artifacts/confusion_matrix"):
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
    # 1) LOITERING vs TRANSSHIPMENT SVM (SPARK)
    # ------------------------
    def train_loitering_transshipment_svm_spark(per_label_n=None, test_size=0.20, random_state=42,
                                max_iteration_steps=20, regParam=None,
                                experiment_name=None, register_model_name=None,
                                number_of_folds: int = 5,
                                impute_strategy: str = "median",
                                histogram_bins: int = 30,
                                histogram_sample_size: Optional[int] = 20000):
        """
        Train OneVsRest(LinearSVC) for LOITERING vs TRANSSHIPMENT.
        """
        logger.info("=== START train_loitering_transshipment_svm_spark ===")
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

            # helper: safely extract objective history from a binary model and log per-step metrics
            def _extract_and_log_objective_history(onevs_model, label_order_list, mlflow_prefix, fold_number=None):
                """
                Tenta extrair objectiveHistory do OneVsRestModel (por classe) e logar
                séries de métricas por step em MLflow. Faz várias tentativas:
                1) Python-side: binary_model.summary.objectiveHistory (comum em alguns modelos)
                2) Java-side: binary_model._java_obj.summary().objectiveHistory() (fallback)
                Também faz logger.info para inspeção local antes de logar em MLflow.
                """
                try:
                    models_list = getattr(onevs_model, "models", None)
                    if not models_list:
                        logger.info("OneVsRest model has no .models attribute -> skipping objectiveHistory extraction")
                        return

                    per_class_histories = {}
                    max_len = 0

                    for idx, binary_model in enumerate(models_list):
                        label_name = label_order_list[idx] if idx < len(label_order_list) else f"class_{idx}"
                        hist = None

                        # 1) try python summary.objectiveHistory
                        try:
                            summ = getattr(binary_model, "summary", None)
                            if summ is not None:
                                hist_attr = getattr(summ, "objectiveHistory", None)
                                if callable(hist_attr):
                                    hist = hist_attr()
                                else:
                                    hist = hist_attr
                            if hist:
                                logger.info("Found objectiveHistory via python summary for label '%s' (len=%d)", label_name, len(hist))
                        except Exception as ex:
                            logger.debug("python-summary attempt failed for label %s: %s", label_name, ex)
                            hist = None

                        # 2) fallback: try java-level access binary_model._java_obj.summary().objectiveHistory()
                        if not hist:
                            try:
                                jmodel = getattr(binary_model, "_java_obj", None)
                                if jmodel is not None:
                                    # some Spark builds expose summary() as a method
                                    try:
                                        jsummary = jmodel.summary()
                                    except Exception:
                                        # sometimes it's an attribute or call fails; try property-like access
                                        try:
                                            jsummary = jmodel.summary
                                        except Exception:
                                            jsummary = None

                                    if jsummary is not None:
                                        try:
                                            jhist = jsummary.objectiveHistory()
                                            # jhist may be a Java array/Seq; convert robustly
                                            try:
                                                # py4j collection -> iterable
                                                hist = list(jhist)
                                            except Exception:
                                                # try to iterate via toArray / toList
                                                try:
                                                    hist = [float(x) for x in jhist.toArray()]
                                                except Exception:
                                                    hist = None
                                            if hist:
                                                logger.info("Found objectiveHistory via java summary for label '%s' (len=%d)", label_name, len(hist))
                                        except Exception as ex2:
                                            logger.debug("java-summary.objectiveHistory() not available for label %s: %s", label_name, ex2)
                            except Exception as ex_outer:
                                logger.debug("java-level attempt failed for label %s: %s", label_name, ex_outer)
                                hist = None

                        # normalize into list[float]
                        if hist:
                            try:
                                hist_list = [float(x) for x in hist]
                                per_class_histories[label_name] = hist_list
                                max_len = max(max_len, len(hist_list))
                            except Exception as econv:
                                logger.debug("Could not convert objectiveHistory for %s: %s", label_name, econv)
                                continue

                    if not per_class_histories:
                        logger.info("No objectiveHistory found for OneVsRest model (no per-step training metrics will be logged).")
                        return

                    # log per-class histories (with verbose logging)
                    for label_name, hist_list in per_class_histories.items():
                        metric_base = f"{mlflow_prefix}_train_step_objective_{label_name}"
                        if fold_number is not None:
                            metric_base = f"{mlflow_prefix}_fold{fold_number}_{metric_base}"
                        for step_idx, val in enumerate(hist_list):
                            logger.info("MLflow metric -> name=%s step=%d value=%.8f", metric_base, step_idx, float(val))
                            mlflow.log_metric(metric_base, float(val), step=step_idx)

                    # log mean across classes per step
                    for step_idx in range(max_len):
                        vals = []
                        for hist_list in per_class_histories.values():
                            if step_idx < len(hist_list):
                                vals.append(hist_list[step_idx])
                        if vals:
                            mean_val = float(sum(vals) / len(vals))
                            mean_name = f"{mlflow_prefix}_train_step_objective_mean"
                            if fold_number is not None:
                                mean_name = f"{mlflow_prefix}_fold{fold_number}_{mean_name}"
                            logger.info("MLflow metric -> name=%s step=%d value=%.8f", mean_name, step_idx, mean_val)
                            mlflow.log_metric(mean_name, mean_val, step=step_idx)

                except Exception as ex_final:
                    logger.exception("Unexpected error extracting objectiveHistory: %s", ex_final)
                    # Don't raise; fall back gracefully
                    return


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
                    # log per-iteration objective history for this fold (if available)
                    _extract_and_log_objective_history(model_fold, label_order, mlflow_prefix="cv", fold_number=fold_idx+1)

                    logger.info("Predicting validation fold %d", fold_idx + 1)
                    pred_val = model_fold.transform(val_scaled)
                    rows = pred_val.groupBy("label", "prediction").count().collect()
                    report_cv = TrainModelService.compute_metrics_from_confusion_rows(rows, label_order)
                    # existing CV aggregated metrics (no change)
                    TrainModelService.log_metrics_from_report(report_cv, prefix="cv", step=fold_idx)
                    logger.info("Completed CV fold %d: accuracy=%.4f", fold_idx + 1, float(report_cv.get("accuracy", 0.0)))

                # final model trained on full train
                logger.info("Training final model on full train set")
                model = ovr.fit(train_scaled)

                # log per-iteration objective history for final model (train), if available
                _extract_and_log_objective_history(model, label_order, mlflow_prefix="train", fold_number=None)

                # train aggregated metrics
                logger.info("Evaluating on train set")
                rows_train = model.transform(train_scaled).groupBy("label", "prediction").count().collect()
                report_train = TrainModelService.compute_metrics_from_confusion_rows(rows_train, label_order)
                TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                # test aggregated metrics (no per-step metrics)
                logger.info("Evaluating on test set")
                rows_test = model.transform(test_scaled).groupBy("label", "prediction").count().collect()
                report_test = TrainModelService.compute_metrics_from_confusion_rows(rows_test, label_order)
                TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                # confusion matrix artifact
                TrainModelService.plot_confusion_matrix_and_log_spark(report_test)

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

            logger.info("=== END train_loitering_transshipment_svm_spark: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id, "registered_model": registration_info}

        except Exception as e:
            logger.exception("Error in train_loitering_transshipment_svm_spark: %s", e)
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
    # 2) train_loitering_stopping_model (RandomForest) (SPARK)
    # ------------------------------------------------------------------
    def train_loitering_stopping_spark_model(per_label_n=None, test_size=0.20, random_state=42,
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

                TrainModelService.plot_confusion_matrix_and_log_spark(report_test)

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
    # 3) train_transshipment_spark_model: multiclass RandomForest (SPARK)
    # ------------------------
    def train_transshipment_spark_model(per_label_n=None, test_size=0.20, random_state=42,
                            n_estimators=100, max_depth=None,
                            experiment_name=None, register_model_name=None,
                            number_of_folds: int = 5,
                            impute_strategy: str = "median",
                            histogram_bins: int = 30,
                            histogram_sample_size: Optional[int] = 20000):
        logger.info("=== START train_transshipment_spark_model ===")
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

                TrainModelService.plot_confusion_matrix_and_log_spark(report_test)

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

            logger.info("=== END train_transshipment_spark_model: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id, "registered_model": registration_info}

        except Exception as e:
            logger.exception("Error in train_transshipment_spark_model: %s", e)
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
    # Pandas specific code
    # ------------------------        

    # ---------- Utility: read aggregated table into pandas ----------
    @staticmethod
    def read_aggregated_table_pandas(
        selected_cols: Optional[List[str]] = None,
        schema: str = "captaima",
        table: str = "aggregated_ais_data",
        where_clause: Optional[str] = None,
        chunksize: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read from captaima.aggregated_ais_data into a pandas DataFrame using a raw DB-API connection
        (engine.raw_connection()). This avoids pandas->SQLAlchemy MetaData signature incompatibility.
        """
        logger.info("Starting pandas JDBC read (raw_connection) for %s.%s (selected_cols=%s)", schema, table, selected_cols)
        if selected_cols:
            select_sql = ", ".join(selected_cols)
        else:
            select_sql = "*"
        where_sql = f" WHERE {where_clause}" if where_clause else ""
        sql = f"SELECT {select_sql} FROM {schema}.{table}{where_sql}"

        raw_conn = None
        try:
            raw_conn = db.engine.raw_connection()  # DB-API connection (psycopg, etc.)
            if chunksize:
                parts = []
                # pd.read_sql with a raw DB-API connection and chunksize returns an iterator of DataFrames
                iterator = pd.read_sql(sql, con=raw_conn, chunksize=chunksize)
                for chunk in iterator:
                    parts.append(chunk)
                df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=selected_cols or [])
            else:
                df = pd.read_sql(sql, con=raw_conn)

            # sanitize column names if needed
            df.columns = [str(c) if (c is not None and str(c).strip() != "") else f"col_{i}" for i, c in enumerate(df.columns)]
            logger.info("Completed pandas read: rows=%d cols=%s", len(df), list(df.columns))
            return df
        except Exception as e:
            logger.exception("Failed to read aggregated table via pandas (raw_connection path): %s", e)
            raise
        finally:
            # ensure raw DB-API connection closed
            try:
                if raw_conn is not None:
                    raw_conn.close()
            except Exception:
                pass


    # ---------- Sampling / splitting helpers ----------
    @staticmethod
    def sample_random_balanced_pandas(df: pd.DataFrame, labels: List[str], per_label_n: int, label_col: str = "behavior_type_label", random_state: int = 42) -> pd.DataFrame:
        """Return balanced sample with per_label_n rows for each label (concatenated)."""
        logger.info("Sampling %d rows per label (pandas) for labels=%s", per_label_n, labels)
        parts = []
        for lbl in labels:
            sub = df[df[label_col] == lbl]
            if sub.shape[0] == 0:
                logger.warning("Label %s has 0 rows", lbl)
                continue
            if per_label_n <= 0:
                continue
            # if per_label_n > available, sample without replacement up to available (keep deterministic)
            n = min(per_label_n, sub.shape[0])
            parts.append(sub.sample(n=n, random_state=int(random_state)))
        if not parts:
            return df.iloc[0:0].copy()
        res = pd.concat(parts, ignore_index=True)
        return res.sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)

    @staticmethod
    def stratified_train_test_split_pandas(df: pd.DataFrame, label_col: str = "behavior_type_label", test_size: float = 0.28, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Deterministic stratified split using sklearn StratifiedShuffleSplit.
        Returns (train_df, test_df).
        """
        logger.info("Performing stratified train/test split (pandas): test_size=%.3f, seed=%d", test_size, random_state)
        if df.empty:
            return df.copy(), df.copy()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
        y = df[label_col].values
        for train_idx, test_idx in sss.split(np.zeros(len(y)), y):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            logger.info("Stratified split done: train=%d test=%d", len(train_df), len(test_df))
            return train_df, test_df
        # fallback
        return df.copy(), df.iloc[0:0].copy()

    # ---------- Clean + impute (pandas + sklearn) ----------
    @staticmethod
    def clean_and_impute_pandas(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], impute_strategy: str = "median"):
        """
        - Convert features to numeric, coerce inf/nan to np.nan
        - Drop zero-variance features (based on train)
        - Fit SimpleImputer on train and transform both train & test
        Returns: train_imputed_df, test_imputed_df, used_features, imputer
        """
        logger.info("Starting clean_and_impute_pandas: strategy=%s, num_features=%d", impute_strategy, len(feature_cols))
        train = train_df.copy()
        test = test_df.copy()
        # ensure features exist
        available = [c for c in feature_cols if c in train.columns or c in test.columns]
        # coerce to numeric and handle inf
        for c in available:
            train[c] = pd.to_numeric(train.get(c), errors="coerce")
            test[c] = pd.to_numeric(test.get(c), errors="coerce")
            train.loc[~np.isfinite(train[c]), c] = np.nan
            test.loc[~np.isfinite(test[c]), c] = np.nan

        # zero-variance detection on train
        zero_var = []
        used_features = []
        for c in available:
            std = float(train[c].std(skipna=True)) if not train[c].isnull().all() else 0.0
            if std == 0.0 or np.isnan(std):
                zero_var.append(c)
            else:
                used_features.append(c)
        logger.info("Zero-variance features (dropped): %s", zero_var)
        if not used_features:
            logger.warning("No usable features after dropping zero-variance.")
            return None, None, [], None

        # Fit imputer on train[used_features]
        imputer = SimpleImputer(strategy=impute_strategy)
        imputer.fit(train[used_features])
        train_imp_vals = imputer.transform(train[used_features])
        test_imp_vals = imputer.transform(test[used_features])
        # put back into frames
        train_imp = train.copy()
        test_imp = test.copy()
        train_imp.loc[:, used_features] = train_imp_vals
        test_imp.loc[:, used_features] = test_imp_vals
        logger.info("Imputation completed (pandas). Used features: %s", used_features)
        return train_imp, test_imp, used_features, imputer

    # ---------- Feature histograms / stats (pandas) ----------
    @staticmethod
    def log_feature_stats_histograms_pandas(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], bins: int, histogram_sample_size: Optional[int] = None, artifact_dir="artifacts/feature_distributions"):
        os.makedirs(artifact_dir, exist_ok=True)
        feature_stats = []
        logger.info("Starting feature histograms (pandas): bins=%d, sample_size=%s", bins, str(histogram_sample_size))

        # optional sampling to limit memory/plot time
        if histogram_sample_size:
            train_for_hist = train_df.sample(n=min(histogram_sample_size, len(train_df)), random_state=42) if len(train_df) > 0 else train_df
            test_for_hist = test_df.sample(n=min(histogram_sample_size, len(test_df)), random_state=42) if len(test_df) > 0 else test_df
        else:
            train_for_hist = train_df
            test_for_hist = test_df

        for feat in features:
            if feat not in train_df.columns:
                logger.debug("Skipping histogram for missing feature: %s", feat)
                continue
            tr_ser = pd.to_numeric(train_df[feat], errors="coerce")
            te_ser = pd.to_numeric(test_df[feat], errors="coerce")
            agg_train = {"count": int(tr_ser.count()), "mean": float(tr_ser.mean()) if tr_ser.count() else None, "std": float(tr_ser.std()) if tr_ser.count() else None, "min": float(tr_ser.min()) if tr_ser.count() else None, "max": float(tr_ser.max()) if tr_ser.count() else None}
            agg_test = {"count": int(te_ser.count()), "mean": float(te_ser.mean()) if te_ser.count() else None, "std": float(te_ser.std()) if te_ser.count() else None, "min": float(te_ser.min()) if te_ser.count() else None, "max": float(te_ser.max()) if te_ser.count() else None}
            feature_stats.append({
                "feature": feat,
                "train_count": agg_train["count"],
                "train_mean": agg_train["mean"],
                "train_std": agg_train["std"],
                "train_min": agg_train["min"],
                "train_max": agg_train["max"],
                "test_count": agg_test["count"],
                "test_mean": agg_test["mean"],
                "test_std": agg_test["std"],
                "test_min": agg_test["min"],
                "test_max": agg_test["max"],
            })

            # compute quantile edges robustly
            try:
                quantiles = np.nanquantile(train_for_hist[feat].dropna().astype(float), np.linspace(0, 1, bins + 1))
                # deduplicate edges
                edges = [float(quantiles[0])]
                for v in quantiles[1:]:
                    if not np.isclose(v, edges[-1], atol=1e-12):
                        edges.append(float(v))
                if len(edges) < 2:
                    logger.debug("Skipping histogram (edges < 2) for: %s", feat)
                    continue
                # counts by bin
                train_counts, _ = np.histogram(train_for_hist[feat].dropna().astype(float), bins=edges)
                test_counts, _ = np.histogram(test_for_hist[feat].dropna().astype(float), bins=edges)
            except Exception:
                # fallback simple bins
                try:
                    train_counts, edges = np.histogram(train_for_hist[feat].dropna().astype(float), bins=bins)
                    test_counts, _ = np.histogram(test_for_hist[feat].dropna().astype(float), bins=edges)
                except Exception:
                    continue

            # plot
            fig, ax = plt.subplots(figsize=(6, 4))
            centers = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(edges)-1)]
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

    # ---------- Metrics conversion (pandas/sklearn) ----------
    @staticmethod
    def compute_metrics_from_predictions(y_true, y_pred, label_order: List[str]) -> dict:
        """
        Build report dict similar to compute_metrics_from_confusion_rows but from arrays.
        """
        labels_idx = list(range(len(label_order)))
        cm = confusion_matrix(y_true, y_pred, labels=label_order).tolist()
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=label_order, zero_division=0)
        per_class = {}
        sum_precision = sum_recall = sum_f1 = 0.0
        for i, lab in enumerate(label_order):
            per_class[lab] = {"precision": float(prec[i]), "recall": float(rec[i]), "f1-score": float(f1[i]), "support": int(sup[i])}
            sum_precision += float(prec[i])
            sum_recall += float(rec[i])
            sum_f1 += float(f1[i])
        n = len(label_order)
        macro_precision = sum_precision / n if n > 0 else 0.0
        macro_recall = sum_recall / n if n > 0 else 0.0
        macro_f1 = sum_f1 / n if n > 0 else 0.0
        accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
        report = {
            "per_class": per_class,
            "macro_avg": {"precision": macro_precision, "recall": macro_recall, "f1-score": macro_f1},
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "total_samples": int(len(y_true)),
            "labels": label_order
        }
        return report

    # ---------- Helper to persist transformer/model artifacts to MLflow ----------
    def _save_and_log_artifact(obj, artifact_subpath, name):
        """
        Save object using joblib into a temporary dir and log as MLflow artifact.
        Returns the path saved.
        """
        tmpd = tempfile.mkdtemp()
        fname = os.path.join(tmpd, f"{name}.joblib")
        joblib.dump(obj, fname)
        mlflow.log_artifact(fname, artifact_path=artifact_subpath)
        return fname

    # ---------------------------
    # pandas-friendly class distribution plot (replacement for spark version)
    # ---------------------------
    @staticmethod
    def log_class_distribution_pandas(train_df, test_df, label_col: str = "behavior_type_label", artifact_dir: str = "artifacts/class_distribution"):
        """
        Creates and logs class-distribution bar charts for train & test pandas DataFrames.
        Logs the images and a JSON with counts to MLflow under artifact_path "class_distribution".
        Returns a dict with counts.
        """
        os.makedirs(artifact_dir, exist_ok=True)
        try:
            train_counts = train_df[label_col].value_counts(dropna=False).to_dict() if train_df is not None and label_col in train_df.columns else {}
            test_counts = test_df[label_col].value_counts(dropna=False).to_dict() if test_df is not None and label_col in test_df.columns else {}

            # ensure union of labels
            labels = sorted(set(list(train_counts.keys()) + list(test_counts.keys())), key=lambda x: str(x))
            train_vals = [int(train_counts.get(l, 0)) for l in labels]
            test_vals = [int(test_counts.get(l, 0)) for l in labels]

            # plot side-by-side bars
            fig, ax = plt.subplots(figsize=(8, 4))
            x = range(len(labels))
            width = 0.35
            ax.bar([i - width/2 for i in x], train_vals, width=width, label="train")
            ax.bar([i + width/2 for i in x], test_vals, width=width, label="test")
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(l) for l in labels], rotation=45, ha="right")
            ax.set_ylabel("Count")
            ax.set_title("Class distribution (train vs test)")
            ax.legend()
            plt.tight_layout()
            img_path = os.path.join(artifact_dir, "class_distribution_train_test.png")
            fig.savefig(img_path)
            plt.close(fig)

            # log JSON counts
            counts_path = os.path.join(artifact_dir, "class_counts.json")
            payload = {"labels": [str(l) for l in labels], "train_counts": train_counts, "test_counts": test_counts}
            with open(counts_path, "w") as fh:
                json.dump(payload, fh)

            # MLflow log
            mlflow.log_artifact(img_path, artifact_path="class_distribution")
            mlflow.log_artifact(counts_path, artifact_path="class_distribution")

            return {"labels": labels, "train_counts": train_counts, "test_counts": test_counts}
        except Exception as e:
            logger.exception("Error in log_class_distribution_pandas: %s", e)
            return {"error": str(e)}
    
    
    @staticmethod
    def _make_iteration_points_int(max_value: int, max_points: int = 10) -> list:
        """
        Helper: create a small set of increasing integer checkpoints from 1..max_value.
        Returns sorted unique ints (at most max_points long).
        """
        if max_value <= 0:
            return [1]
        n_points = min(max_points, max_value)
        pts = np.unique(np.round(np.linspace(1, max_value, num=n_points)).astype(int)).tolist()
        if pts[0] < 1:
            pts[0] = 1
        return pts


    @staticmethod
    def _plot_metrics_over_iterations(iter_x, metrics_by_name: dict, title: str, outpath: str):
        """
        Plot multiple metric curves (metrics_by_name: {name: list(values)}) vs iter_x and save to outpath.
        """
        plt.figure(figsize=(8, 5))
        for name, vals in metrics_by_name.items():
            plt.plot(iter_x, vals, marker='o', label=name)
        plt.xlabel("Iteration / Trees")
        plt.ylabel("Score")
        plt.title(title)
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()

    @staticmethod
    def plot_confusion_matrix_and_log_pandas_sklearn(report: dict, artifact_dir="artifacts/confusion_matrix"):
        """
        Confusion matrix plot (pandas/sklearn friendly).
        - counts + row-% shown in every matrix cell,
        - per-class Precision & Recall shown OUTSIDE the matrix in a left column with class names,
        with P and R positioned *below* the class name to avoid overlap.
        - saves PNG + JSON and logs artifacts to MLflow (best-effort).
        Returns {"png": png_path, "json": json_path}
        """
        import numpy as _np
        import json as _json
        import matplotlib.pyplot as _plt
        import os as _os
        import mlflow as _mlflow

        _os.makedirs(artifact_dir, exist_ok=True)

        cm = report.get("confusion_matrix", [])
        labels = report.get("labels", [])
        per_class = report.get("per_class", {}) or {}

        # safe conversion to numpy array
        try:
            cm_arr = _np.array(cm, dtype=float)
        except Exception:
            cm_arr = _np.zeros((0, 0), dtype=float)

        # infer labels if missing
        if not labels:
            if cm_arr.size:
                n = cm_arr.shape[0]
                labels = [str(i) for i in range(n)]
            else:
                labels = []

        # ensure square matrix if labels present
        if cm_arr.size == 0 and labels:
            n = len(labels)
            cm_arr = _np.zeros((n, n), dtype=float)

        n_rows = cm_arr.shape[0] if cm_arr.size else len(labels)
        n_cols = cm_arr.shape[1] if cm_arr.size else len(labels)
        if n_rows == 0 and n_cols == 0:
            png_path = _os.path.join(artifact_dir, "confusion_matrix.png")
            json_path = _os.path.join(artifact_dir, "confusion_matrix.json")
            with open(json_path, "w") as fh:
                _json.dump({"labels": [], "matrix": [], "per_class": {}}, fh)
            return {"png": png_path, "json": json_path}

        # sums for percentages and fallback safe denominators
        row_sums = _np.sum(cm_arr, axis=1) if cm_arr.size else _np.zeros((n_rows,))
        col_sums = _np.sum(cm_arr, axis=0) if cm_arr.size else _np.zeros((n_cols,))
        diag = _np.diag(cm_arr) if cm_arr.size else _np.zeros((min(n_rows, n_cols),))

        # compute precision / recall per class (fallback to report['per_class'] if present)
        precision_arr = _np.zeros_like(diag, dtype=float)
        recall_arr = _np.zeros_like(diag, dtype=float)
        for idx in range(len(diag)):
            if col_sums.size > idx and col_sums[idx] > 0:
                precision_arr[idx] = float(diag[idx]) / float(col_sums[idx])
            else:
                metrics = per_class.get(labels[idx]) or per_class.get(str(labels[idx])) or {}
                precision_arr[idx] = float(metrics.get("precision") or 0.0)

            if row_sums.size > idx and row_sums[idx] > 0:
                recall_arr[idx] = float(diag[idx]) / float(row_sums[idx])
            else:
                metrics = per_class.get(labels[idx]) or per_class.get(str(labels[idx])) or {}
                recall_arr[idx] = float(metrics.get("recall") or 0.0)

        # prepare labels for display (we will show them in left column, so hide matrix yticklabels)
        display_labels = [str(l) for l in labels]
        display_labels = display_labels[:max(n_rows, n_cols)]
        if len(display_labels) < max(n_rows, n_cols):
            display_labels += [f"lbl_{i}" for i in range(len(display_labels), max(n_rows, n_cols))]

        # Layout params: make wide + tall enough to avoid cropping and overlaps
        left_width = max(2.4, 1.0 + 0.12 * max(0, n_rows))   # PR column width (inches approx.)
        right_width = max(6.0, 0.6 * n_cols)                # matrix width
        total_width = left_width + right_width
        height = max(7.0, 0.9 * n_rows + 1.5)               # taller so x labels not cut

        fig = _plt.figure(figsize=(total_width, height), constrained_layout=False)
        gs = fig.add_gridspec(1, 2, width_ratios=[left_width, right_width], wspace=0.05)

        ax_pr = fig.add_subplot(gs[0, 0])   # left column for class names and precision/recall
        ax = fig.add_subplot(gs[0, 1])      # main confusion matrix

        # Plot confusion matrix on ax
        im = ax.imshow(cm_arr, interpolation="nearest", cmap=_plt.cm.Blues)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

        # hide matrix yticklabels (we'll show class names in the left column)
        ax.set(xticks=list(range(n_cols)), yticks=list(range(n_rows)),
            xticklabels=display_labels[:n_cols], yticklabels=[''] * n_rows,
            ylabel="True label", xlabel="Predicted label",
            title="Confusion matrix (test set)")
        _plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # place ylabel slightly inside the matrix so it doesn't collide with left column
        ax.yaxis.set_label_coords(-0.06, 0.5)

        # threshold for text color
        try:
            vmax = float(cm_arr.max()) if cm_arr.size else 1.0
        except Exception:
            vmax = 1.0
        thresh = vmax / 2.0 if vmax else 0.0

        # annotate counts + percentages inside cells
        for i in range(n_rows):
            for j in range(n_cols):
                try:
                    val = float(cm_arr[i, j])
                    cnt_text = f"{int(val)}"
                except Exception:
                    val = cm_arr[i, j]
                    cnt_text = str(val)

                pct = (val / row_sums[i]) if (row_sums.size and row_sums[i] > 0) else 0.0
                pct_text = f"{pct:.1%}"

                text_color = "white" if (isinstance(val, (int, float)) and val > thresh) else "black"

                ax.text(j, i - 0.18, cnt_text, ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")
                ax.text(j, i + 0.02, pct_text, ha="center", va="center", color=text_color, fontsize=9)

        # Prepare left PR column: clean layout and alignment
        ax_pr.axis("off")
        ax_pr.set_xlim(0, 1)

        # Align ax_pr y-range with ax so labels align with rows
        ax_pr.set_ylim(ax.get_ylim())

        # compute y positions from ax (these correspond to row centers)
        y_positions = ax.get_yticks().tolist()
        # fallback if not present
        if len(y_positions) < n_rows:
            y_positions = list(range(n_rows))

        # Place class name and P/R (P and R positioned below the class name)
        # We'll render class name slightly above center, P below center, R further below.
        class_x = 0.02
        pr_x = 0.62
        for idx, y in enumerate(y_positions[:n_rows]):
            class_name = display_labels[idx] if idx < len(display_labels) else f"lbl_{idx}"
            prec = precision_arr[idx] if idx < len(precision_arr) else 0.0
            rec = recall_arr[idx] if idx < len(recall_arr) else 0.0

            # vertical offsets in data coordinates (row units)
            class_y = y - 0.20   # slightly above center
            p_y = y + 0.03       # slightly below center
            r_y = y + 0.28       # further below

            # class name (left)
            ax_pr.text(class_x, class_y, class_name, ha="left", va="center", fontsize=10, fontweight="normal")

            # Precision and Recall lines shown as percentages, below the class name
            ax_pr.text(pr_x, p_y, f"P: {prec * 100:.1f}%", ha="center", va="center", fontsize=10, family="monospace")
            ax_pr.text(pr_x, r_y, f"R: {rec * 100:.1f}%", ha="center", va="center", fontsize=10, family="monospace")

        # Final layout tweaks: ensure enough bottom margin so x ticklabels show fully
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.20, left=0.09)

        png_path = _os.path.join(artifact_dir, "confusion_matrix.png")
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        _plt.close(fig)

        # Save JSON with matrix + per-class metrics (precision/recall as decimals)
        json_path = _os.path.join(artifact_dir, "confusion_matrix.json")
        try:
            export_payload = {
                "labels": display_labels[:max(n_rows, n_cols)],
                "matrix": cm_arr.astype(int).tolist() if cm_arr.size else [],
                "per_class": per_class,
                "precision_by_class": precision_arr.tolist(),
                "recall_by_class": recall_arr.tolist()
            }
            with open(json_path, "w") as fh:
                _json.dump(export_payload, fh)
        except Exception:
            logger.exception("Failed writing confusion matrix JSON.")

        # Log artifacts to MLflow (best-effort)
        try:
            _mlflow.log_artifacts(artifact_dir, artifact_path="confusion_matrix")
            logger.info("Saved and logged confusion matrix artifact to MLflow.")
        except Exception:
            logger.debug("Could not log confusion matrix artifacts to MLflow (no active run or other error).")

        return {"png": png_path, "json": json_path}


    # ---------- Main training functions (pandas/sklearn). Each coexists with Spark ones ----------
    @staticmethod
    def train_loitering_transshipment_svm_pandas_sklearn_model(per_label_n=None, test_size=0.20, random_state=42,
                                                    max_iteration_steps=1000, C=1.0,
                                                    experiment_name=None, register_model_name=None,
                                                    number_of_folds: int = 5,
                                                    impute_strategy: str = "median",
                                                    histogram_bins: int = 30,
                                                    histogram_sample_size: Optional[int] = 20000):
        """
        Train OneVsRest LinearSVC using pandas/sklearn with step-by-step metric charts based on max_iteration_steps.
        """
        logger.info("=== START train_loitering_transshipment_svm_pandas_sklearn_model ===")
        try:
            cols = ["behavior_type_label"] + FEATURE_COLUMNS
            df = TrainModelService.read_aggregated_table_pandas(selected_cols=cols)
            original_labels = ["LOITERING", "TRANSSHIPMENT"]
            df_filtered = df[df["behavior_type_label"].isin(original_labels)].copy()
            counts = {lbl: int(df_filtered[df_filtered["behavior_type_label"]==lbl].shape[0]) for lbl in original_labels}
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

            sampled = TrainModelService.sample_random_balanced_pandas(df_filtered, original_labels, per_label_n, label_col="behavior_type_label", random_state=int(random_state))
            if sampled.shape[0] == 0:
                logger.warning("No rows sampled")
                return {"error": "No rows sampled", "counts": counts}

            train_df, test_df = TrainModelService.stratified_train_test_split_pandas(sampled, label_col="behavior_type_label", test_size=float(test_size), random_state=int(random_state))
            TrainModelService.log_class_distribution_pandas(train_df, test_df, label_col="behavior_type_label")

            # label encoding
            le = LabelEncoder()
            y_train = le.fit_transform(train_df["behavior_type_label"].astype(str))
            y_test = le.transform(test_df["behavior_type_label"].astype(str))
            label_order = list(le.classes_)

            # clean + impute
            train_imp, test_imp, used_features, imputer = TrainModelService.clean_and_impute_pandas(train_df, test_df, FEATURE_COLUMNS, impute_strategy=impute_strategy)
            if not used_features:
                return {"error": "No usable features after cleaning (all dropped or empty)."}

            mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
            TrainModelService.log_feature_stats_histograms_pandas(train_imp, test_imp, used_features, bins=histogram_bins, histogram_sample_size=histogram_sample_size)

            # scale
            scaler = SklearnStandardScaler()
            X_train = scaler.fit_transform(train_imp[used_features].values)
            X_test = scaler.transform(test_imp[used_features].values)

            # prepare iteration checkpoints (up to 10 points)
            iter_points = TrainModelService._make_iteration_points_int(int(max_iteration_steps), max_points=20)

            # cross-validation folds (StratifiedKFold)
            skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=int(random_state))

            # Container to accumulate CV metrics per iteration
            iter_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

            mlflow.set_experiment(experiment_name or f"SVM_Loitering_vs_Transshipment_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}")
            ok_to_start_non_nested = TrainModelService._ensure_no_active_mlflow_run()
            nested_flag = not ok_to_start_non_nested

            with mlflow.start_run(run_name=f"SVM_Loitering_vs_Transshipment_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}", nested=nested_flag):
                mlflow.log_param("model_type", "SVM_OneVsRest_LinearSVC_sklearn")
                mlflow.log_param("labels_original", ",".join(original_labels))
                mlflow.log_param("label_index_order", ",".join(label_order))
                mlflow.log_param("per_label_n", per_label_n)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("max_iteration_steps", max_iteration_steps)
                mlflow.log_param("impute_strategy", impute_strategy)
                mlflow.log_param("used_features_count", len(used_features))

                # For each iteration checkpoint, run CV and record average metrics across folds
                for it in iter_points:
                    fold_acc = []
                    fold_prec = []
                    fold_rec = []
                    fold_f1 = []
                    fold_idx = 0
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        fold_idx += 1
                        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                        # instantiate classifier with current max_iter
                        clf = OneVsRestClassifier(SklearnLinearSVC(max_iter=int(it), C=float(C), dual=False))
                        clf.fit(X_tr_fold, y_tr_fold)
                        y_val_pred = clf.predict(X_val_fold)

                        # decode numeric indices back to original string labels before computing metrics
                        y_val_fold_names = le.inverse_transform(y_val_fold)
                        y_val_pred_names = le.inverse_transform(y_val_pred)

                        report_cv = TrainModelService.compute_metrics_from_predictions(y_val_fold_names, y_val_pred_names, label_order)
                        # record macro metrics (use macro avg)
                        fold_acc.append(report_cv.get("accuracy", 0.0))
                        fold_prec.append(report_cv.get("macro_avg", {}).get("precision", 0.0))
                        fold_rec.append(report_cv.get("macro_avg", {}).get("recall", 0.0))
                        fold_f1.append(report_cv.get("macro_avg", {}).get("f1-score", 0.0))

                        # also log per-fold CV metrics as steps (optional)
                        TrainModelService.log_metrics_from_report(report_cv, prefix=f"cv_iter_{it}_fold", step=fold_idx)

                    # average across folds for this iteration
                    iter_metrics["accuracy"].append(float(np.mean(fold_acc)) if fold_acc else 0.0)
                    iter_metrics["precision"].append(float(np.mean(fold_prec)) if fold_prec else 0.0)
                    iter_metrics["recall"].append(float(np.mean(fold_rec)) if fold_rec else 0.0)
                    iter_metrics["f1"].append(float(np.mean(fold_f1)) if fold_f1 else 0.0)

                    # log aggregated CV metrics for this checkpoint as MLflow metrics (tagged by iteration)
                    mlflow.log_metric("cv_accuracy", iter_metrics["accuracy"][-1], step=int(it))
                    mlflow.log_metric("cv_precision", iter_metrics["precision"][-1], step=int(it))
                    mlflow.log_metric("cv_recall", iter_metrics["recall"][-1], step=int(it))
                    mlflow.log_metric("cv_f1", iter_metrics["f1"][-1], step=int(it))

                # Plot CV metric curves and log to MLflow
                tmpdir = tempfile.mkdtemp()
                plot_path = os.path.join(tmpdir, "svm_cv_metrics_over_iterations.png")
                TrainModelService._plot_metrics_over_iterations(iter_points, iter_metrics, title="SVM CV metrics over max_iter", outpath=plot_path)
                try:
                    mlflow.log_artifact(plot_path, artifact_path="cv_metrics")
                except Exception:
                    logger.debug("Could not log CV metric plot to MLflow (maybe no active run permitted).")

                # Train final model on full train with full max_iteration_steps
                final_clf = OneVsRestClassifier(SklearnLinearSVC(max_iter=int(max_iteration_steps), C=float(C), dual=False))
                final_clf.fit(X_train, y_train)

                # Evaluate train
                y_train_pred = final_clf.predict(X_train)
                report_train = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_train), le.inverse_transform(y_train_pred), label_order)
                TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                # plot & log confusion matrix for train
                try:
                    cm_paths_train = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_train, artifact_dir=os.path.join("artifacts", "confusion_matrix", "svm", "train"))
                    logger.info("Confusion matrix (train) saved: %s", cm_paths_train)
                except Exception:
                    logger.exception("Failed plotting/logging train confusion matrix.")
                
                # Evaluate test
                y_test_pred = final_clf.predict(X_test)
                report_test = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_test), le.inverse_transform(y_test_pred), label_order)
                TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                # plot & log confusion matrix for test
                try:
                    cm_paths_test = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_test, artifact_dir=os.path.join("artifacts", "confusion_matrix", "svm", "test"))
                    logger.info("Confusion matrix (test) saved: %s", cm_paths_test)
                except Exception:
                    logger.exception("Failed plotting/logging test confusion matrix.")

                # plot test/train single-point bars as a small chart too (optional)
                tiny_plot = os.path.join(tmpdir, "svm_train_test_metrics.png")
                combined = {
                    "train_accuracy": report_train.get("accuracy", 0.0),
                    "test_accuracy": report_test.get("accuracy", 0.0),
                    "train_f1": report_train.get("macro_avg", {}).get("f1-score", 0.0),
                    "test_f1": report_test.get("macro_avg", {}).get("f1-score", 0.0),
                }
                plt.figure(figsize=(6, 4))
                names = list(combined.keys())
                vals = [combined[n] for n in names]
                plt.bar(range(len(names)), vals)
                plt.xticks(range(len(names)), names, rotation=45, ha="right")
                plt.ylabel("Score")
                plt.title("Final train/test metrics")
                plt.tight_layout()
                plt.savefig(tiny_plot)
                plt.close()
                try:
                    mlflow.log_artifact(tiny_plot, artifact_path="final_metrics")
                except Exception:
                    logger.debug("Could not log final metrics plot.")

                # save artifacts
                TrainModelService._save_and_log_artifact(imputer, "preprocessing", "imputer")
                TrainModelService._save_and_log_artifact(scaler, "preprocessing", "scaler")
                mlflow.sklearn.log_model(final_clf, artifact_path="model")
                mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")

                run_id = mlflow.active_run().info.run_id
                if register_model_name:
                    TrainModelService.mlflow_register_model(run_id, "model", register_model_name)

            acc = float(report_test.get("accuracy", 0.0))
            f1 = float(report_test.get("macro_avg", {}).get("f1-score", 0.0))
            prec = float(report_test.get("macro_avg", {}).get("precision", 0.0))
            rec = float(report_test.get("macro_avg", {}).get("recall", 0.0))
            logger.info("=== END train_loitering_transshipment_svm_pandas: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id}

        except Exception as e:
            logger.exception("Error in train_loitering_transshipment_svm_pandas: %s", e)
            TrainModelService._end_mlflow_if_active()
            return {"error": str(e), "trace": traceback.format_exc()}

    @staticmethod
    def train_loitering_transshipment_rf_pandas_sklearn_model(per_label_n=None,
                                        test_size=0.20,
                                        random_state=42,
                                        n_estimators: int = 200,
                                        max_depth: int = None,
                                        experiment_name: str = None,
                                        register_model_name: str = None,
                                        number_of_folds: int = 5,
                                        impute_strategy: str = "median",
                                        histogram_bins: int = 30,
                                        histogram_sample_size: int = 20000):
        """
        Train RandomForest to discriminate LOITERING vs TRANSSHIPMENT using pandas/sklearn.
        Tracks metrics as n_estimators increases and logs line charts.
        """
        logger.info("=== START train_loitering_transshipment_rf_pandas_sklearn_model ===")
        try:
            labels = ["LOITERING", "TRANSSHIPMENT"]
            cols = ["behavior_type_label"] + FEATURE_COLUMNS
            df = TrainModelService.read_aggregated_table_pandas(selected_cols=cols)
            df_filtered = df[df["behavior_type_label"].isin(labels)].copy()

            counts = {lbl: int(df_filtered[df_filtered["behavior_type_label"] == lbl].shape[0]) for lbl in labels}
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

            sampled = TrainModelService.sample_random_balanced_pandas(df_filtered, labels, per_label_n, label_col="behavior_type_label", random_state=int(random_state))
            if sampled.shape[0] == 0:
                return {"error": "No rows sampled", "counts": counts}

            train_df, test_df = TrainModelService.stratified_train_test_split_pandas(sampled, label_col="behavior_type_label", test_size=float(test_size), random_state=int(random_state))

            TrainModelService.log_class_distribution_pandas(train_df, test_df, label_col="behavior_type_label")

            # encode labels
            le = LabelEncoder()
            y_train = le.fit_transform(train_df["behavior_type_label"].astype(str))
            y_test = le.transform(test_df["behavior_type_label"].astype(str))
            label_order = list(le.classes_)

            # clean + impute
            train_imp, test_imp, used_features, imputer = TrainModelService.clean_and_impute_pandas(train_df, test_df, FEATURE_COLUMNS, impute_strategy=impute_strategy)
            if not used_features:
                return {"error": "No usable features after cleaning (all dropped or empty)."}

            mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
            TrainModelService.log_feature_stats_histograms_pandas(train_imp, test_imp, used_features, bins=histogram_bins, histogram_sample_size=histogram_sample_size)

            # scale
            scaler = SklearnStandardScaler()
            X_train = scaler.fit_transform(train_imp[used_features].values)
            X_test = scaler.transform(test_imp[used_features].values)

            # iteration checkpoints for number of trees
            tree_points = TrainModelService._make_iteration_points_int(int(n_estimators), max_points=20)

            skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=int(random_state))

            iter_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

            mlflow.set_experiment(experiment_name or f"RF_Loitering_vs_Transshipment_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}")
            ok_to_start_non_nested = TrainModelService._ensure_no_active_mlflow_run()
            nested_flag = not ok_to_start_non_nested

            with mlflow.start_run(run_name=f"RF_Loitering_vs_Transshipment_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}", nested=nested_flag):
                mlflow.log_param("model_type", "RandomForestClassifier")
                mlflow.log_param("labels_original", ",".join(labels))
                mlflow.log_param("label_index_order", ",".join(label_order))
                mlflow.log_param("per_label_n", per_label_n)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("impute_strategy", impute_strategy)
                mlflow.log_param("used_features_count", len(used_features))

                for n_trees in tree_points:
                    fold_acc = []
                    fold_prec = []
                    fold_rec = []
                    fold_f1 = []
                    fold_idx = 0
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        fold_idx += 1
                        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                        clf = SklearnRandomForestClassifier(n_estimators=int(n_trees), max_depth=(None if max_depth is None else int(max_depth)), random_state=int(random_state), n_jobs=-1)
                        clf.fit(X_tr_fold, y_tr_fold)
                        y_val_pred = clf.predict(X_val_fold)

                        # decode before metrics
                        y_val_fold_names = le.inverse_transform(y_val_fold)
                        y_val_pred_names = le.inverse_transform(y_val_pred)
                        report_cv = TrainModelService.compute_metrics_from_predictions(y_val_fold_names, y_val_pred_names, label_order)

                        fold_acc.append(report_cv.get("accuracy", 0.0))
                        fold_prec.append(report_cv.get("macro_avg", {}).get("precision", 0.0))
                        fold_rec.append(report_cv.get("macro_avg", {}).get("recall", 0.0))
                        fold_f1.append(report_cv.get("macro_avg", {}).get("f1-score", 0.0))

                        TrainModelService.log_metrics_from_report(report_cv, prefix=f"cv_trees_{n_trees}_fold", step=fold_idx)

                    iter_metrics["accuracy"].append(float(np.mean(fold_acc)) if fold_acc else 0.0)
                    iter_metrics["precision"].append(float(np.mean(fold_prec)) if fold_prec else 0.0)
                    iter_metrics["recall"].append(float(np.mean(fold_rec)) if fold_rec else 0.0)
                    iter_metrics["f1"].append(float(np.mean(fold_f1)) if fold_f1 else 0.0)

                    mlflow.log_metric("cv_accuracy", iter_metrics["accuracy"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_precision", iter_metrics["precision"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_recall", iter_metrics["recall"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_f1", iter_metrics["f1"][-1], step=int(n_trees))

                # plot iteration curves
                tmpdir = tempfile.mkdtemp()
                plot_path = os.path.join(tmpdir, "rf_cv_metrics_over_trees.png")
                TrainModelService._plot_metrics_over_iterations(tree_points, iter_metrics, title="RF CV metrics over n_estimators", outpath=plot_path)
                try:
                    mlflow.log_artifact(plot_path, artifact_path="cv_metrics")
                except Exception:
                    logger.debug("Could not log RF CV metric plot to MLflow.")

                # final model
                final_clf = SklearnRandomForestClassifier(n_estimators=int(n_estimators), max_depth=(None if max_depth is None else int(max_depth)), random_state=int(random_state), n_jobs=-1)
                final_clf.fit(X_train, y_train)

                y_train_pred = final_clf.predict(X_train)
                report_train = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_train), le.inverse_transform(y_train_pred), label_order)
                TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                try:
                    cm_paths_train = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_train, artifact_dir=os.path.join("artifacts", "confusion_matrix", "rf_loitering_stopping", "train"))
                    logger.info("Confusion matrix (train) saved: %s", cm_paths_train)
                except Exception:
                    logger.exception("Failed plotting/logging train confusion matrix.")

                y_test_pred = final_clf.predict(X_test)
                report_test = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_test), le.inverse_transform(y_test_pred), label_order)
                TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                try:
                    cm_paths_test = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_test, artifact_dir=os.path.join("artifacts", "confusion_matrix", "rf_loitering_stopping", "test"))
                    logger.info("Confusion matrix (test) saved: %s", cm_paths_test)
                except Exception:
                    logger.exception("Failed plotting/logging test confusion matrix.")

                TrainModelService._save_and_log_artifact(imputer, "preprocessing", "imputer")
                TrainModelService._save_and_log_artifact(scaler, "preprocessing", "scaler")
                mlflow.sklearn.log_model(final_clf, artifact_path="model")
                mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
                run_id = mlflow.active_run().info.run_id

                if register_model_name:
                    TrainModelService.mlflow_register_model(run_id, "model", register_model_name)

            acc = float(report_test.get("accuracy", 0.0))
            f1 = float(report_test.get("macro_avg", {}).get("f1-score", 0.0))
            prec = float(report_test.get("macro_avg", {}).get("precision", 0.0))
            rec = float(report_test.get("macro_avg", {}).get("recall", 0.0))
            logger.info("=== END train_loitering_stopping_pandas: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id}

        except Exception as e:
            logger.exception("Error in train_loitering_stopping_pandas: %s", e)
            TrainModelService._end_mlflow_if_active()
            return {"error": str(e), "trace": traceback.format_exc()}

    @staticmethod
    def train_loitering_stopping_rf_pandas_sklearn_model(per_label_n=None,
                                        test_size=0.20,
                                        random_state=42,
                                        n_estimators: int = 200,
                                        max_depth: int = None,
                                        experiment_name: str = None,
                                        register_model_name: str = None,
                                        number_of_folds: int = 5,
                                        impute_strategy: str = "median",
                                        histogram_bins: int = 30,
                                        histogram_sample_size: int = 20000):
        """
        Train RandomForest to discriminate LOITERING vs STOPPING using pandas/sklearn.
        Tracks metrics as n_estimators increases and logs line charts.
        """
        logger.info("=== START train_loitering_stopping_rf_pandas_sklearn_model ===")
        try:
            labels = ["LOITERING", "STOPPING"]
            cols = ["behavior_type_label"] + FEATURE_COLUMNS
            df = TrainModelService.read_aggregated_table_pandas(selected_cols=cols)
            df_filtered = df[df["behavior_type_label"].isin(labels)].copy()

            counts = {lbl: int(df_filtered[df_filtered["behavior_type_label"] == lbl].shape[0]) for lbl in labels}
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

            sampled = TrainModelService.sample_random_balanced_pandas(df_filtered, labels, per_label_n, label_col="behavior_type_label", random_state=int(random_state))
            if sampled.shape[0] == 0:
                return {"error": "No rows sampled", "counts": counts}

            train_df, test_df = TrainModelService.stratified_train_test_split_pandas(sampled, label_col="behavior_type_label", test_size=float(test_size), random_state=int(random_state))

            TrainModelService.log_class_distribution_pandas(train_df, test_df, label_col="behavior_type_label")

            # encode labels
            le = LabelEncoder()
            y_train = le.fit_transform(train_df["behavior_type_label"].astype(str))
            y_test = le.transform(test_df["behavior_type_label"].astype(str))
            label_order = list(le.classes_)

            # clean + impute
            train_imp, test_imp, used_features, imputer = TrainModelService.clean_and_impute_pandas(train_df, test_df, FEATURE_COLUMNS, impute_strategy=impute_strategy)
            if not used_features:
                return {"error": "No usable features after cleaning (all dropped or empty)."}

            mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
            TrainModelService.log_feature_stats_histograms_pandas(train_imp, test_imp, used_features, bins=histogram_bins, histogram_sample_size=histogram_sample_size)

            # scale
            scaler = SklearnStandardScaler()
            X_train = scaler.fit_transform(train_imp[used_features].values)
            X_test = scaler.transform(test_imp[used_features].values)

            # iteration checkpoints for number of trees
            tree_points = TrainModelService._make_iteration_points_int(int(n_estimators), max_points=20)

            skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=int(random_state))

            iter_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

            mlflow.set_experiment(experiment_name or f"RF_Loitering_vs_Stopping_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}")
            ok_to_start_non_nested = TrainModelService._ensure_no_active_mlflow_run()
            nested_flag = not ok_to_start_non_nested

            with mlflow.start_run(run_name=f"RF_Loitering_vs_Stopping_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}", nested=nested_flag):
                mlflow.log_param("model_type", "RandomForestClassifier")
                mlflow.log_param("labels_original", ",".join(labels))
                mlflow.log_param("label_index_order", ",".join(label_order))
                mlflow.log_param("per_label_n", per_label_n)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("impute_strategy", impute_strategy)
                mlflow.log_param("used_features_count", len(used_features))

                for n_trees in tree_points:
                    fold_acc = []
                    fold_prec = []
                    fold_rec = []
                    fold_f1 = []
                    fold_idx = 0
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        fold_idx += 1
                        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                        clf = SklearnRandomForestClassifier(n_estimators=int(n_trees), max_depth=(None if max_depth is None else int(max_depth)), random_state=int(random_state), n_jobs=-1)
                        clf.fit(X_tr_fold, y_tr_fold)
                        y_val_pred = clf.predict(X_val_fold)

                        # decode before metrics
                        y_val_fold_names = le.inverse_transform(y_val_fold)
                        y_val_pred_names = le.inverse_transform(y_val_pred)
                        report_cv = TrainModelService.compute_metrics_from_predictions(y_val_fold_names, y_val_pred_names, label_order)

                        fold_acc.append(report_cv.get("accuracy", 0.0))
                        fold_prec.append(report_cv.get("macro_avg", {}).get("precision", 0.0))
                        fold_rec.append(report_cv.get("macro_avg", {}).get("recall", 0.0))
                        fold_f1.append(report_cv.get("macro_avg", {}).get("f1-score", 0.0))

                        TrainModelService.log_metrics_from_report(report_cv, prefix=f"cv_trees_{n_trees}_fold", step=fold_idx)

                    iter_metrics["accuracy"].append(float(np.mean(fold_acc)) if fold_acc else 0.0)
                    iter_metrics["precision"].append(float(np.mean(fold_prec)) if fold_prec else 0.0)
                    iter_metrics["recall"].append(float(np.mean(fold_rec)) if fold_rec else 0.0)
                    iter_metrics["f1"].append(float(np.mean(fold_f1)) if fold_f1 else 0.0)

                    mlflow.log_metric("cv_accuracy", iter_metrics["accuracy"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_precision", iter_metrics["precision"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_recall", iter_metrics["recall"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_f1", iter_metrics["f1"][-1], step=int(n_trees))

                # plot iteration curves
                tmpdir = tempfile.mkdtemp()
                plot_path = os.path.join(tmpdir, "rf_cv_metrics_over_trees.png")
                TrainModelService._plot_metrics_over_iterations(tree_points, iter_metrics, title="RF CV metrics over n_estimators", outpath=plot_path)
                try:
                    mlflow.log_artifact(plot_path, artifact_path="cv_metrics")
                except Exception:
                    logger.debug("Could not log RF CV metric plot to MLflow.")

                # final model
                final_clf = SklearnRandomForestClassifier(n_estimators=int(n_estimators), max_depth=(None if max_depth is None else int(max_depth)), random_state=int(random_state), n_jobs=-1)
                final_clf.fit(X_train, y_train)

                y_train_pred = final_clf.predict(X_train)
                report_train = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_train), le.inverse_transform(y_train_pred), label_order)
                TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                try:
                    cm_paths_train = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_train, artifact_dir=os.path.join("artifacts", "confusion_matrix", "rf_loitering_stopping", "train"))
                    logger.info("Confusion matrix (train) saved: %s", cm_paths_train)
                except Exception:
                    logger.exception("Failed plotting/logging train confusion matrix.")

                y_test_pred = final_clf.predict(X_test)
                report_test = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_test), le.inverse_transform(y_test_pred), label_order)
                TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                try:
                    cm_paths_test = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_test, artifact_dir=os.path.join("artifacts", "confusion_matrix", "rf_loitering_stopping", "test"))
                    logger.info("Confusion matrix (test) saved: %s", cm_paths_test)
                except Exception:
                    logger.exception("Failed plotting/logging test confusion matrix.")

                TrainModelService._save_and_log_artifact(imputer, "preprocessing", "imputer")
                TrainModelService._save_and_log_artifact(scaler, "preprocessing", "scaler")
                mlflow.sklearn.log_model(final_clf, artifact_path="model")
                mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
                run_id = mlflow.active_run().info.run_id

                if register_model_name:
                    TrainModelService.mlflow_register_model(run_id, "model", register_model_name)

            acc = float(report_test.get("accuracy", 0.0))
            f1 = float(report_test.get("macro_avg", {}).get("f1-score", 0.0))
            prec = float(report_test.get("macro_avg", {}).get("precision", 0.0))
            rec = float(report_test.get("macro_avg", {}).get("recall", 0.0))
            logger.info("=== END train_loitering_stopping_pandas: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id}

        except Exception as e:
            logger.exception("Error in train_loitering_stopping_pandas: %s", e)
            TrainModelService._end_mlflow_if_active()
            return {"error": str(e), "trace": traceback.format_exc()}


    @staticmethod
    def train_multiclass_behavior_type_svm_pandas_sklearn_model(per_label_n=None, test_size=0.20, random_state=42,
                                                    max_iteration_steps=1000, C=1.0,
                                                    experiment_name=None, register_model_name=None,
                                                    number_of_folds: int = 5,
                                                    impute_strategy: str = "median",
                                                    histogram_bins: int = 30,
                                                    histogram_sample_size: Optional[int] = 20000):
        """
        Train OneVsRest LinearSVC using pandas/sklearn with step-by-step metric charts based on max_iteration_steps.
        """
        logger.info("=== START train_multiclass_behavior_type_svm_pandas_sklearn_model ===")
        try:
            labels = list(ALLOWED_LABELS)
            cols = ["behavior_type_label"] + FEATURE_COLUMNS
            df = TrainModelService.read_aggregated_table_pandas(selected_cols=cols)
            df_filtered = df[df["behavior_type_label"].isin(labels)].copy()
            counts = {lbl: int(df_filtered[df_filtered["behavior_type_label"]==lbl].shape[0]) for lbl in labels}
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

            sampled = TrainModelService.sample_random_balanced_pandas(df_filtered, labels, per_label_n, label_col="behavior_type_label", random_state=int(random_state))
            if sampled.shape[0] == 0:
                logger.warning("No rows sampled")
                return {"error": "No rows sampled", "counts": counts}

            train_df, test_df = TrainModelService.stratified_train_test_split_pandas(sampled, label_col="behavior_type_label", test_size=float(test_size), random_state=int(random_state))
            TrainModelService.log_class_distribution_pandas(train_df, test_df, label_col="behavior_type_label")

            # label encoding
            le = LabelEncoder()
            y_train = le.fit_transform(train_df["behavior_type_label"].astype(str))
            y_test = le.transform(test_df["behavior_type_label"].astype(str))
            label_order = list(le.classes_)

            # clean + impute
            train_imp, test_imp, used_features, imputer = TrainModelService.clean_and_impute_pandas(train_df, test_df, FEATURE_COLUMNS, impute_strategy=impute_strategy)
            if not used_features:
                return {"error": "No usable features after cleaning (all dropped or empty)."}

            mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
            TrainModelService.log_feature_stats_histograms_pandas(train_imp, test_imp, used_features, bins=histogram_bins, histogram_sample_size=histogram_sample_size)

            # scale
            scaler = SklearnStandardScaler()
            X_train = scaler.fit_transform(train_imp[used_features].values)
            X_test = scaler.transform(test_imp[used_features].values)

            # prepare iteration checkpoints (up to 10 points)
            iter_points = TrainModelService._make_iteration_points_int(int(max_iteration_steps), max_points=20)

            # cross-validation folds (StratifiedKFold)
            skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=int(random_state))

            # Container to accumulate CV metrics per iteration
            iter_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

            mlflow.set_experiment(experiment_name or f"SVM_Multiclass_Behavior_Type_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}")
            ok_to_start_non_nested = TrainModelService._ensure_no_active_mlflow_run()
            nested_flag = not ok_to_start_non_nested

            with mlflow.start_run(run_name=f"SVM_Multiclass_Behavior_Type_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}", nested=nested_flag):
                mlflow.log_param("model_type", "SVM_OneVsRest_LinearSVC_sklearn")
                mlflow.log_param("labels_original", ",".join(labels))
                mlflow.log_param("label_index_order", ",".join(label_order))
                mlflow.log_param("per_label_n", per_label_n)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("max_iteration_steps", max_iteration_steps)
                mlflow.log_param("impute_strategy", impute_strategy)
                mlflow.log_param("used_features_count", len(used_features))

                # For each iteration checkpoint, run CV and record average metrics across folds
                for it in iter_points:
                    fold_acc = []
                    fold_prec = []
                    fold_rec = []
                    fold_f1 = []
                    fold_idx = 0
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        fold_idx += 1
                        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                        # instantiate classifier with current max_iter
                        clf = OneVsRestClassifier(SklearnLinearSVC(max_iter=int(it), C=float(C), dual=False))
                        clf.fit(X_tr_fold, y_tr_fold)
                        y_val_pred = clf.predict(X_val_fold)

                        # decode numeric indices back to original string labels before computing metrics
                        y_val_fold_names = le.inverse_transform(y_val_fold)
                        y_val_pred_names = le.inverse_transform(y_val_pred)

                        report_cv = TrainModelService.compute_metrics_from_predictions(y_val_fold_names, y_val_pred_names, label_order)
                        # record macro metrics (use macro avg)
                        fold_acc.append(report_cv.get("accuracy", 0.0))
                        fold_prec.append(report_cv.get("macro_avg", {}).get("precision", 0.0))
                        fold_rec.append(report_cv.get("macro_avg", {}).get("recall", 0.0))
                        fold_f1.append(report_cv.get("macro_avg", {}).get("f1-score", 0.0))

                        # also log per-fold CV metrics as steps (optional)
                        TrainModelService.log_metrics_from_report(report_cv, prefix=f"cv_iter_{it}_fold", step=fold_idx)

                    # average across folds for this iteration
                    iter_metrics["accuracy"].append(float(np.mean(fold_acc)) if fold_acc else 0.0)
                    iter_metrics["precision"].append(float(np.mean(fold_prec)) if fold_prec else 0.0)
                    iter_metrics["recall"].append(float(np.mean(fold_rec)) if fold_rec else 0.0)
                    iter_metrics["f1"].append(float(np.mean(fold_f1)) if fold_f1 else 0.0)

                    # log aggregated CV metrics for this checkpoint as MLflow metrics (tagged by iteration)
                    mlflow.log_metric("cv_accuracy", iter_metrics["accuracy"][-1], step=int(it))
                    mlflow.log_metric("cv_precision", iter_metrics["precision"][-1], step=int(it))
                    mlflow.log_metric("cv_recall", iter_metrics["recall"][-1], step=int(it))
                    mlflow.log_metric("cv_f1", iter_metrics["f1"][-1], step=int(it))

                # Plot CV metric curves and log to MLflow
                tmpdir = tempfile.mkdtemp()
                plot_path = os.path.join(tmpdir, "svm_cv_metrics_over_iterations.png")
                TrainModelService._plot_metrics_over_iterations(iter_points, iter_metrics, title="SVM CV metrics over max_iter", outpath=plot_path)
                try:
                    mlflow.log_artifact(plot_path, artifact_path="cv_metrics")
                except Exception:
                    logger.debug("Could not log CV metric plot to MLflow (maybe no active run permitted).")

                # Train final model on full train with full max_iteration_steps
                final_clf = OneVsRestClassifier(SklearnLinearSVC(max_iter=int(max_iteration_steps), C=float(C), dual=False))
                final_clf.fit(X_train, y_train)

                # Evaluate train
                y_train_pred = final_clf.predict(X_train)
                report_train = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_train), le.inverse_transform(y_train_pred), label_order)
                TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                # plot & log confusion matrix for train
                try:
                    cm_paths_train = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_train, artifact_dir=os.path.join("artifacts", "confusion_matrix", "svm", "train"))
                    logger.info("Confusion matrix (train) saved: %s", cm_paths_train)
                except Exception:
                    logger.exception("Failed plotting/logging train confusion matrix.")
                
                # Evaluate test
                y_test_pred = final_clf.predict(X_test)
                report_test = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_test), le.inverse_transform(y_test_pred), label_order)
                TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                # plot & log confusion matrix for test
                try:
                    cm_paths_test = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_test, artifact_dir=os.path.join("artifacts", "confusion_matrix", "svm", "test"))
                    logger.info("Confusion matrix (test) saved: %s", cm_paths_test)
                except Exception:
                    logger.exception("Failed plotting/logging test confusion matrix.")

                # plot test/train single-point bars as a small chart too (optional)
                tiny_plot = os.path.join(tmpdir, "svm_train_test_metrics.png")
                combined = {
                    "train_accuracy": report_train.get("accuracy", 0.0),
                    "test_accuracy": report_test.get("accuracy", 0.0),
                    "train_f1": report_train.get("macro_avg", {}).get("f1-score", 0.0),
                    "test_f1": report_test.get("macro_avg", {}).get("f1-score", 0.0),
                }
                plt.figure(figsize=(6, 4))
                names = list(combined.keys())
                vals = [combined[n] for n in names]
                plt.bar(range(len(names)), vals)
                plt.xticks(range(len(names)), names, rotation=45, ha="right")
                plt.ylabel("Score")
                plt.title("Final train/test metrics")
                plt.tight_layout()
                plt.savefig(tiny_plot)
                plt.close()
                try:
                    mlflow.log_artifact(tiny_plot, artifact_path="final_metrics")
                except Exception:
                    logger.debug("Could not log final metrics plot.")

                # save artifacts
                TrainModelService._save_and_log_artifact(imputer, "preprocessing", "imputer")
                TrainModelService._save_and_log_artifact(scaler, "preprocessing", "scaler")
                mlflow.sklearn.log_model(final_clf, artifact_path="model")
                mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")

                run_id = mlflow.active_run().info.run_id
                if register_model_name:
                    TrainModelService.mlflow_register_model(run_id, "model", register_model_name)

            acc = float(report_test.get("accuracy", 0.0))
            f1 = float(report_test.get("macro_avg", {}).get("f1-score", 0.0))
            prec = float(report_test.get("macro_avg", {}).get("precision", 0.0))
            rec = float(report_test.get("macro_avg", {}).get("recall", 0.0))
            logger.info("=== END train_multiclass_behavior_type_svm_pandas: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id}

        except Exception as e:
            logger.exception("Error in train_multiclass_behavior_type_svm_pandas: %s", e)
            TrainModelService._end_mlflow_if_active()
            return {"error": str(e), "trace": traceback.format_exc()}
    
    @staticmethod
    def train_multiclass_behavior_type_rf_pandas_sklearn_model(per_label_n=None,
                                test_size=0.20,
                                random_state=42,
                                n_estimators: int = 200,
                                max_depth: int = None,
                                experiment_name: str = None,
                                register_model_name: str = None,
                                number_of_folds: int = 5,
                                impute_strategy: str = "median",
                                histogram_bins: int = 30,
                                histogram_sample_size: int = 20000):
        """
        Train RandomForest multiclass to classify among ALLOWED_LABELS using pandas/sklearn.
        Tracks metrics as n_estimators increases and logs line charts.
        """
        logger.info("=== START train_multiclass_behavior_type_rf_pandas_sklearn_model ===")
        try:
            labels = list(ALLOWED_LABELS)
            cols = ["behavior_type_label"] + FEATURE_COLUMNS
            df = TrainModelService.read_aggregated_table_pandas(selected_cols=cols)
            df_filtered = df[df["behavior_type_label"].isin(labels)].copy()

            counts = {lbl: int(df_filtered[df_filtered["behavior_type_label"] == lbl].shape[0]) for lbl in labels}
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

            sampled = TrainModelService.sample_random_balanced_pandas(df_filtered, labels, per_label_n, label_col="behavior_type_label", random_state=int(random_state))
            if sampled.shape[0] == 0:
                return {"error": "No rows sampled", "counts": counts}

            train_df, test_df = TrainModelService.stratified_train_test_split_pandas(sampled, label_col="behavior_type_label", test_size=float(test_size), random_state=int(random_state))

            TrainModelService.log_class_distribution_pandas(train_df, test_df, label_col="behavior_type_label")

            # encode labels
            le = LabelEncoder()
            y_train = le.fit_transform(train_df["behavior_type_label"].astype(str))
            y_test = le.transform(test_df["behavior_type_label"].astype(str))
            label_order = list(le.classes_)

            # clean + impute
            train_imp, test_imp, used_features, imputer = TrainModelService.clean_and_impute_pandas(train_df, test_df, FEATURE_COLUMNS, impute_strategy=impute_strategy)
            if not used_features:
                return {"error": "No usable features after cleaning (all dropped or empty)."}

            mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
            TrainModelService.log_feature_stats_histograms_pandas(train_imp, test_imp, used_features, bins=histogram_bins, histogram_sample_size=histogram_sample_size)

            # scale
            scaler = SklearnStandardScaler()
            X_train = scaler.fit_transform(train_imp[used_features].values)
            X_test = scaler.transform(test_imp[used_features].values)

            tree_points = TrainModelService._make_iteration_points_int(int(n_estimators), max_points=20)
            skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=int(random_state))

            iter_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

            mlflow.set_experiment(experiment_name or f"RF_Transshipment_Multiclass_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}")
            ok_to_start_non_nested = TrainModelService._ensure_no_active_mlflow_run()
            nested_flag = not ok_to_start_non_nested

            with mlflow.start_run(run_name=f"RF_Transshipment_Multiclass_pandas_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}", nested=nested_flag):
                mlflow.log_param("model_type", "RandomForestClassifier_multiclass")
                mlflow.log_param("labels_original", ",".join(labels))
                mlflow.log_param("label_index_order", ",".join(label_order))
                mlflow.log_param("per_label_n", per_label_n)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("impute_strategy", impute_strategy)
                mlflow.log_param("used_features_count", len(used_features))

                for n_trees in tree_points:
                    fold_acc = []
                    fold_prec = []
                    fold_rec = []
                    fold_f1 = []
                    fold_idx = 0
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        fold_idx += 1
                        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                        clf = SklearnRandomForestClassifier(n_estimators=int(n_trees), max_depth=(None if max_depth is None else int(max_depth)), random_state=int(random_state), n_jobs=-1)
                        clf.fit(X_tr_fold, y_tr_fold)
                        y_val_pred = clf.predict(X_val_fold)

                        y_val_fold_names = le.inverse_transform(y_val_fold)
                        y_val_pred_names = le.inverse_transform(y_val_pred)
                        report_cv = TrainModelService.compute_metrics_from_predictions(y_val_fold_names, y_val_pred_names, label_order)

                        fold_acc.append(report_cv.get("accuracy", 0.0))
                        fold_prec.append(report_cv.get("macro_avg", {}).get("precision", 0.0))
                        fold_rec.append(report_cv.get("macro_avg", {}).get("recall", 0.0))
                        fold_f1.append(report_cv.get("macro_avg", {}).get("f1-score", 0.0))

                        TrainModelService.log_metrics_from_report(report_cv, prefix=f"cv_trees_{n_trees}_fold", step=fold_idx)

                    iter_metrics["accuracy"].append(float(np.mean(fold_acc)) if fold_acc else 0.0)
                    iter_metrics["precision"].append(float(np.mean(fold_prec)) if fold_prec else 0.0)
                    iter_metrics["recall"].append(float(np.mean(fold_rec)) if fold_rec else 0.0)
                    iter_metrics["f1"].append(float(np.mean(fold_f1)) if fold_f1 else 0.0)

                    mlflow.log_metric("cv_accuracy", iter_metrics["accuracy"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_precision", iter_metrics["precision"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_recall", iter_metrics["recall"][-1], step=int(n_trees))
                    mlflow.log_metric("cv_f1", iter_metrics["f1"][-1], step=int(n_trees))

                # plot iteration curves
                tmpdir = tempfile.mkdtemp()
                plot_path = os.path.join(tmpdir, "rf_multiclass_behavior_type_cv_metrics_over_trees.png")
                TrainModelService._plot_metrics_over_iterations(tree_points, iter_metrics, title="RF (multiclass) CV metrics over n_estimators", outpath=plot_path)
                try:
                    mlflow.log_artifact(plot_path, artifact_path="cv_metrics")
                except Exception:
                    logger.debug("Could not log RF multiclass CV metric plot to MLflow.")

                # final model fit & eval
                final_clf = SklearnRandomForestClassifier(n_estimators=int(n_estimators), max_depth=(None if max_depth is None else int(max_depth)), random_state=int(random_state), n_jobs=-1)
                final_clf.fit(X_train, y_train)

                y_train_pred = final_clf.predict(X_train)
                report_train = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_train), le.inverse_transform(y_train_pred), label_order)
                TrainModelService.log_metrics_from_report(report_train, prefix="train", step=None)

                try:
                    cm_paths_train = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_train, artifact_dir=os.path.join("artifacts", "confusion_matrix", "rf_multiclass_behavior_type", "train"))
                    logger.info("Confusion matrix (train) saved: %s", cm_paths_train)
                except Exception:
                    logger.exception("Failed plotting/logging train confusion matrix.")

                y_test_pred = final_clf.predict(X_test)
                report_test = TrainModelService.compute_metrics_from_predictions(le.inverse_transform(y_test), le.inverse_transform(y_test_pred), label_order)
                TrainModelService.log_metrics_from_report(report_test, prefix="test", step=None)

                try:
                    cm_paths_test = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report_test, artifact_dir=os.path.join("artifacts", "confusion_matrix", "rf_multiclass_behavior_type", "test"))
                    logger.info("Confusion matrix (test) saved: %s", cm_paths_test)
                except Exception:
                    logger.exception("Failed plotting/logging test confusion matrix.")

                TrainModelService._save_and_log_artifact(imputer, "preprocessing", "imputer")
                TrainModelService._save_and_log_artifact(scaler, "preprocessing", "scaler")
                mlflow.sklearn.log_model(final_clf, artifact_path="model")
                mlflow.log_dict({"used_features": used_features}, "used_features/used_features.json")
                run_id = mlflow.active_run().info.run_id

                if register_model_name:
                    TrainModelService.mlflow_register_model(run_id, "model", register_model_name)

            acc = float(report_test.get("accuracy", 0.0))
            f1 = float(report_test.get("macro_avg", {}).get("f1-score", 0.0))
            prec = float(report_test.get("macro_avg", {}).get("precision", 0.0))
            rec = float(report_test.get("macro_avg", {}).get("recall", 0.0))
            logger.info("=== END train_multiclass_behavior_type_rf_pandas: accuracy=%.4f f1=%.4f ===", acc, f1)
            return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "mlflow_run_id": run_id}

        except Exception as e:
            logger.exception("Error in train_multiclass_behavior_type_rf_pandas: %s", e)
            TrainModelService._end_mlflow_if_active()
            return {"error": str(e), "trace": traceback.format_exc()}
        
    # TENSORFLOW-BASED METHODS BELOW
    # optional plotting
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        _HAS_PLOTTING = True
    except Exception:
        _HAS_PLOTTING = False

    logger = logging.getLogger(__name__)

    # ---------------------------
    # Helper: choose preprocess_input by model name
    # ---------------------------
    @staticmethod
    def _get_preprocess_fn_for_model(model_name: str):
        """
        Return the appropriate preprocessing function for a given model name.
        Handles MobileNet, MobileNetV2, MobileNetV3, EfficientNet, Xception, and VGG variants.
        """
        name = model_name.lower()
        
        # MobileNet variants
        if "mobilenetv3small" in name or "mobilenet_v3_small" in name or "mobilenetv3small_model" in name:
            return tf.keras.applications.mobilenet_v3.preprocess_input
        if "mobilenetv3large" in name or "mobilenet_v3_large" in name or "mobilenetv3large_model" in name:
            return tf.keras.applications.mobilenet_v3.preprocess_input
        if "mobilenetv2" in name or "mobilenet_v2" in name or "mobilenetv2_model" in name:
            return tf.keras.applications.mobilenet_v2.preprocess_input
        if "mobilenet" in name and "v2" not in name and "v3" not in name:
            return tf.keras.applications.mobilenet.preprocess_input
        
        # EfficientNet variants
        if "efficientnet" in name:
            return tf.keras.applications.efficientnet.preprocess_input
        
        # Xception
        if "xception" in name:
            return tf.keras.applications.xception.preprocess_input
        
        # VGG variants
        if "vgg16" in name:
            return tf.keras.applications.vgg16.preprocess_input
        if "vgg19" in name:
            return tf.keras.applications.vgg19.preprocess_input
        
        # Fallback: scale to 0-255 range
        return lambda x: x * 255.0

    # ---------------------------
    # Helper: build tf.data dataset returning (image, label_int)
    # ---------------------------
    def _build_tf_dataset(filepaths, labels, preprocess_fn, img_size=(120, 120), batch_size=16, shuffle=True):
        AUTOTUNE = tf.data.AUTOTUNE

        filepaths = list(filepaths)
        labels = list(labels)

        def _load_and_preprocess(path, label):
            # path: tf.string scalar
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
            img = tf.image.resize(img, img_size)
            # many preprocess_input expect 0..255; our preprocess function multiplies accordingly
            img = preprocess_fn(img)
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(filepaths), seed=42)
        ds = ds.map(lambda p, l: tf.py_function(func=_load_and_preprocess, inp=[p, l], Tout=(tf.float32, tf.int32)),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda img, lbl: (tf.reshape(img, (img_size[0], img_size[1], 3)), tf.cast(lbl, tf.int32)), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    # ---------------------------
    # Helper: Train & log one Keras model, ensuring class names are used in all CSVs & MLflow logs
    # ---------------------------
    def _train_and_log_keras_model_with_classnames(
        base_model,
        model_name: str,
        X_train_paths, y_train_int,
        X_test_paths, y_test_int,
        label_encoder: LabelEncoder,
        epochs=100,
        batch_size=16,
        learning_rate=0.001,
        callbacks_list=None,
        experiment_name=None,
        register_model_name=None,
        out_dir="artifacts/image_training"
    ):
        os.makedirs(out_dir, exist_ok=True)
        num_classes = len(label_encoder.classes_)
        preprocess_fn = TrainModelService._get_preprocess_fn_for_model(model_name)

        # Build datasets
        train_ds = TrainModelService._build_tf_dataset(
            X_train_paths, y_train_int, preprocess_fn,
            img_size=(120, 120), batch_size=batch_size, shuffle=True
        )
        test_ds = TrainModelService._build_tf_dataset(
            X_test_paths, y_test_int, preprocess_fn,
            img_size=(120, 120), batch_size=batch_size, shuffle=False
        )

        # ----------------------------
        # Model architecture
        # ----------------------------
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(120, 120, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs, name=f"{model_name}_full")

        # ----------------------------
        # Callbacks
        # ----------------------------
        cb = callbacks_list[:] if callbacks_list else [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5),
        ]
        csv_log_path = os.path.join(out_dir, f"{model_name}_epoch_history.csv")
        cb.append(tf.keras.callbacks.CSVLogger(csv_log_path))

        # ----------------------------
        # MLflow setup
        # ----------------------------
        mlflow.set_experiment(experiment_name or f"image_training_{model_name}")
        with mlflow.start_run(run_name=f"{model_name}_train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):

            mlflow.log_param("model_base", model_name)
            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            # ----------------------------
            # Phase 1
            # ----------------------------
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )

            history_phase1 = model.fit(
                train_ds,
                epochs=max(3, epochs // 4),
                validation_data=test_ds,
                callbacks=cb,
                verbose=1,
            )

            # ----------------------------
            # Phase 2
            # ----------------------------
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )

            history_phase2 = model.fit(
                train_ds,
                epochs=epochs,
                validation_data=test_ds,
                callbacks=cb,
                verbose=1,
            )

            # ----------------------------
            # Merge histories
            # ----------------------------
            history = {}
            for k in history_phase1.history:
                history[k] = history_phase1.history[k] + history_phase2.history.get(k, [])

            # ----------------------------
            # Evaluation
            # ----------------------------
            eval_res = model.evaluate(test_ds, verbose=1)
            mlflow.log_metric("test_loss", float(eval_res[0]))
            if len(eval_res) >= 2:
                mlflow.log_metric("test_sparse_categorical_accuracy", float(eval_res[1]))

            # ----------------------------
            # Predictions
            # ----------------------------
            y_true, y_pred = [], []
            for imgs, labels in test_ds:
                preds = model.predict(imgs)
                y_pred.extend(np.argmax(preds, axis=1).tolist())
                y_true.extend(labels.numpy().tolist())

            y_true = np.array(y_true, dtype=int)
            y_pred = np.array(y_pred, dtype=int)

            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            cm_csv = os.path.join(out_dir, f"{model_name}_confusion_matrix.csv")
            with open(cm_csv, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([""] + list(label_encoder.classes_))
                for i, row in enumerate(cm):
                    writer.writerow([label_encoder.classes_[i]] + row.tolist())

            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=list(range(num_classes)), zero_division=0
            )

            per_class_csv = os.path.join(out_dir, f"{model_name}_per_class_metrics.csv")
            with open(per_class_csv, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["class_name", "precision", "recall", "f1", "support"])
                for i, name in enumerate(label_encoder.classes_):
                    writer.writerow([name, float(precision[i]), float(recall[i]), float(f1[i]), int(support[i])])

            for i, name in enumerate(label_encoder.classes_):
                mlflow.log_metric(f"precision_{name}", float(precision[i]))
                mlflow.log_metric(f"recall_{name}", float(recall[i]))
                mlflow.log_metric(f"f1_{name}", float(f1[i]))
                mlflow.log_metric(f"support_{name}", int(support[i]))

            # ----------------------------
            # Confusion matrix plot (same folder as CSV)
            # ----------------------------
            report = {
                "confusion_matrix": cm.tolist(),
                "labels": list(label_encoder.classes_),
                "per_class": {
                    label_encoder.classes_[i]: {
                        "precision": float(precision[i]),
                        "recall": float(recall[i]),
                        "f1": float(f1[i]),
                        "support": int(support[i]),
                    }
                    for i in range(num_classes)
                },  
            }

            TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(
                report=report,
                artifact_dir=out_dir
            )

            classes_meta = os.path.join(out_dir, f"{model_name}_class_names.txt")
            with open(classes_meta, "w") as fh:
                fh.write("\n".join(label_encoder.classes_))

            metrics_csv = os.path.join(out_dir, f"{model_name}_metrics_per_epoch.csv")
            with open(metrics_csv, "w", newline="") as fh:
                writer = csv.writer(fh)
                header = ["epoch"] + list(history.keys())
                writer.writerow(header)
                n = len(next(iter(history.values())))
                for e in range(n):
                    writer.writerow([e + 1] + [history[k][e] for k in history])

            try:
                mlflow.log_artifact(cm_csv, "confusion_matrix")
                mlflow.log_artifact(per_class_csv, "per_class_metrics")
                mlflow.log_artifact(metrics_csv, "metrics")
                mlflow.log_artifact(classes_meta, "metadata")
                mlflow.keras.log_model(model, artifact_path="model")
            except Exception:
                logger.exception("Failed to log artifacts or model for %s", model_name)

            if register_model_name:
                try:
                    run_id = mlflow.active_run().info.run_id
                except Exception:
                    logger.exception("Failed registering model for %s", model_name)

        return {
            "history": history,
            "confusion_matrix": cm,
            "per_class_metrics": {
                "class_names": list(label_encoder.classes_),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1": f1.tolist(),
                "support": support.tolist(),
            },
            "eval": eval_res,
        }

    # ---------------------------
    # Top-level: LOITERING vs TRANSSHIPMENT training
    # ---------------------------
    def train_loitering_transshipment_image_models(
        dataset_dir: str = "/app/processed_output/image_trajectory_datasets_all_behavior_types/image_resolution_224x224",
        models_dict: dict = None,
        per_label_n: int = None,
        test_size: float = 0.2,
        random_state: int = 42,
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        callbacks_list: list = None,
        experiment_name: str = None,
        register_model_name: str = None
    ):
        """
        Train models for LOITERING vs TRANSSHIPMENT; ensures all CSVs and MLflow logs include original class names.
        """
        base = pathlib.Path(dataset_dir)
        labels = ["LOITERING", "TRANSSHIPMENT"]
        rows = []
        for lab in labels:
            d = base / lab
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        rows.append({"path": str(p.resolve()), "label": lab})
        df = pd.DataFrame(rows)
        if df.empty:
            return {"error": "No images found under dataset_dir for desired labels."}

        # Determine how many per label to use
        counts = df["label"].value_counts().to_dict()
        min_count = min(counts.values())
        if per_label_n is None:
            per_label_n = min_count
        else:
            per_label_n = min(per_label_n, min_count)

        sampled_parts = []
        for lab in labels:
            sub = df[df["label"] == lab]
            if sub.shape[0] == 0:
                return {"error": f"No images for label {lab}"}
            sampled_parts.append(sub.sample(n=per_label_n, random_state=int(random_state)))
        df_sampled = pd.concat(sampled_parts).reset_index(drop=True)

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
        X = df_sampled["path"].values
        y = df_sampled["label"].values
        train_idx, test_idx = next(sss.split(X, y))
        X_train_paths, X_test_paths = X[train_idx].tolist(), X[test_idx].tolist()
        y_train_labels, y_test_labels = y[train_idx].tolist(), y[test_idx].tolist()

        # Label encoder fit on original labels (ensures inverse_transform returns original names)
        le = LabelEncoder().fit(labels)
        y_train_int = le.transform(y_train_labels)
        y_test_int = le.transform(y_test_labels)

        results = {}
        for model_name, base_model in (models_dict or {}).items():
            logger.info("Training image model: %s", model_name)
            out_dir = os.path.join("artifacts", "image_models", model_name)
            res = TrainModelService._train_and_log_keras_model_with_classnames(
                base_model=base_model,
                model_name=model_name,
                X_train_paths=X_train_paths,
                y_train_int=y_train_int,
                X_test_paths=X_test_paths,
                y_test_int=y_test_int,
                label_encoder=le,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                callbacks_list=callbacks_list,
                experiment_name=experiment_name,
                register_model_name=register_model_name,
                out_dir=out_dir
            )
            results[model_name] = res
        return results

    # ---------------------------
    # Top-level: LOITERING vs STOPPING training
    # ---------------------------
    def train_loitering_stopping_image_models(
        dataset_dir: str = "/app/processed_output/image_trajectory_datasets_all_behavior_types/image_resolution_224x224",
        models_dict: dict = None,
        per_label_n: int = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ):
        # identical pipeline but labels ["LOITERING", "STOPPING"]
        base = pathlib.Path(dataset_dir)
        labels = ["LOITERING", "STOPPING"]
        rows = []
        for lab in labels:
            d = base / lab
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        rows.append({"path": str(p.resolve()), "label": lab})
        df = pd.DataFrame(rows)
        if df.empty:
            return {"error": "No images found under dataset_dir for desired labels."}

        counts = df["label"].value_counts().to_dict()
        min_count = min(counts.values())
        if per_label_n is None:
            per_label_n = min_count
        else:
            per_label_n = min(per_label_n, min_count)

        sampled_parts = []
        for lab in labels:
            sub = df[df["label"] == lab]
            if sub.shape[0] == 0:
                return {"error": f"No images for label {lab}"}
            sampled_parts.append(sub.sample(n=per_label_n, random_state=int(random_state)))
        df_sampled = pd.concat(sampled_parts).reset_index(drop=True)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
        X = df_sampled["path"].values
        y = df_sampled["label"].values
        train_idx, test_idx = next(sss.split(X, y))
        X_train_paths, X_test_paths = X[train_idx].tolist(), X[test_idx].tolist()
        y_train_labels, y_test_labels = y[train_idx].tolist(), y[test_idx].tolist()

        le = LabelEncoder().fit(labels)
        y_train_int = le.transform(y_train_labels)
        y_test_int = le.transform(y_test_labels)

        results = {}
        for model_name, base_model in (models_dict or {}).items():
            logger.info("Training image model: %s", model_name)
            out_dir = os.path.join("artifacts", "image_models", model_name)
            res = TrainModelService._train_and_log_keras_model_with_classnames(
                base_model=base_model,
                model_name=model_name,
                X_train_paths=X_train_paths,
                y_train_int=y_train_int,
                X_test_paths=X_test_paths,
                y_test_int=y_test_int,
                label_encoder=le,
                epochs=kwargs.get("epochs", 20),
                batch_size=kwargs.get("batch_size", 16),
                learning_rate=kwargs.get("learning_rate", 0.001),
                callbacks_list=kwargs.get("callbacks_list", None),
                experiment_name=kwargs.get("experiment_name", None),
                register_model_name=kwargs.get("register_model_name", None),
                out_dir=out_dir
            )
            results[model_name] = res
        return results

    # ---------------------------
    # Defensive GPU configuration helper (English logs)
    # ---------------------------
    @staticmethod
    def _configure_gpu_defensively():
        """
        Detect GPUs and configure TensorFlow memory growth defensively.
        Returns True if a GPU is available/configured, False otherwise.
        All log messages are in English.
        """
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPU devices detected. Training will run on CPU.")
            return False

        try:
            for gpu in gpus:
                # Prevent TensorFlow from pre-allocating all GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"Detected {len(gpus)} physical GPU(s) and {len(logical_gpus)} logical GPU(s).")
            return True
        except RuntimeError as e:
            # Happens if GPUs were already initialized when this code runs
            logger.warning(f"Could not set memory growth for GPU(s) (they may already be initialized): {e}")
            # We still return True because GPUs exist and TF will choose what it can
            return True
        
    @staticmethod
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: TrainModelService.make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [TrainModelService.make_json_safe(v) for v in obj]
        elif hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        else:
            return obj
    
    # ---------------------------
    # Top-level: LOITERING vs TRANSSHIPMENT training
    # ---------------------------
    def train_loitering_transshipment_image_models(
        dataset_dir: str = "/app/processed_output/image_trajectory_datasets_all_behavior_types/image_resolution_224x224",
        models_dict: dict = None,
        per_label_n: int = None,
        test_size: float = 0.2,
        random_state: int = 42,
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        callbacks_list: list = None,
        experiment_name: str = None,
        register_model_name: str = None
    ):
        """
        Train models for LOITERING vs TRANSSHIPMENT; ensures all CSVs and MLflow logs include original class names.
        """
        # defensive GPU configuration
        try:
            use_gpu = TrainModelService._configure_gpu_defensively()
        except Exception as e:
            logger.warning("GPU configuration helper failed; falling back to CPU. Error: %s", e)
            use_gpu = False

        device = "/GPU:0" if use_gpu else "/CPU:0"
        # small debug log in English immediately after GPU configuration
        try:
            logger.info("tf.config.list_physical_devices('GPU') -> %s; chosen device -> %s", tf.config.list_physical_devices('GPU'), device)
        except Exception:
            logger.info("Could not list physical devices; chosen device -> %s", device)

        base = pathlib.Path(dataset_dir)
        logger.info("Debug: pathing dataset_dir=%s", dataset_dir)
        labels = ["LOITERING", "TRANSSHIPMENT"]
        rows = []
        for lab in labels:
            d = os.path.join(base, lab)
            logger.info("Debug: processing label=%s at dir=%s", lab, d)
            if os.path.isdir(d):
                for fname in os.listdir(d):
                    p = os.path.join(d, fname)
                    ext = os.path.splitext(p)[1].lower()
                    if ext in [".png", ".jpg", ".jpeg"] and os.path.isfile(p):
                        rows.append({"path": os.path.abspath(p), "label": lab})
        df = pd.DataFrame(rows)
        if df.empty:
            return {"error": "No images found under dataset_dir for desired labels."}

        # Determine how many per label to use
        counts = df["label"].value_counts().to_dict()
        min_count = min(counts.values())
        if per_label_n is None:
            per_label_n = min_count
        else:
            per_label_n = min(per_label_n, min_count)

        sampled_parts = []
        for lab in labels:
            sub = df[df["label"] == lab]
            if sub.shape[0] == 0:
                return {"error": f"No images for label {lab}"}
            sampled_parts.append(sub.sample(n=per_label_n, random_state=int(random_state)))
        df_sampled = pd.concat(sampled_parts).reset_index(drop=True)

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
        X = df_sampled["path"].values
        y = df_sampled["label"].values
        train_idx, test_idx = next(sss.split(X, y))
        X_train_paths, X_test_paths = X[train_idx].tolist(), X[test_idx].tolist()
        y_train_labels, y_test_labels = y[train_idx].tolist(), y[test_idx].tolist()

        # Label encoder fit on original labels (ensures inverse_transform returns original names)
        le = LabelEncoder().fit(labels)
        y_train_int = le.transform(y_train_labels)
        y_test_int = le.transform(y_test_labels)

        results = {}
        for model_name, base_model in (models_dict or {}).items():
            logger.info("Training image model: %s", model_name)
            out_dir = os.path.join("artifacts", "image_models", model_name)
            # run training under preferred device context
            try:
                with tf.device(device):
                    res = TrainModelService._train_and_log_keras_model_with_classnames(
                        base_model=base_model,
                        model_name=model_name,
                        X_train_paths=X_train_paths,
                        y_train_int=y_train_int,
                        X_test_paths=X_test_paths,
                        y_test_int=y_test_int,
                        label_encoder=le,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        callbacks_list=callbacks_list,
                        experiment_name=experiment_name,
                        register_model_name=register_model_name,
                        out_dir=out_dir
                    )
            except Exception as e:
                # If explicit device placement fails for any reason, fall back to calling without device context
                logger.warning("Device context failed for model %s with device %s, falling back to default device. Error: %s", model_name, device, str(e))
                res = TrainModelService._train_and_log_keras_model_with_classnames(
                    base_model=base_model,
                    model_name=model_name,
                    X_train_paths=X_train_paths,
                    y_train_int=y_train_int,
                    X_test_paths=X_test_paths,
                    y_test_int=y_test_int,
                    label_encoder=le,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    callbacks_list=callbacks_list,
                    experiment_name=experiment_name,
                    register_model_name=register_model_name,
                    out_dir=out_dir
                )
            results[model_name] = res
        return results


    # ---------------------------
    # Top-level: LOITERING vs STOPPING training
    # ---------------------------
    def train_loitering_stopping_image_models(
        dataset_dir: str = "/app/processed_output/image_trajectory_datasets_all_behavior_types/image_resolution_224x224",
        models_dict: dict = None,
        per_label_n: int = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ):
        # defensive GPU configuration
        try:
            use_gpu = TrainModelService._configure_gpu_defensively()
        except Exception as e:
            logger.warning("GPU configuration helper failed; falling back to CPU. Error: %s", e)
            use_gpu = False

        device = "/GPU:0" if use_gpu else "/CPU:0"
        # small debug log in English immediately after GPU configuration
        try:
            logger.info("tf.config.list_physical_devices('GPU') -> %s; chosen device -> %s", tf.config.list_physical_devices('GPU'), device)
        except Exception:
            logger.info("Could not list physical devices; chosen device -> %s", device)

        # identical pipeline but labels ["LOITERING", "STOPPING"]
        base = pathlib.Path(dataset_dir)
        logger.info("Debug: pathing dataset_dir=%s", dataset_dir)
        labels = ["LOITERING", "STOPPING"]
        rows = []
        for lab in labels:
            d = os.path.join(base, lab)
            logger.info("Debug: processing label=%s at dir=%s", lab, d)
            if os.path.isdir(d):
                for fname in os.listdir(d):
                    p = os.path.join(d, fname)
                    ext = os.path.splitext(p)[1].lower()
                    if ext in [".png", ".jpg", ".jpeg"] and os.path.isfile(p):
                        rows.append({"path": os.path.abspath(p), "label": lab})
        df = pd.DataFrame(rows)
        if df.empty:
            return {"error": "No images found under dataset_dir for desired labels."}

        counts = df["label"].value_counts().to_dict()
        min_count = min(counts.values())
        if per_label_n is None:
            per_label_n = min_count
        else:
            per_label_n = min(per_label_n, min_count)

        sampled_parts = []
        for lab in labels:
            sub = df[df["label"] == lab]
            if sub.shape[0] == 0:
                return {"error": f"No images for label {lab}"}
            sampled_parts.append(sub.sample(n=per_label_n, random_state=int(random_state)))
        df_sampled = pd.concat(sampled_parts).reset_index(drop=True)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
        X = df_sampled["path"].values
        y = df_sampled["label"].values
        train_idx, test_idx = next(sss.split(X, y))
        X_train_paths, X_test_paths = X[train_idx].tolist(), X[test_idx].tolist()
        y_train_labels, y_test_labels = y[train_idx].tolist(), y[test_idx].tolist()

        le = LabelEncoder().fit(labels)
        y_train_int = le.transform(y_train_labels)
        y_test_int = le.transform(y_test_labels)

        results = {}
        for model_name, base_model in (models_dict or {}).items():
            logger.info("Training image model: %s", model_name)
            out_dir = os.path.join("artifacts", "image_models", model_name)
            try:
                with tf.device(device):
                    res = TrainModelService._train_and_log_keras_model_with_classnames(
                        base_model=base_model,
                        model_name=model_name,
                        X_train_paths=X_train_paths,
                        y_train_int=y_train_int,
                        X_test_paths=X_test_paths,
                        y_test_int=y_test_int,
                        label_encoder=le,
                        epochs=kwargs.get("epochs", 20),
                        batch_size=kwargs.get("batch_size", 16),
                        learning_rate=kwargs.get("learning_rate", 0.001),
                        callbacks_list=kwargs.get("callbacks_list", None),
                        experiment_name=kwargs.get("experiment_name", None),
                        register_model_name=kwargs.get("register_model_name", None),
                        out_dir=out_dir
                    )
            except Exception as e:
                logger.warning("Device context failed for model %s with device %s, falling back to default device. Error: %s", model_name, device, str(e))
                res = TrainModelService._train_and_log_keras_model_with_classnames(
                    base_model=base_model,
                    model_name=model_name,
                    X_train_paths=X_train_paths,
                    y_train_int=y_train_int,
                    X_test_paths=X_test_paths,
                    y_test_int=y_test_int,
                    label_encoder=le,
                    epochs=kwargs.get("epochs", 20),
                    batch_size=kwargs.get("batch_size", 16),
                    learning_rate=kwargs.get("learning_rate", 0.001),
                    callbacks_list=kwargs.get("callbacks_list", None),
                    experiment_name=kwargs.get("experiment_name", None),
                    register_model_name=kwargs.get("register_model_name", None),
                    out_dir=out_dir
                )
            results[model_name] = res
        return results


    # ---------------------------
    # Top-level: Multi-class training for all allowed labels
    # ---------------------------
    def train_all_behavior_types_image_models(
        dataset_dir: str = "/app/processed_output/image_trajectory_datasets_all_behavior_types/image_resolution_224x224",
        models_dict: dict = None,
        allowed_labels: list = None,
        per_label_n: int = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ):
        """
        Train models for all provided allowed_labels (e.g. ["LOITERING","NORMAL","STOPPING","TRANSSHIPMENT"]).
        Ensures CSVs and MLflow use original class names.
        """
        # defensive GPU configuration
        try:
            use_gpu = TrainModelService._configure_gpu_defensively()
        except Exception as e:
            logger.warning("GPU configuration helper failed; falling back to CPU. Error: %s", e)
            use_gpu = False

        device = "/GPU:0" if use_gpu else "/CPU:0"
        # small debug log in English immediately after GPU configuration
        try:
            logger.info("tf.config.list_physical_devices('GPU') -> %s; chosen device -> %s", tf.config.list_physical_devices('GPU'), device)
        except Exception:
            logger.info("Could not list physical devices; chosen device -> %s", device)

        if allowed_labels is None:
            allowed_labels = ["LOITERING", "NORMAL", "STOPPING", "TRANSSHIPMENT"]

        base = pathlib.Path(dataset_dir)
        logger.info("Debug: pathing dataset_dir=%s", dataset_dir)
        labels = allowed_labels
        rows = []
        for lab in labels:
            d = os.path.join(base, lab)
            logger.info("Debug: processing label=%s at dir=%s", lab, d)
            if os.path.isdir(d):
                for fname in os.listdir(d):
                    p = os.path.join(d, fname)
                    ext = os.path.splitext(p)[1].lower()
                    if ext in [".png", ".jpg", ".jpeg"] and os.path.isfile(p):
                        rows.append({"path": os.path.abspath(p), "label": lab})
        df = pd.DataFrame(rows)
        if df.empty:
            return {"error": "No images found under dataset_dir for desired labels."}

        counts = df["label"].value_counts().to_dict()
        min_count = min(counts.values())
        if per_label_n is None:
            per_label_n = min_count
        else:
            per_label_n = min(per_label_n, min_count)

        sampled_parts = []
        for lab in labels:
            sub = df[df["label"] == lab]
            if sub.shape[0] == 0:
                return {"error": f"No images for label {lab}"}
            sampled_parts.append(sub.sample(n=per_label_n, random_state=int(random_state)))
        df_sampled = pd.concat(sampled_parts).reset_index(drop=True)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
        X = df_sampled["path"].values
        y = df_sampled["label"].values
        train_idx, test_idx = next(sss.split(X, y))
        X_train_paths, X_test_paths = X[train_idx].tolist(), X[test_idx].tolist()
        y_train_labels, y_test_labels = y[train_idx].tolist(), y[test_idx].tolist()

        le = LabelEncoder().fit(labels)
        y_train_int = le.transform(y_train_labels)
        y_test_int = le.transform(y_test_labels)

        results = {}
        for model_name, base_model in (models_dict or {}).items():
            logger.info("Training image model: %s", model_name)
            out_dir = os.path.join("artifacts", "image_models", model_name)
            try:
                with tf.device(device):
                    res = TrainModelService._train_and_log_keras_model_with_classnames(
                        base_model=base_model,
                        model_name=model_name,
                        X_train_paths=X_train_paths,
                        y_train_int=y_train_int,
                        X_test_paths=X_test_paths,
                        y_test_int=y_test_int,
                        label_encoder=le,
                        epochs=kwargs.get("epochs", 20),
                        batch_size=kwargs.get("batch_size", 16),
                        learning_rate=kwargs.get("learning_rate", 0.001),
                        callbacks_list=kwargs.get("callbacks_list", None),
                        experiment_name=kwargs.get("experiment_name", None),
                        register_model_name=kwargs.get("register_model_name", None),
                        out_dir=out_dir
                    )
            except Exception as e:
                logger.warning("Device context failed for model %s with device %s, falling back to default device. Error: %s", model_name, device, str(e))
                res = TrainModelService._train_and_log_keras_model_with_classnames(
                    base_model=base_model,
                    model_name=model_name,
                    X_train_paths=X_train_paths,
                    y_train_int=y_train_int,
                    X_test_paths=X_test_paths,
                    y_test_int=y_test_int,
                    label_encoder=le,
                    epochs=kwargs.get("epochs", 20),
                    batch_size=kwargs.get("batch_size", 16),
                    learning_rate=kwargs.get("learning_rate", 0.001),
                    callbacks_list=kwargs.get("callbacks_list", None),
                    experiment_name=kwargs.get("experiment_name", None),
                    register_model_name=kwargs.get("register_model_name", None),
                    out_dir=out_dir
                )
            results[model_name] = res
        return results
    
    # Paste inside class TrainModelService (or replace existing function definitions).
    # Requires imports already present at top of file (tensorflow, numpy, pandas, mlflow, etc).

    @staticmethod
    def _build_tf_dataset_from_matrix_column(
        images_list,
        labels_list,
        aux_data: dict,
        preprocess_fn=None,
        img_size=(120, 120),
        batch_size=16,
        shuffle=True,
        use_aux_inputs=True,
        aux_column_order=None,
        aux_column_lengths=None,
        random_state=42
    ):
        """
        Build a tf.data.Dataset from in-memory lists:
          - images_list: list/np.array of image arrays (uint8 or float)
          - labels_list: list/np.int (int labels)
          - aux_data: dict {colname: list/np.array} where each element is a 1D list/array for that row
          - preprocess_fn: model-specific preprocess (can expect 0..1 or 0..255 input)
          - use_aux_inputs: whether to include auxiliary inputs
          - aux_column_order: explicit ordering of aux columns (list). If None, use aux_data.keys()
          - aux_column_lengths: dict {col: length} to pad/truncate aux inputs (if None, computed from aux_data)
        Returns: tf.data.Dataset that yields ((img_batch, aux1_batch, aux2_batch, ...), label_batch)
        Mirrors style/structure of TrainModelService._build_tf_dataset (file-based) but for matrix-in-memory.
        """
        AUTOTUNE = tf.data.AUTOTUNE

        images_list = list(images_list)
        labels_list = list(labels_list or [])
        if aux_column_order is None:
            aux_column_order = list(aux_data.keys()) if aux_data is not None else []

        # compute aux column lengths if not provided
        if use_aux_inputs:
            if aux_column_lengths is None:
                aux_column_lengths = {}
                for col in aux_column_order:
                    col_lists = aux_data.get(col, [])
                    maxlen = 0
                    for arr in col_lists:
                        if arr is None:
                            continue
                        try:
                            l = len(arr)
                        except Exception:
                            # fallback: treat as scalar
                            l = 1
                        if l > maxlen:
                            maxlen = l
                    aux_column_lengths[col] = int(maxlen)
            else:
                # ensure ints
                aux_column_lengths = {k: int(v) for k, v in aux_column_lengths.items()}

        # Build a generator that yields (img, aux1, aux2, ..., label)
        def gen():
            n = len(images_list)
            for i in range(n):
                img = images_list[i]
                label = int(labels_list[i])
                if use_aux_inputs:
                    row_aux = []
                    for col in aux_column_order:
                        colvals = aux_data.get(col, [])
                        v = colvals[i] if i < len(colvals) else None
                        # ensure a numpy 1D array (float32)
                        if v is None:
                            vec = np.zeros((aux_column_lengths.get(col, 0),), dtype=np.float32)
                        else:
                            arr = np.array(v, dtype=np.float32)
                            L = aux_column_lengths.get(col, 0)
                            if L <= 0:
                                vec = np.array([], dtype=np.float32)
                            else:
                                if arr.size >= L:
                                    vec = arr.flatten()[:L].astype(np.float32)
                                else:
                                    # pad with zeros
                                    pad = np.zeros((L - arr.size,), dtype=np.float32)
                                    vec = np.concatenate([arr.flatten(), pad]).astype(np.float32)
                        row_aux.append(vec)
                    # yield tuple: (img, aux1, aux2, ... , label)
                    yield tuple([np.array(img)] + row_aux + [np.int32(label)])
                else:
                    yield np.array(img), np.int32(label)

        # Build output_signature
        if use_aux_inputs:
            specs = []
            # image: variable H,W,3 uint8/float32 - we will cast later
            specs.append(tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8))
            for col in aux_column_order:
                L = aux_column_lengths.get(col, 0)
                specs.append(tf.TensorSpec(shape=(L,), dtype=tf.float32))
            specs.append(tf.TensorSpec(shape=(), dtype=tf.int32))
            ds = tf.data.Dataset.from_generator(gen, output_signature=tuple(specs))
        else:
            ds = tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                )
            )

        # Mapping: resize image, cast to float32 and normalize/apply preprocess_fn.
        def _map_with_aux(*args):
            # args: img, aux1, aux2, ..., label
            img = args[0]
            aux_inputs = args[1:-1]
            lab = args[-1]
            # ensure float image
            img_rs = tf.image.resize(img, img_size)
            img_f = tf.cast(img_rs, tf.float32)  # currently 0..255 likely
            # apply preprocess_fn if provided
            if preprocess_fn is not None:
                try:
                    img_f = preprocess_fn(img_f)
                except Exception:
                    # fallback: if preprocess fails, normalize 0..1
                    img_f = img_f / 255.0
            else:
                # ensure normalized 0..1 (requested)
                img_f = img_f / 255.0
            # aux inputs already padded to fixed length and dtype float32
            return (img_f, *aux_inputs), lab

        def _map_no_aux(img, lab):
            img_rs = tf.image.resize(img, img_size)
            img_f = tf.cast(img_rs, tf.float32)
            if preprocess_fn is not None:
                try:
                    img_f = preprocess_fn(img_f)
                except Exception:
                    img_f = img_f / 255.0
            else:
                img_f = img_f / 255.0
            return img_f, lab

        if use_aux_inputs:
            ds = ds.map(lambda *a: _map_with_aux(*a), num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda img, lab: _map_no_aux(img, lab), num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size=max(512, len(images_list)), seed=int(random_state))
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds
    
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import mlflow

    from datetime import datetime
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    # ----------------------
    # Example compatible base model helper
    # ----------------------
    def example_compatible_base_model(img_size=(120, 120), aux_column_lengths: dict = None):
        """Example compatible base model (backbone) demonstrating expected input signatures.

        - image input: (H, W, 3)
        - for each aux column a 1D vector of length L
        The model returns a vector representation (not final predictions). Use this
        as a baseline for building custom backbones that integrate image + aux inputs.
        """
        if aux_column_lengths is None:
            aux_column_lengths = {}

        image_input = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="image_input")
        aux_inputs = [tf.keras.Input(shape=(L,), name=f"aux_{c}") for c, L in aux_column_lengths.items()]

        # simple conv backbone for image -> vector
        y = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(image_input)
        y = tf.keras.layers.MaxPool2D()(y)
        y = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(y)
        y = tf.keras.layers.GlobalAveragePooling2D()(y)  # -> vector

        aux_heads = []
        for inp in aux_inputs:
            h = tf.keras.layers.Dense(32, activation="relu")(inp)
            aux_heads.append(h)

        if aux_heads:
            merged = tf.keras.layers.concatenate([y] + aux_heads)
        else:
            merged = y

        merged = tf.keras.layers.Dense(128, activation="relu")(merged)
        model = tf.keras.Model(inputs=[image_input] + aux_inputs, outputs=merged, name="example_backbone")
        return model

    @staticmethod
    def _build_tf_dataset_from_matrix_column_streaming(
            csv_files: Iterable[str],
            index_set: set,
            label_to_int: dict,
            aux_columns: list,
            aux_col_lengths: dict,
            aux_max: dict,
            img_size=(120, 120),
            preprocess_fn: Callable = None,         # numpy-based fallback
            tf_preprocess_fn: Callable = None,      # preferred: tf.Tensor -> tf.Tensor
            batch_size: int = 24,
            shuffle: bool = True,
            seed: int = 42,
            use_aux_inputs: bool = True,
        ):
        """
        Streaming tf.data.Dataset builder that prioritizes TF ops where possible,
        but keeps the generator-based input pipeline intact.

        Returns:
            (dataset, tried_device_prefetch: bool)
        """
        import csv as _csv
        import tensorflow as tf
        from tensorflow.data import AUTOTUNE

        files = list(csv_files)

        # cap aux lengths (safety)
        MAX_AUX_LEN = 1024
        for k, orig in list(aux_col_lengths.items()):
            if orig > MAX_AUX_LEN:
                try:
                    logger and logger.warning(
                        "Aux column '%s' max length %d capped to %d for streaming.",
                        k, orig, MAX_AUX_LEN
                    )
                except Exception:
                    pass
                aux_col_lengths[k] = MAX_AUX_LEN

        def _parse_field_text_to_obj(text):
            if text is None or (isinstance(text, str) and text.strip() == ""):
                return None
            s = text.strip()
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
            try:
                return json.loads(s)
            except Exception:
                pass
            return None

        # ---------- generator: pure python / numpy / io ----------
        def generator():
            global_idx = 0
            try:
                for fpath in files:
                    try:
                        # log which file we're currently reading (helpful)
                        try:
                            logger and logger.info("Generator: reading file %s", fpath)
                        except Exception:
                            pass

                        with open(fpath, newline="") as fh:
                            reader = _csv.DictReader(fh)
                            for row in reader:
                                if global_idx in index_set:
                                    # label lookup
                                    lab = (row.get("behavior_type_label") or row.get("behavior_type") or row.get("label"))
                                    if lab is not None:
                                        lab = lab.strip()

                                    lab_int = label_to_int.get(global_idx)
                                    if lab_int is None and lab is not None:
                                        lab_int = label_to_int.get(lab)
                                    if lab_int is None:
                                        global_idx += 1
                                        continue

                                    # parse matrix -> numpy
                                    mat_text = (row.get("trajectory_image_matrix") or row.get("trajectory_image") or row.get("matrix"))
                                    parsed_mat = _parse_field_text_to_obj(mat_text)
                                    if parsed_mat is None:
                                        global_idx += 1
                                        continue

                                    try:
                                        arr = np.array(parsed_mat)
                                        # coerce to HxWx3
                                        if arr.ndim == 2:
                                            arr_rgb = np.stack([arr, arr, arr], axis=-1)
                                        elif arr.ndim == 3:
                                            if arr.shape[2] == 3:
                                                arr_rgb = arr
                                            elif arr.shape[2] == 1:
                                                arr_rgb = np.concatenate([arr, arr, arr], axis=-1)
                                            else:
                                                arr_rgb = arr[:, :, :3]
                                        else:
                                            flat = arr.flatten()
                                            total_pix = flat.size // 3
                                            if total_pix <= 0:
                                                global_idx += 1
                                                try:
                                                    del arr
                                                except Exception:
                                                    pass
                                                gc.collect()
                                                continue
                                            side = int(np.sqrt(total_pix))
                                            flat = flat[: (side * side * 3)]
                                            arr_rgb = flat.reshape((side, side, 3))
                                    except Exception:
                                        try:
                                            del parsed_mat
                                        except Exception:
                                            pass
                                        gc.collect()
                                        global_idx += 1
                                        continue

                                    # Normalize to uint8 to reduce host->device transfer size
                                    try:
                                        if np.issubdtype(arr_rgb.dtype, np.floating):
                                            maxv = float(np.nanmax(arr_rgb)) if arr_rgb.size else 0.0
                                            if maxv <= 1.0:
                                                img_out = (np.clip(arr_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                                            else:
                                                img_out = (np.clip(arr_rgb, 0.0, 255.0)).astype(np.uint8)
                                        else:
                                            img_out = np.clip(arr_rgb, 0, 255).astype(np.uint8)
                                    except Exception:
                                        try:
                                            del arr, arr_rgb
                                        except Exception:
                                            pass
                                        gc.collect()
                                        global_idx += 1
                                        continue

                                    # Build aux arrays
                                    aux_inputs = []
                                    if use_aux_inputs:
                                        for col in aux_columns:
                                            L = aux_col_lengths.get(col, 0)
                                            raw_aux = row.get(col, None)
                                            parsed_aux = _parse_field_text_to_obj(raw_aux)
                                            if L <= 0:
                                                vec = np.zeros((0,), dtype=np.float32)
                                            else:
                                                if parsed_aux is None:
                                                    vec = np.zeros((L,), dtype=np.float32)
                                                else:
                                                    try:
                                                        arr_aux = np.array([float(x) for x in parsed_aux], dtype=np.float32)
                                                    except Exception:
                                                        arr_aux = np.zeros((0,), dtype=np.float32)
                                                    n = arr_aux.size
                                                    if n == 0:
                                                        vec = np.zeros((L,), dtype=np.float32)
                                                    elif n == L:
                                                        vec = arr_aux.astype(np.float32)
                                                    elif n < L:
                                                        pad = np.zeros((L - n,), dtype=np.float32)
                                                        vec = np.concatenate([arr_aux, pad], axis=0)
                                                    else:
                                                        idx_target = np.linspace(0.0, float(n - 1), num=L)
                                                        vec = np.interp(idx_target, np.arange(n, dtype=np.float32), arr_aux).astype(np.float32)
                                            maxv = float(aux_max.get(col, 1.0) or 1.0)
                                            if maxv != 0.0 and vec.size > 0:
                                                vec = vec / maxv
                                            aux_inputs.append(vec.astype(np.float32))

                                    if use_aux_inputs and len(aux_inputs) > 0:
                                        inputs = (img_out,) + tuple(aux_inputs)
                                    else:
                                        inputs = (img_out,)

                                    try:
                                        label_value = int(lab_int)
                                    except Exception:
                                        try:
                                            del arr, arr_rgb, img_out, inputs
                                        except Exception:
                                            pass
                                        gc.collect()
                                        global_idx += 1
                                        continue

                                    yield inputs, np.int32(label_value)

                                    # cleanup per-row temporaries
                                    try:
                                        del arr, arr_rgb, img_out
                                    except Exception:
                                        pass
                                    gc.collect()

                                global_idx += 1
                    except StopIteration:
                        break
                    except Exception:
                        # file-level exceptions are swallowed and pipeline continues to next file
                        continue
            finally:
                gc.collect()

        # ---------- output_signature ----------
        image_spec = tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)
        aux_specs = []
        if use_aux_inputs:
            for col in aux_columns:
                L = aux_col_lengths.get(col, 0)
                if L > 0:
                    aux_specs.append(tf.TensorSpec(shape=(L,), dtype=tf.float32))
        input_signature = ((image_spec,) + tuple(aux_specs), tf.TensorSpec(shape=(), dtype=tf.int32))

        ds = tf.data.Dataset.from_generator(generator, output_signature=input_signature)

        # shuffle small buffer
        if shuffle:
            # Smaller shuffle buffer lowers RAM usage; it does not remove samples from the dataset.
            ds = ds.shuffle(buffer_size=min(256, max(32, batch_size * 4)), seed=seed)

        # map: uint8 -> float32, resize, apply tf_preprocess_fn if available
        def _map_to_model_tensors(inputs, label):
            image = inputs[0]
            aux = inputs[1:] if len(inputs) > 1 else ()

            img = tf.image.convert_image_dtype(image, dtype=tf.float32)
            img = tf.image.resize(img, (img_size[0], img_size[1]), method="bilinear")

            if callable(tf_preprocess_fn):
                try:
                    img = tf_preprocess_fn(img)
                except Exception:
                    pass
            else:
                if callable(preprocess_fn):
                    try:
                        def _call_pre(x_np):
                            try:
                                out = preprocess_fn(x_np)
                            except Exception:
                                out = x_np
                            return np.asarray(out, dtype=np.float32)

                        img = tf.numpy_function(func=_call_pre, inp=[tf.expand_dims(img, axis=0)], Tout=tf.float32)
                        img.set_shape([1, img_size[0], img_size[1], 3])
                        img = tf.squeeze(img, axis=0)
                    except Exception:
                        pass

            if aux:
                aux_tensors = tuple(tf.cast(a, tf.float32) for a in aux)
                return ((img,) + aux_tensors, label)
            else:
                return (img, label)

        ds = ds.map(_map_to_model_tensors, num_parallel_calls=AUTOTUNE)
        # batch
        ds = ds.batch(batch_size, drop_remainder=False)

        # dataset options (autotune)
        try:
            options = tf.data.Options()
            try:
                options.experimental_optimization.autotune = True
            except Exception:
                pass
            try:
                options.experimental_deterministic = False
            except Exception:
                pass
            ds = ds.with_options(options)
        except Exception:
            pass

        # Keep host-side buffering modest to reduce peak RAM usage.
        ds = ds.prefetch(1)

        tried_device_prefetch = False
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for g in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(g, True)
                    except Exception:
                        pass

                try:
                    ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1))
                    tried_device_prefetch = True
                    logger and logger.info("Dataset: applied prefetch_to_device('/GPU:0', buffer_size=1)")
                except Exception:
                    logger and logger.debug(
                        "Dataset: prefetch_to_device('/GPU:0') not available or failed; continuing with host prefetch."
                    )
                    tried_device_prefetch = False
        except Exception:
            tried_device_prefetch = False

        return ds, tried_device_prefetch


    @staticmethod
    def train_all_behavior_types_image_models_from_csv_separate_aux(
        dataset_dir: str,
        matrix_column: str = "trajectory_image_matrix",
        label_column: str = "behavior_type_label",
        aux_columns: list = None,
        models_dict: dict = None,
        base_tensorflow_model=None,
        allowed_labels: list = None,
        per_label_n: int = 100,
        test_size: float = 0.2,
        random_state: float = 42,
        img_size: tuple = (120, 120),
        epochs: int = 12,
        batch_size: int = 24,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        patience: int = 3,
        callbacks_list: list = None,
        experiment_name: str = None,
        register_model_name: str = None,
        use_aux_inputs: bool = True,
        warmup_epochs: int = 3,
        enable_mixed_precision: bool = True,
        tf_preprocess_fn: Callable = None,
    ):
        """
        Training pipeline that prefers GPU usage, while keeping the generator-backed input pipeline.
        Adds per-epoch elapsed/ETA logging per model, and falls back to an eager manual loop only
        if graph-mode fit fails due to generator serialization.

        `optimizer` accepts the built-in TensorFlow/Keras optimizer names available in the installed
        version, including: sgd, rmsprop, adam, adamw, adadelta, adagrad, adamax, adafactor, nadam,
        ftrl, lion, lamb, and muon. The helper below normalizes strings before instantiation.
        """
        import csv as _csv
        import sys as _sys
        import tensorflow as tf
        import time as _time
        import os as _os
        import random as _random

        def _format_seconds(s):
            try:
                s = int(max(0, float(s)))
            except Exception:
                return "n/a"
            hh = s // 3600
            mm = (s % 3600) // 60
            ss = s % 60
            if hh > 0:
                return f"{hh}h{mm:02d}m{ss:02d}s"
            if mm > 0:
                return f"{mm}m{ss:02d}s"
            return f"{ss}s"

        def _build_optimizer(optimizer_name: str, current_lr: float):
            # Accepts strings (e.g. "adam", "SGD", etc). If string is unrecognized, defaults to Adam.
            try:
                if optimizer_name == "sgd" or optimizer_name == "SGD":
                    return tf.keras.optimizers.SGD(learning_rate=float(current_lr))
                elif optimizer_name == "rmsprop" or optimizer_name == "RMSPROP":
                    return tf.keras.optimizers.RMSprop(learning_rate=float(current_lr))
                elif optimizer_name == "adam" or optimizer_name == "ADAM":
                    return tf.keras.optimizers.Adam(learning_rate=float(current_lr))
                elif optimizer_name == "adamw" or optimizer_name == "ADAMW":
                    return tf.keras.optimizers.AdamW(learning_rate=float(current_lr), weight_decay=0.0)
                elif optimizer_name == "adadelta" or optimizer_name == "ADADELTA":
                    return tf.keras.optimizers.Adadelta(learning_rate=float(current_lr))
                elif optimizer_name == "adagrad" or optimizer_name == "ADAGRAD":
                    return tf.keras.optimizers.Adagrad(learning_rate=float(current_lr))
                elif optimizer_name == "adamax" or optimizer_name == "ADAMAX":
                    return tf.keras.optimizers.Adamax(learning_rate=float(current_lr))
                elif optimizer_name == "adafactor" or optimizer_name == "ADAFACTOR":
                    return tf.keras.optimizers.Adafactor(learning_rate=float(current_lr))
                elif optimizer_name == "nadam" or optimizer_name == "NADAM":
                    return tf.keras.optimizers.Nadam(learning_rate=float(current_lr))
                elif optimizer_name == "ftrl" or optimizer_name == "FTRL":
                    return tf.keras.optimizers.Ftrl(learning_rate=float(current_lr))
                elif optimizer_name == "lion" or optimizer_name == "LION":
                    return tf.keras.optimizers.Lion(learning_rate=float(current_lr))
                elif optimizer_name == "lamb" or optimizer_name == "LAMB":
                    return tf.keras.optimizers.Lamb(learning_rate=float(current_lr))
                elif optimizer_name == "muon" or optimizer_name == "MUON":
                    return tf.keras.optimizers.Muon(learning_rate=float(current_lr))

                logger.info("Optimizer specified as '%s' did not match built-in names; Check if you have the correct spelling.", str(optimizer_name))
            except Exception:
                logger and logger.exception("Failed to build optimizer from string '%s'; falling back to default Adam.", str(optimizer_name))
                return tf.keras.optimizers.Adam(learning_rate=float(current_lr))

        def _log_history_metrics_to_mlflow(history_obj, model_key_local):
            try:
                history_dict = {}
                if isinstance(history_obj, dict):
                    history_dict = history_obj
                elif hasattr(history_obj, "history"):
                    history_dict = history_obj.history or {}

                if not history_dict:
                    return 0

                n_epochs_total_local = 0
                for values in history_dict.values():
                    try:
                        n_epochs_total_local = max(n_epochs_total_local, len(values))
                    except Exception:
                        pass

                for epoch_idx_local in range(n_epochs_total_local):
                    for metric_name_local, values_local in history_dict.items():
                        try:
                            if epoch_idx_local < len(values_local):
                                metric_value = values_local[epoch_idx_local]
                                if metric_value is not None:
                                    mlflow.log_metric(
                                        metric_name_local,
                                        float(metric_value),
                                        step=int(epoch_idx_local + 1),
                                    )
                        except Exception:
                            pass

                metrics_csv_local = os.path.join(out_dir, f"{model_key_local}_metrics_per_epoch.csv")
                try:
                    with open(metrics_csv_local, "w", newline="") as fh:
                        writer = _csv.writer(fh)
                        header = ["epoch"] + list(history_dict.keys())
                        writer.writerow(header)
                        for e_local in range(n_epochs_total_local):
                            row = [e_local + 1]
                            for k_local in history_dict.keys():
                                vals_local = history_dict.get(k_local, [])
                                row.append(vals_local[e_local] if e_local < len(vals_local) else "")
                            writer.writerow(row)
                    try:
                        mlflow.log_artifact(metrics_csv_local, artifact_path="metrics")
                        logger and logger.info("Logged per-epoch metrics CSV to MLflow: %s", metrics_csv_local)
                    except Exception:
                        logger and logger.debug("Couldn't log per-epoch metrics CSV to MLflow.")
                except Exception:
                    logger and logger.exception("Failed writing per-epoch metrics CSV.")
                return n_epochs_total_local
            except Exception:
                logger and logger.exception("Failed logging history metrics to MLflow.")
                return 0

        def _build_mlflow_input_example(model_obj):
            try:
                model_inputs_local = list(getattr(model_obj, "inputs", []) or [])
                if len(model_inputs_local) <= 1:
                    return np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32)

                example_local = {}
                for inp_local in model_inputs_local:
                    try:
                        input_name_local = str(getattr(inp_local, "name", "input")).split(":")[0]
                    except Exception:
                        input_name_local = "input"

                    if "image" in input_name_local.lower():
                        example_local[input_name_local] = np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32)
                    else:
                        matched_col_local = None
                        for col_local in aux_columns:
                            if col_local in input_name_local:
                                matched_col_local = col_local
                                break
                        L_local = int(aux_col_lengths.get(matched_col_local, 0) or 0) if matched_col_local else 0
                        example_local[input_name_local] = np.zeros((1, L_local), dtype=np.float32)
                return example_local
            except Exception:
                return np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32)


        def _save_model_architecture_artifacts(model_obj, model_key_local, out_dir_local):
            """
            Save a model diagram PNG and a plain-text model.summary() artifact.
            Falls back to a text-rendered PNG when tensorflow's plot_model backend
            is unavailable (e.g., missing pydot/graphviz).
            """
            import os
            import io
            import re as _re
            import textwrap as _textwrap

            artifact_dir_local = os.path.join(out_dir_local, "model_architecture")
            os.makedirs(artifact_dir_local, exist_ok=True)

            summary_path_local = os.path.join(artifact_dir_local, f"{model_key_local}_model_summary.txt")
            diagram_path_local = os.path.join(artifact_dir_local, f"{model_key_local}_model_diagram.png")
            nested_diagram_path_local = os.path.join(artifact_dir_local, f"{model_key_local}_nested_model_diagram.png")

            # Save the textual summary first.
            try:
                with open(summary_path_local, "w", encoding="utf-8") as fh:
                    model_obj.summary(print_fn=lambda s: fh.write(s + "\n"))
            except Exception as e_sum:
                with open(summary_path_local, "w", encoding="utf-8") as fh:
                    fh.write(f"Could not generate model.summary(): {e_sum}\n")

            # Try the native TensorFlow diagram path.
            diagram_saved = False
            try:
                from tensorflow.keras.utils import plot_model as _plot_model

                _plot_model(
                    model_obj,
                    to_file=diagram_path_local,
                    show_shapes=True,
                    show_layer_names=True,
                    expand_nested=False,
                    dpi=400,
                )

                _plot_model(
                    model_obj,
                    to_file=nested_diagram_path_local,
                    show_shapes=True,
                    show_layer_names=True,
                    expand_nested=True,
                    dpi=400,
                )

                diagram_saved = True
            except Exception as e_plot:
                logger and logger.warning(
                    "Could not save model diagram artifact for %s with plot_model: %s",
                    model_key_local,
                    str(e_plot),
                )

            # Fallback: render the summary text into a PNG so a diagram artifact always exists.
            if not diagram_saved:
                try:
                    import matplotlib.pyplot as _plt

                    fig = _plt.figure(figsize=(12, 0.35 * max(12, len(open(summary_path_local, encoding="utf-8").read().splitlines()))))
                    _plt.axis("off")
                    with open(summary_path_local, "r", encoding="utf-8") as fh:
                        summary_text_local = fh.read()

                    # keep the PNG reasonably compact
                    summary_text_local = "\n".join(summary_text_local.splitlines()[:250])
                    summary_text_local = _textwrap.shorten(summary_text_local, width=12000, placeholder="\n...[truncated]...")
                    _plt.text(
                        0.01,
                        0.99,
                        summary_text_local,
                        va="top",
                        ha="left",
                        family="monospace",
                        fontsize=8,
                    )
                    _plt.savefig(diagram_path_local, bbox_inches="tight")
                    _plt.close(fig)
                    diagram_saved = True
                except Exception as e_fallback:
                    # As a last resort, create a small text file with the same name so logging does not break.
                    with open(diagram_path_local.replace(".png", ".txt"), "w", encoding="utf-8") as fh:
                        fh.write(f"Could not render diagram PNG for {model_key_local}: {e_fallback}\n")

            # Log both artifacts to MLflow.
            try:
                mlflow.log_artifact(summary_path_local, artifact_path="model_architecture")
            except Exception:
                logger and logger.exception("Failed logging model summary artifact for %s.", model_key_local)
            try:
                mlflow.log_artifact(diagram_path_local, artifact_path="model_architecture")
            except Exception:
                logger and logger.exception("Failed logging model diagram artifact for %s.", model_key_local)
            try:
                mlflow.log_artifact(nested_diagram_path_local, artifact_path="model_architecture")
            except Exception:
                logger and logger.exception("Failed logging nested model diagram artifact for %s.", model_key_local)

            return {
                "summary": summary_path_local,
                "diagram": diagram_path_local,
                "nested_diagram": nested_diagram_path_local,
                "artifact_dir": artifact_dir_local,
            }

        def _log_model_to_mlflow_resiliently(model, model_key: str, out_dir: str):
            import json
            import os
            import shutil
            import tempfile
            import sys as _sys
            import yaml as _yaml

            # 1) Try the normal MLflow Keras path first.
            try:
                mlflow.keras.log_model(model, artifact_path="model")
                logger.info("Logged Keras model to MLflow using the standard Keras path for %s.", model_key)
                return {"mode": "mlflow.keras.log_model", "artifact_path": "model"}
            except Exception as e1:
                logger.warning(
                    "Primary MLflow Keras logging failed for %s; trying MLflow TensorFlow/SavedModel fallback. Error: %s",
                    model_key,
                    e1,
                )

            # 2) Build a full MLflow model directory manually around a raw SavedModel export.
            #    This avoids Keras serialization/deepcopy issues while still creating MLflow
            #    metadata files (MLmodel, conda.yaml, python_env.yaml, requirements.txt).
            tmp_model_dir = tempfile.mkdtemp(prefix=f"{model_key}_mlflow_model_")
            data_model_dir = os.path.join(tmp_model_dir, "data", "model")
            os.makedirs(data_model_dir, exist_ok=True)

            try:
                # Raw TensorFlow export (works even when Keras save/log_model fails).
                tf.saved_model.save(model, data_model_dir)

                # MLflow metadata
                try:
                    from mlflow.models import Model as _MlflowModel
                except Exception:
                    from mlflow.models.model import Model as _MlflowModel  # pragma: no cover

                mlflow_model = _MlflowModel()
                mlflow_model.add_flavor(
                    "tensorflow",
                    saved_model_dir=os.path.join("data", "model"),
                    model_type="tf2-module",
                )

                try:
                    mlflow.pyfunc.add_to_model(
                        mlflow_model,
                        loader_module="mlflow.tensorflow",
                        conda_env="conda.yaml",
                        python_env="python_env.yaml",
                    )
                except Exception:
                    # Keep the MLflow model valid even if pyfunc metadata helper changes.
                    pass

                mlflow_model.save(os.path.join(tmp_model_dir, "MLmodel"))

                # Environment files
                try:
                    conda_env = mlflow.tensorflow.get_default_conda_env()
                except Exception:
                    conda_env = {
                        "name": "mlflow-env",
                        "channels": ["conda-forge"],
                        "dependencies": [
                            f"python={_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
                            {"pip": [f"tensorflow=={tf.__version__}"]},
                        ],
                    }

                try:
                    pip_reqs = mlflow.tensorflow.get_default_pip_requirements()
                except Exception:
                    pip_reqs = [f"tensorflow=={tf.__version__}"]

                with open(os.path.join(tmp_model_dir, "conda.yaml"), "w", encoding="utf-8") as fh:
                    _yaml.safe_dump(conda_env, stream=fh, default_flow_style=False)

                with open(os.path.join(tmp_model_dir, "requirements.txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(pip_reqs) + "\n")

                # Best-effort python_env.yaml that matches the MLflow layout.
                try:
                    import importlib.metadata as _importlib_metadata
                    pip_ver = _importlib_metadata.version("pip")
                    setuptools_ver = _importlib_metadata.version("setuptools")
                    wheel_ver = _importlib_metadata.version("wheel")
                except Exception:
                    pip_ver = "unknown"
                    setuptools_ver = "unknown"
                    wheel_ver = "unknown"

                python_env_payload = {
                    "python": f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
                    "build_dependencies": [
                        f"pip=={pip_ver}",
                        f"setuptools=={setuptools_ver}",
                        f"wheel=={wheel_ver}",
                    ],
                    "dependencies": ["-r requirements.txt"],
                }
                with open(os.path.join(tmp_model_dir, "python_env.yaml"), "w", encoding="utf-8") as fh:
                    _yaml.safe_dump(python_env_payload, stream=fh, default_flow_style=False)

                manifest_path = os.path.join(tmp_model_dir, f"{model_key}_mlflow_export_manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "model_key": model_key,
                            "export_mode": "tf.saved_model.save",
                            "artifact_path": "model",
                            "mlflow_dir": True,
                        },
                        fh,
                        indent=2,
                    )

                mlflow.log_artifacts(tmp_model_dir, artifact_path="model")
                logger.info(
                    "Logged TensorFlow SavedModel wrapped as MLflow artifacts for %s.",
                    model_key,
                )
                return {
                    "mode": "tf.saved_model.save + mlflow_model_dir",
                    "artifact_path": "model",
                    "registered_model_ready": True,
                }

            except Exception as e2:
                logger.exception("TensorFlow/MLflow export failed for %s", model_key)

                # 3) Last-resort fallback: keep the core artifacts so nothing is lost.
                try:
                    weights_path = os.path.join(out_dir, f"{model_key}.weights.h5")
                    model.save_weights(weights_path)

                    summary_path = os.path.join(out_dir, f"{model_key}_model_summary.txt")
                    with open(summary_path, "w", encoding="utf-8") as fh:
                        model.summary(print_fn=lambda s: fh.write(s + "\n"))

                    mlflow.log_artifacts(out_dir, artifact_path=f"fallback_{model_key}")
                    logger.warning("Saved weights + summary fallback artifact for %s.", model_key)
                    return {
                        "mode": "weights_and_summary",
                        "artifact_path": f"fallback_{model_key}",
                        "error": str(e2),
                    }
                except Exception as e3:
                    logger.exception("Final fallback also failed for %s", model_key)
                    return {"mode": "failed", "artifact_path": None, "error": f"{e2} | fallback failed: {e3}"}
            finally:
                shutil.rmtree(tmp_model_dir, ignore_errors=True)
        # --- early GPU setup ---
        gpu_available = False
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                gpu_available = True
                logger and logger.info("GPUs detected: %s", gpus)
                for g in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(g, True)
                    except Exception:
                        pass
                if enable_mixed_precision:
                    try:
                        from tensorflow.keras import mixed_precision
                        mixed_precision.set_global_policy("mixed_float16")
                        logger and logger.info("Enabled mixed precision policy: mixed_float16")
                    except Exception:
                        logger and logger.debug("Could not enable mixed precision policy.")
            else:
                logger and logger.info("No physical GPUs detected; training will run on CPU.")
        except Exception as e:
            logger and logger.debug("GPU setup failed: %s", e)
            gpu_available = False

        # increase CSV parser field size limit
        _max_int = _sys.maxsize
        while True:
            try:
                _csv.field_size_limit(_max_int)
                break
            except OverflowError:
                _max_int = int(_max_int / 10)

        if aux_columns is None:
            aux_columns = ["sog_array", "cog_array", "timestamp_array"]
        if allowed_labels is None:
            allowed_labels = ["LOITERING", "NORMAL", "STOPPING", "TRANSSHIPMENT"]

        if not os.path.isdir(dataset_dir):
            return {"error": f"dataset_dir not found or not a directory: {dataset_dir}"}
        logger and logger.info("Scanning dataset_dir for CSV files: %s", dataset_dir)

        csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
        if len(csv_files) == 0:
            return {"error": f"No CSV files found in dataset_dir: {dataset_dir}"}

        # Prepare progress logging (bytes)
        try:
            total_bytes_all = sum((_os.path.getsize(p) for p in csv_files))
        except Exception:
            total_bytes_all = 0

        start_all = _time.time()
        bytes_processed_cumulative_before = 0

        # first pass - labels and aux metadata with progress logging
        labels_list = []
        shapes_set = set()
        aux_col_lengths = {c: 0 for c in aux_columns}
        aux_max = {c: 0.0 for c in aux_columns}
        total_rows = 0
        skipped_rows = 0

        def _parse_field_text_local(text):
            if text is None or (isinstance(text, str) and text.strip() == ""):
                return None
            s = text.strip()
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
            try:
                return json.loads(s)
            except Exception:
                pass
            return None

        for idx, fpath in enumerate(csv_files):
            try:
                try:
                    file_size = _os.path.getsize(fpath)
                except Exception:
                    file_size = 0

                logger and logger.info(
                    "Reading CSV %d/%d: %s (size=%s bytes). Total dataset size ~ %s bytes.",
                    idx + 1,
                    len(csv_files),
                    fpath,
                    file_size,
                    total_bytes_all,
                )

                file_start = _time.time()
                last_log_time = file_start
                rows_in_file = 0

                with open(fpath, newline="") as fh:
                    reader = _csv.DictReader(fh)
                    for row in reader:
                        total_rows += 1
                        rows_in_file += 1

                        lab = row.get(label_column) or row.get("behavior_type_label")
                        if lab is None or lab.strip() == "" or lab.strip() not in allowed_labels:
                            skipped_rows += 1
                        else:
                            labels_list.append(lab.strip())

                            # aux stats
                            for col in aux_columns:
                                raw = row.get(col, None)
                                parsed = _parse_field_text_local(raw)
                                if parsed is None:
                                    continue
                                try:
                                    plen = len(parsed)
                                except Exception:
                                    plen = 0
                                if plen > aux_col_lengths[col]:
                                    aux_col_lengths[col] = plen
                                try:
                                    cand = float(np.max(np.abs(np.array(parsed, dtype=float))))
                                    if cand > aux_max[col]:
                                        aux_max[col] = cand
                                except Exception:
                                    pass

                            # detect image shape
                            mat_text = row.get(matrix_column, None)
                            parsed_mat = _parse_field_text_local(mat_text)
                            if parsed_mat is not None:
                                try:
                                    if isinstance(parsed_mat, list) and len(parsed_mat) > 0:
                                        h = len(parsed_mat)
                                        first_row = parsed_mat[0]
                                        if isinstance(first_row, list):
                                            w = len(first_row)
                                        else:
                                            w = 1
                                        shapes_set.add((h, w))
                                except Exception:
                                    pass

                        now = _time.time()
                        if rows_in_file % 1000 == 0 or (now - last_log_time) >= 5.0:
                            try:
                                try:
                                    bytes_read_in_file = fh.tell()
                                except Exception:
                                    bytes_read_in_file = 0

                                processed_bytes = bytes_processed_cumulative_before + bytes_read_in_file
                                pct_file = (bytes_read_in_file / file_size * 100.0) if file_size else None
                                pct_all = (processed_bytes / total_bytes_all * 100.0) if total_bytes_all else None

                                elapsed = now - start_all
                                est_remaining_all = None
                                if total_bytes_all and processed_bytes > 0:
                                    frac = processed_bytes / total_bytes_all
                                    if frac > 0:
                                        est_total = elapsed / frac
                                        est_remaining_all = est_total - elapsed

                                logger and logger.info(
                                    "Progress: file %d/%d '%s' rows=%d, file_progress=%s, overall_progress=%s, elapsed=%s, eta=%s",
                                    idx + 1,
                                    len(csv_files),
                                    os.path.basename(fpath),
                                    rows_in_file,
                                    f"{pct_file:.2f}%" if pct_file is not None else "n/a",
                                    f"{pct_all:.2f}%" if pct_all is not None else "n/a",
                                    _format_seconds(elapsed),
                                    _format_seconds(est_remaining_all) if est_remaining_all is not None else "n/a",
                                )
                            except Exception:
                                pass
                            last_log_time = now

                try:
                    bytes_processed_cumulative_before += file_size
                except Exception:
                    pass
                file_elapsed = _time.time() - file_start
                try:
                    logger and logger.info(
                        "Finished reading CSV %d/%d: %s — rows_in_file=%d, time=%s",
                        idx + 1,
                        len(csv_files),
                        fpath,
                        rows_in_file,
                        _format_seconds(file_elapsed),
                    )
                except Exception:
                    pass

            except Exception:
                logger and logger.exception("Failed first-pass read of %s", fpath)
                return {"error": f"Failed reading CSV {fpath} during first pass"}

        if len(labels_list) == 0:
            return {"error": "No usable rows found in CSV files after first pass."}

        logger and logger.info("First pass done: total_rows=%d, usable=%d, skipped=%d", total_rows, len(labels_list), skipped_rows)
        logger and logger.info("Detected image shapes (sample): %s", shapes_set)
        logger and logger.info("aux_col_lengths: %s", aux_col_lengths)
        logger and logger.info("aux_max: %s", aux_max)

        # fallback aux_max
        for k, v in aux_max.items():
            if v == 0.0:
                aux_max[k] = 1.0

        # cap aux lengths again
        MAX_AUX_LEN = 1024
        for k, orig in list(aux_col_lengths.items()):
            if orig > MAX_AUX_LEN:
                logger and logger.warning("Aux column '%s' max length %d capped to %d for streaming.", k, orig, MAX_AUX_LEN)
                aux_col_lengths[k] = MAX_AUX_LEN

        # sampling per_label_n
        label_to_indices = {lab: [] for lab in set(labels_list)}
        total_idx = 0
        for fpath in csv_files:
            with open(fpath, newline="") as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    lab = row.get(label_column) or row.get("behavior_type_label")
                    if lab is None or lab.strip() == "" or lab.strip() not in allowed_labels:
                        total_idx += 1
                        continue
                    lab = lab.strip()
                    label_to_indices.setdefault(lab, []).append(total_idx)
                    total_idx += 1

        chosen_global_indices = []
        rng = _random.Random(int(random_state))
        logger and logger.info("Applying per-label sampling per_label_n=%s", per_label_n)
        if per_label_n and per_label_n > 0:
            for lab, idxs in label_to_indices.items():
                if len(idxs) <= per_label_n:
                    chosen = idxs
                else:
                    chosen = rng.sample(idxs, per_label_n)
                chosen_global_indices.extend(chosen)
        else:
            for idxs in label_to_indices.values():
                chosen_global_indices.extend(idxs)

        chosen_global_indices = sorted(chosen_global_indices)
        # fetch chosen labels
        chosen_labels = []
        chosen_set = set(chosen_global_indices)
        gidx = 0
        for fpath in csv_files:
            with open(fpath, newline="") as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    if gidx in chosen_set:
                        lab = row.get(label_column) or row.get("behavior_type_label")
                        chosen_labels.append(lab.strip())
                    gidx += 1

        le = LabelEncoder()
        y_int_all = le.fit_transform(chosen_labels)
        logger and logger.info("Final classes: %s", list(le.classes_))
        logger and logger.info("Total chosen samples: %d", len(y_int_all))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state))
        indices = np.arange(len(chosen_global_indices))
        train_sub_idx, test_sub_idx = next(sss.split(indices, y_int_all))
        train_global_idx = set(int(chosen_global_indices[i]) for i in train_sub_idx)
        test_global_idx = set(int(chosen_global_indices[i]) for i in test_sub_idx)

        logger and logger.info("Train samples: %d, Test samples: %d", len(train_sub_idx), len(test_sub_idx))

        # build index -> label int mapping
        global_to_pos = {int(v): i for i, v in enumerate(chosen_global_indices)}
        index_to_labelint = {}
        gidx = 0
        with_indices_populated = 0
        for fpath in csv_files:
            with open(fpath, newline="") as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    if gidx in global_to_pos:
                        pos = global_to_pos[gidx]
                        index_to_labelint[gidx] = int(y_int_all[pos])
                        with_indices_populated += 1
                    gidx += 1
        logger and logger.info("Built index->labelint mapping entries=%d", with_indices_populated)

        # write aux_meta artifact
        aux_meta = {"aux_max": aux_max, "aux_col_lengths": aux_col_lengths, "aux_columns": aux_columns}
        aux_meta_path = os.path.join("artifacts", "aux_meta.json")
        os.makedirs(os.path.dirname(aux_meta_path), exist_ok=True)
        with open(aux_meta_path, "w") as fh:
            json.dump(aux_meta, fh)
        logger and logger.info("Wrote aux_meta to %s", aux_meta_path)

        # Free first-pass CSV scan structures as soon as they are no longer needed.
        try:
            del labels_list
            del shapes_set
            del total_rows
            del skipped_rows
            del label_to_indices
            del chosen_labels
            del chosen_set
            del chosen_global_indices
            del global_to_pos
            del y_int_all
            del indices
            del train_sub_idx
            del test_sub_idx
        except Exception:
            pass
        gc.collect()

        # base model compatibility check (unchanged)
        base_integration_success = False
        integration_mode = None
        base_model_name = None
        if base_tensorflow_model is not None:
            base_model_name = getattr(base_tensorflow_model, "name", str(base_tensorflow_model))
            logger and logger.info("Performing strict base model compatibility check for %s", base_model_name)
            img_dummy = tf.zeros((1, img_size[0], img_size[1], 3), dtype=tf.float32)
            aux_dummies = []
            for col in aux_columns:
                L = aux_col_lengths.get(col, 0)
                if L > 0:
                    aux_dummies.append(tf.zeros((1, L), dtype=tf.float32))
                else:
                    aux_dummies.append(tf.zeros((1, 0), dtype=tf.float32))
            errors = []
            try:
                try:
                    _ = base_tensorflow_model([img_dummy] + aux_dummies)
                    base_integration_success = True
                    integration_mode = "all_inputs"
                except Exception as e_a:
                    errors.append(("pattern_all_inputs", str(e_a)))
                    try:
                        _ = base_tensorflow_model(img_dummy)
                        base_integration_success = True
                        integration_mode = "image_only"
                    except Exception as e_b:
                        errors.append(("pattern_image_only", str(e_b)))
                        base_integration_success = False
                        integration_mode = None
            except Exception as e_general:
                errors.append(("general", str(e_general)))
                base_integration_success = False
                integration_mode = None

            if not base_integration_success:
                msg = (
                    "Provided base_tensorflow_model is incompatible with expected input signatures.\n"
                    "Tried calling it with either ([image] + aux_vectors) or (image) and both failed.\n"
                    f"aux_col_lengths: {aux_col_lengths}\n"
                    f"errors: {errors}\n"
                    "Please adapt your model input signature to accept either (image) OR (image, aux1, aux2, ...).\n"
                )
                logger and logger.error(msg)
                raise ValueError(msg)
            else:
                logger and logger.info(
                    "base_tensorflow_model '%s' integrated successfully using mode=%s",
                    str(base_model_name),
                    integration_mode,
                )

        train_ds, train_prefetched_to_device = TrainModelService._build_tf_dataset_from_matrix_column_streaming(
            csv_files=csv_files,
            index_set=train_global_idx,
            label_to_int=index_to_labelint,
            aux_columns=aux_columns,
            aux_col_lengths=aux_col_lengths,
            aux_max=aux_max,
            img_size=img_size,
            preprocess_fn=None,
            tf_preprocess_fn=tf_preprocess_fn,
            batch_size=batch_size,
            shuffle=True,
            use_aux_inputs=use_aux_inputs,
        )

        test_ds, test_prefetched_to_device = TrainModelService._build_tf_dataset_from_matrix_column_streaming(
            csv_files=csv_files,
            index_set=test_global_idx,
            label_to_int=index_to_labelint,
            aux_columns=aux_columns,
            aux_col_lengths=aux_col_lengths,
            aux_max=aux_max,
            img_size=img_size,
            preprocess_fn=None,
            tf_preprocess_fn=tf_preprocess_fn,
            batch_size=batch_size,
            shuffle=False,
            use_aux_inputs=use_aux_inputs,
        )

        # Optional check: log whether sample batches appear to land on GPU
        def _sample_dataset_devices(ds, n_samples=3):
            observed = set()
            try:
                it = iter(ds)
                for _ in range(n_samples):
                    try:
                        batch = next(it)
                    except Exception:
                        break

                    elems = batch if isinstance(batch, (list, tuple)) else (batch,)

                    def _collect_devices(x):
                        devs = set()
                        if isinstance(x, (list, tuple)):
                            for y in x:
                                devs |= _collect_devices(y)
                        else:
                            try:
                                d = getattr(x, "device", None)
                                if d:
                                    devs.add(d)
                                else:
                                    devs.add("host:CPU")
                            except Exception:
                                devs.add("host:CPU")
                        return devs

                    for el in elems:
                        observed |= _collect_devices(el)
            except Exception:
                pass
            return observed

        # build preprocess layer for inside-model preprocessing
        def _make_preprocess_layer(pre_fn):
            if pre_fn is None:
                return tf.keras.layers.Lambda(lambda x: x, name="identity_preprocess")

            def _tf_preprocess(x):
                def _np_call(x_np):
                    try:
                        out = pre_fn(x_np)
                    except Exception:
                        out = x_np
                    out = np.asarray(out, dtype=np.float32)
                    return out

                y = tf.numpy_function(_np_call, [x], Tout=tf.float32)
                try:
                    y.set_shape([None, img_size[0], img_size[1], 3])
                except Exception:
                    pass
                return y

            return tf.keras.layers.Lambda(_tf_preprocess, name="model_preprocess")

        # iterate models and build/compile inside device strategy scope if GPU available
        results = {}
        iter_items = (models_dict.items() if models_dict else [("custom", None)])

        try:
            strategy = None
            if gpu_available:
                try:
                    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
                except Exception:
                    strategy = None

            for model_key, iter_base in iter_items:
                if models_dict:
                    logger and logger.info("=== Training model from models_dict: %s ===", model_key)

                out_dir = os.path.join("artifacts", "image_models", model_key)
                os.makedirs(out_dir, exist_ok=True)

                chosen_base = base_tensorflow_model if base_tensorflow_model is not None else iter_base
                exp_name = experiment_name or f"image_training_{model_key}"
                if chosen_base is not None:
                    prefix = getattr(chosen_base, "name", None) or str(chosen_base)
                    exp_name = f"{prefix}_{exp_name}"

                per_model_pre_fn = TrainModelService._get_preprocess_fn_for_model(model_key)
                preprocess_layer = _make_preprocess_layer(per_model_pre_fn)

                with strategy.scope() if strategy is not None else tf.device("/CPU:0"):
                    image_input = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="image_input")
                    image_preprocessed = preprocess_layer(image_input)

                    internal_conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="internal_conv1")
                    internal_pool1 = tf.keras.layers.MaxPool2D(name="internal_pool1")
                    internal_conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="internal_conv2")
                    internal_pool2 = tf.keras.layers.MaxPool2D(name="internal_pool2")
                    internal_conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="internal_conv3")
                    internal_gap = tf.keras.layers.GlobalAveragePooling2D(name="internal_gap")

                    aux_inputs = []
                    aux_processed = []
                    aux_conv_layers = []
                    aux_pool_layers = []
                    aux_dense_layers = []
                    aux_dropout_layers = []

                    if use_aux_inputs:
                        for col in aux_columns:
                            L = aux_col_lengths.get(col, 0)
                            if L <= 0:
                                logger and logger.info("Skipping aux column %s because max length == 0", col)
                                continue

                            inp = tf.keras.Input(shape=(L,), name=f"aux_{col}")
                            aux_inputs.append(inp)

                            # dense_specs entries: (units, dropout_rate, dense_name, dropout_name)
                            dense_specs = [
                                #(512, 0.2, f"aux_dense_{col}", f"aux_drop_{col}"),
                                (128, None, f"aux_dense_{col}", None),
                                (32, None, f"aux_dense2_{col}", None),
                                # (128, None, f"aux_dense3_{col}", None),
                                # (128, None, f"aux_dense4_{col}", None),
                            ]

                            h = inp
                            for units, dropout_rate, dense_name, drop_name in dense_specs:
                                dense_layer = tf.keras.layers.Dense(units, activation="relu", name=dense_name)
                                h = dense_layer(h)
                                aux_dense_layers.append(dense_layer)

                                if dropout_rate is not None:
                                    drop_layer = tf.keras.layers.Dropout(dropout_rate, name=drop_name)
                                    h = drop_layer(h)
                                    aux_dropout_layers.append(drop_layer)

                            aux_processed.append(h)

                    x = None
                    used_internal_backbone = False
                    backbone_model = None

                    try:
                        if chosen_base is not None:
                            # Clone inside the active strategy scope so the variables belong to this scope.
                            # This keeps the original tf.keras.applications architecture visible in
                            # model.summary() / plot_model(expand_nested=True), and also avoids the
                            # distribution-scope variable error.
                            backbone_model = tf.keras.models.clone_model(chosen_base)
                            try:
                                backbone_model.set_weights(chosen_base.get_weights())
                            except Exception:
                                pass

                            backbone_model._name = getattr(chosen_base, "name", f"{model_key}_backbone")
                            backbone_model.trainable = True
                            chosen_base = backbone_model  # keep later freeze/unfreeze logic working

                            # For models_dict entries, always use the original pretrained backbone directly.
                            # For a supplied custom base_tensorflow_model, keep the existing mode behavior.
                            if base_tensorflow_model is None and iter_base is not None:
                                x = chosen_base(image_preprocessed)
                                base_integration_success = True
                                integration_mode = "image_only"
                            elif base_integration_success and integration_mode == "all_inputs":
                                x = chosen_base([image_preprocessed] + aux_inputs)
                            elif base_integration_success and integration_mode == "image_only":
                                x = chosen_base(image_preprocessed)
                            else:
                                x = chosen_base(image_preprocessed)
                                base_integration_success = True
                                integration_mode = "image_only"

                            if hasattr(x, "shape") and len(x.shape) == 4:
                                x = tf.keras.layers.GlobalAveragePooling2D(name=f"{model_key}_backbone_gap")(x)
                        else:
                            x = None
                    except Exception as e:
                        logger and logger.exception(
                            "Error while building chosen_base inside strategy scope for model %s: %s",
                            model_key,
                            e,
                        )
                        x = None
                        base_integration_success = False
                        integration_mode = None

                    if x is None:
                        used_internal_backbone = True
                        y = internal_conv1(image_preprocessed)
                        y = internal_pool1(y)
                        y = internal_conv2(y)
                        y = internal_pool2(y)
                        y = internal_conv3(y)
                        y = internal_gap(y)
                        x = tf.keras.layers.Dense(128, activation="relu", name="internal_head_dense")(y)

                    # Restore the v1 head order: project the backbone output first, then merge aux features.
                    head_dense1 = tf.keras.layers.Dense(256, activation="relu", name="head_dense1")
                    head_drop1 = tf.keras.layers.Dropout(0.5, name="head_drop1")
                    x = head_dense1(x)
                    x = head_drop1(x)

                    if aux_processed:
                        merged_concat = tf.keras.layers.Concatenate(name="head_concat")([x] + aux_processed)
                        head_dense2 = tf.keras.layers.Dense(128, activation="relu", name="head_dense2")
                        head_drop2 = tf.keras.layers.Dropout(0.3, name="head_drop2")
                        merged = head_dense2(merged_concat)
                        merged = head_drop2(merged)

                        outputs = tf.keras.layers.Dense(len(le.classes_), activation="softmax", name="predictions")(merged)
                        model_inputs = [image_input] + aux_inputs
                        model = tf.keras.Model(inputs=model_inputs, outputs=outputs, name=f"{model_key}_full")
                    else:
                        outputs = tf.keras.layers.Dense(len(le.classes_), activation="softmax", name="predictions")(x)
                        model = tf.keras.Model(inputs=image_input, outputs=outputs, name=f"{model_key}_full")

                    # Keep head trainable, freeze backbone only for warmup
                    try:
                        if chosen_base is not None and base_integration_success:
                            chosen_base.trainable = False
                            logger and logger.info("Frozen chosen_base for warmup (phase 1) for model %s", model_key)
                        elif used_internal_backbone:
                            internal_conv1.trainable = False
                            internal_conv2.trainable = False
                            internal_conv3.trainable = False
                            logger and logger.info("Frozen internal backbone conv layers for warmup (phase 1) for model %s", model_key)
                    except Exception:
                        logger and logger.debug("Could not set backbone trainable flags for warmup for model %s", model_key)

                    # ensure head layers trainable
                    try:
                        head_dense1.trainable = True
                        head_drop1.trainable = True
                        if aux_processed:
                            head_dense2.trainable = True
                            head_drop2.trainable = True
                        for a in aux_dense_layers:
                            try:
                                a.trainable = True
                            except Exception:
                                pass
                    except Exception:
                        logger and logger.debug("Could not set head layer trainable flags for model %s", model_key)

                    # compile inside strategy scope (ensures optimizer vars on GPU)
                    try:
                        opt = _build_optimizer(optimizer_name, learning_rate)
                        model.compile(
                            optimizer=opt,
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                            run_eagerly=False,
                            jit_compile=bool(gpu_available),
                            steps_per_execution=16 if gpu_available else 1,
                        )
                    except Exception:
                        model.compile(
                            optimizer=_build_optimizer(optimizer_name, learning_rate),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                        )

                # prepare callbacks for the normal fit path (kept)
                cb = callbacks_list[:] if callbacks_list else []
                if not callbacks_list:
                    cb = [
                        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
                    ]
                csv_log_path = os.path.join(out_dir, f"{model_key}_epoch_history.csv")
                cb.append(tf.keras.callbacks.CSVLogger(csv_log_path))

                # add our epoch-timing callback (total epochs = warmup + fine-tune)
                total_epochs_for_model = int(warmup_epochs) + int(epochs)

                class EpochTimingCallback(tf.keras.callbacks.Callback):
                    def __init__(self, total_epochs: int, model_key: str):
                        super().__init__()
                        self.total_epochs = int(total_epochs)
                        self.model_key = model_key
                        self.start_time = None
                        self.completed_epochs = 0

                    def on_train_begin(self, logs=None):
                        if self.start_time is None:
                            self.start_time = _time.time()
                        logger and logger.info("------------------------------------------------------------")
                        logger and logger.info("********** Training started for model '%s' **********", self.model_key)
                        logger and logger.info("------------------------------------------------------------")

                    def on_epoch_end(self, epoch, logs=None):
                        self.completed_epochs += 1
                        now = _time.time()
                        elapsed = now - (self.start_time or now)
                        avg = elapsed / max(1, self.completed_epochs)
                        remaining = max(0.0, (self.total_epochs - self.completed_epochs) * avg)

                        loss_value = None
                        acc_value = None
                        val_loss_value = None
                        val_acc_value = None
                        if isinstance(logs, dict):
                            loss_value = logs.get("loss", None)
                            acc_value = logs.get("sparse_categorical_accuracy", logs.get("accuracy", None))
                            val_loss_value = logs.get("val_loss", None)
                            val_acc_value = logs.get("val_sparse_categorical_accuracy", logs.get("val_accuracy", None))

                        logger and logger.info("------------------------------------------------------------")
                        logger and logger.info("********** Model '%s' epoch %d/%d finished **********", self.model_key, self.completed_epochs, self.total_epochs)
                        logger and logger.info("loss: %s | sparse_categorical_accuracy: %s | val_loss: %s | val_sparse_categorical_accuracy: %s",
                                            ("n/a" if loss_value is None else f"{float(loss_value):.6f}"),
                                            ("n/a" if acc_value is None else f"{float(acc_value):.6f}"),
                                            ("n/a" if val_loss_value is None else f"{float(val_loss_value):.6f}"),
                                            ("n/a" if val_acc_value is None else f"{float(val_acc_value):.6f}"))
                        logger and logger.info("elapsed=%s | avg_epoch=%s | eta_total=%s",
                                            _format_seconds(elapsed),
                                            _format_seconds(avg),
                                            _format_seconds(remaining))
                        logger and logger.info("------------------------------------------------------------")

                    def manual_epoch_end(self, logs=None):
                        self.completed_epochs += 1
                        now = _time.time()
                        elapsed = now - (self.start_time or now)
                        avg = elapsed / max(1, self.completed_epochs)
                        remaining = max(0.0, (self.total_epochs - self.completed_epochs) * avg)

                        loss_value = None
                        acc_value = None
                        val_loss_value = None
                        val_acc_value = None
                        if isinstance(logs, dict):
                            loss_value = logs.get("loss", None)
                            acc_value = logs.get("sparse_categorical_accuracy", logs.get("accuracy", None))
                            val_loss_value = logs.get("val_loss", None)
                            val_acc_value = logs.get("val_sparse_categorical_accuracy", logs.get("val_accuracy", None))

                        logger and logger.info("------------------------------------------------------------")
                        logger and logger.info("********** Model '%s' manual epoch %d/%d finished **********", self.model_key, self.completed_epochs, self.total_epochs)
                        logger and logger.info("loss: %s | sparse_categorical_accuracy: %s | val_loss: %s | val_sparse_categorical_accuracy: %s",
                                            ("n/a" if loss_value is None else f"{float(loss_value):.6f}"),
                                            ("n/a" if acc_value is None else f"{float(acc_value):.6f}"),
                                            ("n/a" if val_loss_value is None else f"{float(val_loss_value):.6f}"),
                                            ("n/a" if val_acc_value is None else f"{float(val_acc_value):.6f}"))
                        logger and logger.info("elapsed=%s | avg_epoch=%s | eta_total=%s",
                                            _format_seconds(elapsed),
                                            _format_seconds(avg),
                                            _format_seconds(remaining))
                        logger and logger.info("------------------------------------------------------------")

                    def set_completed(self, n):
                        self.completed_epochs = int(n)

                timing_cb = EpochTimingCallback(total_epochs=total_epochs_for_model, model_key=model_key)
                cb.append(timing_cb)

                # Small device-placement check & log (keeps your helper usage)
                try:
                    observed_train_devices = _sample_dataset_devices(train_ds, n_samples=4)
                    observed_test_devices = _sample_dataset_devices(test_ds, n_samples=2)
                    observed_train_devices_n = {str(d) for d in observed_train_devices}
                    observed_test_devices_n = {str(d) for d in observed_test_devices}
                    found_gpu_on_train = any(("GPU" in d or "gpu" in d or "/device:GPU" in d) for d in observed_train_devices_n)
                    found_gpu_on_test = any(("GPU" in d or "gpu" in d or "/device:GPU" in d) for d in observed_test_devices_n)
                    logger and logger.info(
                        "Dataset placement check for train_ds: observed devices: %s ; gpu_available=%s ; found_gpu=%s",
                        observed_train_devices_n, gpu_available, found_gpu_on_train
                    )
                    logger and logger.info(
                        "Dataset placement check for test_ds: observed devices: %s ; gpu_available=%s ; found_gpu=%s",
                        observed_test_devices_n, gpu_available, found_gpu_on_test
                    )
                    if gpu_available and not found_gpu_on_train:
                        logger and logger.warning(
                            "GPU(s) present but sample batches from train_ds were NOT observed on GPU. "
                            "Possible reasons: generator-backed dataset, numpy_function in preprocessing, or device prefetch not supported."
                        )
                except Exception:
                    pass

                # MLflow run
                mlflow.set_experiment(exp_name)
                with mlflow.start_run(run_name=f"{model_key}_train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("model_base", model_key)
                    mlflow.log_param("num_classes", len(le.classes_))
                    mlflow.log_param("warmup_epochs", int(warmup_epochs))
                    mlflow.log_param("epochs_phase2", int(epochs))
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("learning_rate", learning_rate)
                    mlflow.log_param("optimizer", str(optimizer_name))
                    mlflow.log_param("patience", int(patience))
                    mlflow.log_param("use_aux_inputs", bool(use_aux_inputs))
                    mlflow.log_param("aux_col_lengths", aux_col_lengths)
                    mlflow.log_param("per_label_n", int(per_label_n or 0))

                    # --- Try normal model.fit path first (graph mode) ---
                    use_manual_loop = False
                    history_phase1 = None
                    history_phase2 = None

                    try:
                        logger.info(
                            "Starting phase 1 WARMUP training (head-only) for model %s (warmup_epochs=%d)",
                            model_key, warmup_epochs
                        )
                        history_phase1 = model.fit(
                            train_ds,
                            epochs=int(warmup_epochs),
                            validation_data=test_ds,
                            callbacks=cb,
                            verbose=1,
                        )
                    except Exception as e_fit:
                        msg = str(e_fit)
                        logger and logger.warning("Dataset or fit raised exception: %s", msg)
                        if "GeneratorDatasetOp::Dataset does not support serialization" in msg or "Failed to serialize the input pipeline graph" in msg:
                            logger and logger.warning(
                                "Dataset serialization error during model.fit: %s. Will fallback to manual eager training loop.",
                                msg,
                            )
                            use_manual_loop = True
                        else:
                            raise

                    # If Keras fit succeeded for phase1, update timing callback completed count
                    if history_phase1 is not None and hasattr(timing_cb, "set_completed"):
                        try:
                            num_done = 0
                            if hasattr(history_phase1, "history"):
                                num_done = len(next(iter(history_phase1.history.values()))) if history_phase1.history else 0
                            elif isinstance(history_phase1, dict):
                                num_done = len(history_phase1.get("loss", []))
                            if num_done:
                                timing_cb.set_completed(num_done)
                        except Exception:
                            pass

                    # If normal fit worked for phase1, continue to phase2 normally (unless we decided to fallback)
                    if not use_manual_loop:
                        try:
                            # unfreeze for phase 2
                            try:
                                if chosen_base is not None and base_integration_success:
                                    chosen_base.trainable = True
                                    for layer in getattr(chosen_base, "layers", []):
                                        try:
                                            layer.trainable = True
                                        except Exception:
                                            pass
                                    logger and logger.info("Unfroze chosen_base for phase 2 for model %s", model_key)
                                elif used_internal_backbone:
                                    internal_conv1.trainable = True
                                    internal_conv2.trainable = True
                                    internal_conv3.trainable = True
                                    logger and logger.info("Unfroze internal backbone conv layers for phase 2 for model %s", model_key)
                            except Exception:
                                logger and logger.debug("Failed unfreezing backbone layers for phase 2")

                            # recompile with lower lr
                            try:
                                lower_lr = float(learning_rate * 0.1)
                                opt2 = _build_optimizer(optimizer_name, lower_lr)
                                model.compile(
                                    optimizer=opt2,
                                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                                    run_eagerly=False,
                                    jit_compile=bool(gpu_available),
                                    steps_per_execution=16 if gpu_available else 1,
                                )
                            except Exception:
                                lower_learning_rate = float(learning_rate * 0.1)
                                model.compile(
                                    optimizer=_build_optimizer(optimizer_name, lower_learning_rate),
                                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                                )

                            logger.info(
                                "Starting phase 2 FINE-TUNING training (all layers trainable) for model %s (epochs=%d)",
                                model_key, epochs
                            )
                            history_phase2 = model.fit(
                                train_ds,
                                epochs=int(epochs),
                                validation_data=test_ds,
                                callbacks=cb,
                                verbose=1,
                            )
                        except Exception as e2:
                            if "GeneratorDatasetOp::Dataset does not support serialization" in str(e2) or "Failed to serialize the input pipeline graph" in str(e2):
                                logger and logger.warning("Phase2 fit serialization error: %s. Falling back to manual loop.", str(e2))
                                use_manual_loop = True
                            else:
                                raise

                    # If we decided to use a manual training loop (due to serialization problems),
                    # use conservative manual training loop using GradientTape so model ops run on GPU.
                    if use_manual_loop:
                        logger and logger.info(
                            "Using manual eager training loop for model %s (warmup_epochs=%d, phase2_epochs=%d).",
                            model_key, warmup_epochs, epochs
                        )

                        # Prepare optimizer/loss/metrics consistent with model.compile above
                        optimizer = model.optimizer if hasattr(model, "optimizer") else _build_optimizer(optimizer_name, learning_rate)
                        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
                        train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
                        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
                        val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
                        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

                        # Wrap optimizer for mixed precision if necessary
                        try:
                            policy_name = tf.keras.mixed_precision.global_policy().name
                            if "mixed_float" in policy_name:
                                from tensorflow.keras.mixed_precision import LossScaleOptimizer
                                optimizer = LossScaleOptimizer(optimizer)
                                logger and logger.info("Wrapped optimizer with LossScaleOptimizer for mixed precision.")
                        except Exception:
                            pass

                        def _run_epoch_manual(ds, training=True, max_steps=None):
                            train_loss_metric.reset_state()
                            train_acc_metric.reset_state()
                            val_loss_metric.reset_state()
                            val_acc_metric.reset_state()
                            steps = 0

                            for step, batch in enumerate(ds):
                                # Accept both (inputs, labels) and ((img, aux1...), label)
                                try:
                                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                                        inputs_batch, labels_batch = batch
                                    else:
                                        # fallback: maybe dataset yields (img,label) or direct tuple
                                        labels_batch = batch[1] if (isinstance(batch, (list, tuple)) and len(batch) > 1) else None
                                        inputs_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                                except Exception:
                                    try:
                                        inputs_batch, labels_batch = batch
                                    except Exception:
                                        # skip invalid batch
                                        continue

                                # convert numpy->tf if needed (tf will copy to device)
                                try:
                                    if isinstance(inputs_batch, (list, tuple)):
                                        inputs_tf = [tf.convert_to_tensor(x) for x in inputs_batch]
                                    else:
                                        inputs_tf = tf.convert_to_tensor(inputs_batch)
                                    labels_tf = tf.convert_to_tensor(labels_batch)
                                except Exception:
                                    # on unexpected types, try to let TF handle it
                                    inputs_tf = inputs_batch
                                    labels_tf = labels_batch

                                if training:
                                    with tf.GradientTape() as tape:
                                        preds = model(inputs_tf, training=True)
                                        loss_value = loss_fn(labels_tf, preds)
                                        try:
                                            if hasattr(optimizer, "get_scaled_loss"):
                                                scaled_loss = optimizer.get_scaled_loss(loss_value)
                                                scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
                                                grads = optimizer.get_unscaled_gradients(scaled_grads)
                                            else:
                                                grads = tape.gradient(loss_value, model.trainable_variables)
                                        except Exception:
                                            grads = tape.gradient(loss_value, model.trainable_variables)

                                    try:
                                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                                    except Exception:
                                        try:
                                            if hasattr(optimizer, "optimizer"):
                                                optimizer.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                                        except Exception:
                                            pass

                                    train_loss_metric.update_state(loss_value)
                                    train_acc_metric.update_state(labels_tf, preds)
                                else:
                                    preds = model(inputs_tf, training=False)
                                    v_loss = loss_fn(labels_tf, preds)
                                    val_loss_metric.update_state(v_loss)
                                    val_acc_metric.update_state(labels_tf, preds)

                                steps += 1
                                if max_steps and steps >= max_steps:
                                    break

                            if training:
                                return float(train_loss_metric.result().numpy()), float(train_acc_metric.result().numpy()), steps
                            else:
                                return float(val_loss_metric.result().numpy()), float(val_acc_metric.result().numpy()), steps

                        # Phase 1 manual warmup
                        history_phase1 = {"loss": [], "sparse_categorical_accuracy": [], "val_loss": [], "val_sparse_categorical_accuracy": []}
                        # ensure timing callback knows start time
                        try:
                            timing_cb.on_train_begin()
                        except Exception:
                            pass

                        for epoch_idx in range(int(warmup_epochs)):
                            # log start of manual epoch
                            logger and logger.info("Manual loop: warmup epoch %d/%d", epoch_idx + 1, warmup_epochs)
                            tr_loss, tr_acc, tr_steps = _run_epoch_manual(train_ds, training=True)
                            val_loss, val_acc, val_steps = _run_epoch_manual(test_ds, training=False)
                            history_phase1["loss"].append(tr_loss)
                            history_phase1["sparse_categorical_accuracy"].append(tr_acc)
                            history_phase1["val_loss"].append(val_loss)
                            history_phase1["val_sparse_categorical_accuracy"].append(val_acc)
                            logger and logger.info(
                                "Warmup epoch %d metrics for model %s -> train_loss=%.6f | train_acc=%.6f | val_loss=%.6f | val_acc=%.6f",
                                epoch_idx + 1,
                                model_key,
                                tr_loss,
                                tr_acc,
                                val_loss,
                                val_acc,
                            )
                            logger and logger.info(
                                "Warmup epoch %d completed for model %s (train_loss=%.4f val_loss=%.4f)",
                                epoch_idx + 1, model_key, tr_loss, val_loss
                            )
                            # notify timing callback of manual epoch completion (it will log elapsed+ETA)
                            try:
                                timing_cb.manual_epoch_end(logs={
                                    "loss": tr_loss,
                                    "sparse_categorical_accuracy": tr_acc,
                                    "val_loss": val_loss,
                                    "val_sparse_categorical_accuracy": val_acc,
                                })
                            except Exception:
                                # fallback log
                                logger and logger.info("Warmup epoch %d completed for model %s (train_loss=%.4f val_loss=%.4f)", epoch_idx + 1, model_key, tr_loss, val_loss)

                        # Unfreeze for phase2
                        try:
                            if chosen_base is not None and base_integration_success:
                                chosen_base.trainable = True
                                for layer in getattr(chosen_base, "layers", []):
                                    try:
                                        layer.trainable = True
                                    except Exception:
                                        pass
                                logger and logger.info("Unfroze chosen_base for phase 2 for model %s", model_key)
                            elif used_internal_backbone:
                                internal_conv1.trainable = True
                                internal_conv2.trainable = True
                                internal_conv3.trainable = True
                                logger and logger.info("Unfroze internal backbone conv layers for phase 2 for model %s", model_key)
                        except Exception:
                            logger and logger.debug("Failed unfreezing backbone layers for phase 2")

                        # Re-create or adjust optimizer learning rate for fine-tuning
                        try:
                            other_lower_lr = float(learning_rate * 0.1)
                            optimizer = _build_optimizer(optimizer_name, other_lower_lr)
                            try:
                                policy_name = tf.keras.mixed_precision.global_policy().name
                                if "mixed_float" in policy_name:
                                    from tensorflow.keras.mixed_precision import LossScaleOptimizer
                                    optimizer = LossScaleOptimizer(optimizer)
                            except Exception:
                                pass
                            model.optimizer = optimizer
                        except Exception:
                            pass

                        # Phase 2 manual fine-tune
                        history_phase2 = {"loss": [], "sparse_categorical_accuracy": [], "val_loss": [], "val_sparse_categorical_accuracy": []}
                        for epoch_idx in range(int(epochs)):
                            logger and logger.info("Manual loop: fine-tune epoch %d/%d", epoch_idx + 1, epochs)
                            tr_loss, tr_acc, tr_steps = _run_epoch_manual(train_ds, training=True)
                            val_loss, val_acc, val_steps = _run_epoch_manual(test_ds, training=False)
                            history_phase2["loss"].append(tr_loss)
                            history_phase2["sparse_categorical_accuracy"].append(tr_acc)
                            history_phase2["val_loss"].append(val_loss)
                            history_phase2["val_sparse_categorical_accuracy"].append(val_acc)
                            logger and logger.info(
                                "Fine-tune epoch %d metrics for model %s -> train_loss=%.6f | train_acc=%.6f | val_loss=%.6f | val_acc=%.6f",
                                epoch_idx + 1,
                                model_key,
                                tr_loss,
                                tr_acc,
                                val_loss,
                                val_acc,
                            )
                            logger and logger.info(
                                "Fine-tune epoch %d completed for model %s (train_loss=%.4f val_loss=%.4f)",
                                epoch_idx + 1, model_key, tr_loss, val_loss
                            )
                            try:
                                timing_cb.manual_epoch_end(logs={
                                    "loss": tr_loss,
                                    "sparse_categorical_accuracy": tr_acc,
                                    "val_loss": val_loss,
                                    "val_sparse_categorical_accuracy": val_acc,
                                })
                            except Exception:
                                logger and logger.info(
                                    "Fine-tune epoch %d completed for model %s (train_loss=%.4f val_loss=%.4f)",
                                    epoch_idx + 1, model_key, tr_loss, val_loss
                                )

                    # Merge histories (phase1 + phase2) into a single history structure
                    history = {}
                    if isinstance(history_phase1, dict) and isinstance(history_phase2, dict):
                        for k in set(list(history_phase1.keys()) + list(history_phase2.keys())):
                            a = history_phase1.get(k, []) if history_phase1 else []
                            b = history_phase2.get(k, []) if history_phase2 else []
                            history[k] = a + b
                    elif history_phase1 and history_phase2:
                        try:
                            a = history_phase1.history if hasattr(history_phase1, "history") else {}
                            b = history_phase2.history if hasattr(history_phase2, "history") else {}
                            for k in set(list(a.keys()) + list(b.keys())):
                                history[k] = list(a.get(k, [])) + list(b.get(k, []))
                        except Exception:
                            history = {}

                    # Evaluate/test and package results (attempt safe evaluation)
                    try:
                        eval_res = model.evaluate(test_ds, verbose=1)
                    except Exception:
                        try:
                            val_loss_metric = tf.keras.metrics.Mean()
                            val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
                            for batch in test_ds:
                                try:
                                    inputs_batch, labels_batch = batch
                                except Exception:
                                    try:
                                        inputs_batch, labels_batch = batch[0], batch[1]
                                    except Exception:
                                        continue
                                try:
                                    if isinstance(inputs_batch, (list, tuple)):
                                        inputs_tf = [tf.convert_to_tensor(x) for x in inputs_batch]
                                    else:
                                        inputs_tf = tf.convert_to_tensor(inputs_batch)
                                    labels_tf = tf.convert_to_tensor(labels_batch)
                                except Exception:
                                    inputs_tf = inputs_batch
                                    labels_tf = labels_batch
                                preds = model(inputs_tf, training=False)
                                v_loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tf, preds)
                                val_loss_metric.update_state(tf.reduce_mean(v_loss))
                                val_acc_metric.update_state(labels_tf, preds)
                            eval_res = [float(val_loss_metric.result().numpy()), float(val_acc_metric.result().numpy())]
                        except Exception:
                            eval_res = [None]

                    # Build predictions and detailed report for MLflow metrics/artifacts
                    y_true = []
                    y_pred = []
                    try:
                        for batch in test_ds:
                            try:
                                inputs_batch, labels_batch = batch
                            except Exception:
                                try:
                                    inputs_batch, labels_batch = batch[0], batch[1]
                                except Exception:
                                    continue

                            try:
                                if isinstance(inputs_batch, dict):
                                    preds = model(inputs_batch, training=False)
                                elif isinstance(inputs_batch, (list, tuple)):
                                    preds = model(list(inputs_batch), training=False)
                                else:
                                    preds = model(inputs_batch, training=False)
                            except Exception:
                                try:
                                    preds = model.predict(inputs_batch, verbose=0)
                                except Exception:
                                    preds = model.predict(list(inputs_batch), verbose=0) if isinstance(inputs_batch, (list, tuple)) else model.predict(inputs_batch, verbose=0)

                            try:
                                preds_np = preds.numpy() if hasattr(preds, "numpy") else np.asarray(preds)
                                pred_ints = np.argmax(preds_np, axis=1)
                            except Exception:
                                preds_np = np.asarray(preds)
                                pred_ints = np.argmax(preds_np, axis=1)

                            y_pred.extend(pred_ints.tolist())
                            try:
                                y_true.extend([int(x) for x in labels_batch.numpy().tolist()])
                            except Exception:
                                y_true.extend([int(x) for x in np.asarray(labels_batch).tolist()])
                    except Exception:
                        logger and logger.exception("Failed generating predictions for detailed metrics on %s", model_key)

                    y_true = np.array(y_true, dtype=int) if len(y_true) else np.array([], dtype=int)
                    y_pred = np.array(y_pred, dtype=int) if len(y_pred) else np.array([], dtype=int)

                    try:
                        y_true_names = le.inverse_transform(y_true) if y_true.size else np.array([], dtype=object)
                        y_pred_names = le.inverse_transform(y_pred) if y_pred.size else np.array([], dtype=object)
                        label_order = list(le.classes_)
                    except Exception:
                        y_true_names = [str(int(x)) for x in y_true.tolist()]
                        y_pred_names = [str(int(x)) for x in y_pred.tolist()]
                        label_order = list(map(str, sorted(set(y_true_names + y_pred_names)))) if (y_true_names or y_pred_names) else []

                    try:
                        report = TrainModelService.compute_metrics_from_predictions(y_true_names, y_pred_names, label_order)
                    except Exception:
                        try:
                            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_order))))
                            precision, recall, f1, support = precision_recall_fscore_support(
                                y_true,
                                y_pred,
                                labels=list(range(len(label_order))),
                                zero_division=0,
                            )
                            per_class = {
                                (label_order[i] if i < len(label_order) else str(i)): {
                                    "precision": float(precision[i]) if i < len(precision) else 0.0,
                                    "recall": float(recall[i]) if i < len(recall) else 0.0,
                                    "f1-score": float(f1[i]) if i < len(f1) else 0.0,
                                    "support": int(support[i]) if i < len(support) else 0,
                                }
                                for i in range(len(cm))
                            }
                            report = {
                                "per_class": per_class,
                                "macro_avg": {
                                    "precision": float(np.mean(precision)) if precision.size else 0.0,
                                    "recall": float(np.mean(recall)) if recall.size else 0.0,
                                    "f1-score": float(np.mean(f1)) if f1.size else 0.0,
                                },
                                "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
                                "confusion_matrix": cm.tolist(),
                                "total_samples": int(len(y_true)),
                                "labels": label_order,
                            }
                        except Exception:
                            report = {
                                "per_class": {},
                                "macro_avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
                                "accuracy": 0.0,
                                "confusion_matrix": [],
                                "total_samples": int(len(y_true)),
                                "labels": label_order,
                            }

                    # Log training / test metrics to MLflow
                    try:
                        _log_history_metrics_to_mlflow(history, model_key)

                        if len(eval_res) >= 1 and eval_res[0] is not None:
                            mlflow.log_metric("test_loss", float(eval_res[0]))
                        if len(eval_res) >= 2 and eval_res[1] is not None:
                            mlflow.log_metric("test_sparse_categorical_accuracy", float(eval_res[1]))

                        mlflow.log_metric("test_accuracy", float(report.get("accuracy", 0.0)))
                        mlflow.log_metric("test_macro_precision", float(report.get("macro_avg", {}).get("precision", 0.0)))
                        mlflow.log_metric("test_macro_recall", float(report.get("macro_avg", {}).get("recall", 0.0)))
                        mlflow.log_metric("test_macro_f1", float(report.get("macro_avg", {}).get("f1-score", 0.0)))
                        mlflow.log_metric("test_total_samples", int(report.get("total_samples", 0)))

                        for cname, cmets in (report.get("per_class") or {}).items():
                            try:
                                safe_cname = str(cname).replace(" ", "_")
                                mlflow.log_metric(f"test_precision_{safe_cname}", float(cmets.get("precision", 0.0)))
                                mlflow.log_metric(f"test_recall_{safe_cname}", float(cmets.get("recall", 0.0)))
                                mlflow.log_metric(f"test_f1_{safe_cname}", float(cmets.get("f1-score", 0.0)))
                                mlflow.log_metric(f"test_support_{safe_cname}", int(cmets.get("support", 0)))
                            except Exception:
                                pass
                    except Exception:
                        logger and logger.exception("Failed logging MLflow metrics for %s", model_key)

                    # Reintegrate the artifact logic: confusion matrix image, confusion matrix CSV, per-class CSV, and model signature
                    cm_csv = os.path.join(out_dir, f"{model_key}_confusion_matrix.csv")
                    per_class_csv = os.path.join(out_dir, f"{model_key}_per_class_metrics.csv")
                    classes_meta = os.path.join(out_dir, f"{model_key}_class_names.txt")
                    metrics_csv = os.path.join(out_dir, f"{model_key}_metrics_per_epoch.csv")

                    try:
                        cm_arr = np.array(report.get("confusion_matrix", []), dtype=int)
                        with open(cm_csv, "w", newline="") as fh:
                            writer = _csv.writer(fh)
                            writer.writerow([""] + list(label_order))
                            for i, row in enumerate(cm_arr):
                                writer.writerow([label_order[i] if i < len(label_order) else str(i)] + list(row.tolist()))
                        try:
                            mlflow.log_artifact(cm_csv, artifact_path="confusion_matrix")
                        except Exception:
                            logger and logger.debug("Could not log confusion CSV to MLflow.")
                    except Exception:
                        logger and logger.exception("Failed writing/logging confusion CSV.")

                    try:
                        with open(per_class_csv, "w", newline="") as fh:
                            writer = _csv.writer(fh)
                            writer.writerow(["class_name", "precision", "recall", "f1", "support"])
                            for cname, cmets in (report.get("per_class") or {}).items():
                                writer.writerow([
                                    cname,
                                    float(cmets.get("precision", 0.0)),
                                    float(cmets.get("recall", 0.0)),
                                    float(cmets.get("f1-score", 0.0)),
                                    int(cmets.get("support", 0)),
                                ])
                        try:
                            mlflow.log_artifact(per_class_csv, artifact_path="per_class_metrics")
                        except Exception:
                            logger and logger.debug("Could not log per-class CSV to MLflow.")
                    except Exception:
                        logger and logger.exception("Failed writing/logging per-class CSV.")

                    try:
                        with open(classes_meta, "w") as fh:
                            fh.write("\n".join(label_order))
                        try:
                            mlflow.log_artifact(classes_meta, artifact_path="metadata")
                        except Exception:
                            pass
                    except Exception:
                        logger and logger.debug("Failed saving class names file.")

                    try:
                        cm_paths = TrainModelService.plot_confusion_matrix_and_log_pandas_sklearn(report=report, artifact_dir=out_dir)
                        logger and logger.info("Confusion matrix artifacts saved: %s", cm_paths)
                    except Exception:
                        logger and logger.exception("Failed plotting/logging confusion matrix via helper.")

                    signature = None
                    input_example = None

                    try:
                        mlflow.log_artifact(aux_meta_path, artifact_path="metadata")
                        mlflow.log_artifact(metrics_csv, artifact_path="metrics")

                        # Save architecture artifacts even if the model itself later fails to log.
                        try:
                            arch_paths = _save_model_architecture_artifacts(model, model_key, out_dir)
                            logger.info("Saved model architecture artifacts for %s: %s", model_key, arch_paths)
                        except Exception:
                            logger and logger.exception("Could not save model architecture artifacts for %s.", model_key)

                        log_result = _log_model_to_mlflow_resiliently(
                            model=model,
                            model_key=model_key,
                            out_dir=out_dir,
                        )
                        logger.info("Model artifact logging result for %s: %s", model_key, log_result)
                    except Exception:
                        logger and logger.exception("Failed to log artifacts or model for %s", model_key)

                    # minimal packaging into results
                    results[model_key] = {
                        "history": history,
                        "eval": eval_res,
                        "report": report,
                        "confusion_matrix": np.array(report.get("confusion_matrix", []), dtype=int),
                        "per_class_metrics": {
                            "class_names": label_order,
                            "precision": [float(report.get("per_class", {}).get(c, {}).get("precision", 0.0)) for c in label_order],
                            "recall": [float(report.get("per_class", {}).get(c, {}).get("recall", 0.0)) for c in label_order],
                            "f1": [float(report.get("per_class", {}).get(c, {}).get("f1-score", 0.0)) for c in label_order],
                            "support": [int(report.get("per_class", {}).get(c, {}).get("support", 0)) for c in label_order],
                        },
                    }

                # no more training epochs after this model; keep the existing cleanup logic
                try:
                    del model
                except Exception:
                    pass
                gc.collect()
                try:
                    tf.keras.backend.clear_session()
                    gc.collect()
                except Exception:
                    pass

        except Exception as e:
            logger and logger.exception("Error during training: %s", e)
            results = {"error": f"Error during training: {e}"}
        finally:
            try:
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception:
                pass

        return results