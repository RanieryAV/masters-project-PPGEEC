#!/bin/bash
set -e

# --- Pre-entrypoint customization ---

# Ensure that the mount points are writable by user spark
# (you pode ajustar chown/chmod conforme seus diretórios)

# Export envs from .env if needed
export $(grep -v '^#' /app/.env | xargs || true)

chmod -R 777 /app/datasets
chown -R spark:spark /app/datasets || true

chmod -R 777 /app/processed_output
chown -R spark:spark /app/processed_output || true

# If environment variable defines a subfolder
# Data processing specific env vars
PROCESSED_OUTPUT_DIR=${PROCESSED_OUTPUT_DIR}
OUTPUT_FOLDER_NAME_FOR_LABELS_DATA_SPARK_PITSIKALIS_2019=${OUTPUT_FOLDER_NAME_FOR_LABELS_DATA_SPARK_PITSIKALIS_2019}
OUTPUT_FOLDER_NAME_FOR_TRANSSHIP_AIS_DATA_SPARK_PITSIKALIS_2019=${OUTPUT_FOLDER_NAME_FOR_TRANSSHIP_AIS_DATA_SPARK_PITSIKALIS_2019}
mkdir -p "/app/processed_output/${OUTPUT_SUBFOLDER}"
chmod -R 777 "/app/processed_output/${OUTPUT_SUBFOLDER}"
chown -R spark:spark "/app/processed_output/${OUTPUT_SUBFOLDER}" || true

# Spark local tmp dir
mkdir -p /app/processed_output/spark_tmp
chmod -R 777 /app/processed_output/spark_tmp
chown -R spark:spark /app/processed_output/spark_tmp || true

# Debug logs
echo "[wrapper] Running as $(id)"
echo "[wrapper] Permissions of /app/processed_output:"
ls -ld /app/processed_output
echo "[wrapper] Permissions of subfolder ${OUTPUT_SUBFOLDER}:"
ls -ld "/app/processed_output/${OUTPUT_SUBFOLDER}"
echo "[wrapper] Permissions of spark_tmp:"
ls -ld /app/processed_output/spark_tmp

# --- Delegate to the original Bitnami Spark entrypoint ---
exec /opt/bitnami/scripts/spark/entrypoint.sh "$@"
