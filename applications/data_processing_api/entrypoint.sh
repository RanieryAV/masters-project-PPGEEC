#!/bin/bash

## Guarantee datasets read/write
#chmod -R 777 /app/datasets

## Create processed_output root if missing
#mkdir -p /app/processed_output
#chmod -R 777 /app/processed_output
#chown -R spark:spark /app/processed_output

# Export envs from .env if needed
export $(grep -v '^#' /app/.env | xargs || true)

OUTPUT_SUBFOLDER="${OUTPUT_FOLDER_NAME_FOR_DATA_PROCESSED_BY_SPARK:-Placeholder_output_folder}"

# Create the job subfolder
#mkdir -p "/app/processed_output/${OUTPUT_SUBFOLDER}"
#chmod -R 777 "/app/processed_output/${OUTPUT_SUBFOLDER}"
#chown -R spark:spark "/app/processed_output/${OUTPUT_SUBFOLDER}"

# Prepare tmp dir
#mkdir -p /app/processed_output/spark_tmp
#chmod -R 777 /app/processed_output/spark_tmp
#chown -R spark:spark /app/processed_output/spark_tmp

# Debug logs
echo "Running as user: $(id)"
#ls -ld /app/processed_output
#ls -ld "/app/processed_output/${OUTPUT_SUBFOLDER}"
#ls -ld /app/processed_output/spark_tmp

exec "$@"
