#!/bin/bash

# Export envs from .env if needed
export $(grep -v '^#' /app/.env | xargs || true)

OUTPUT_SUBFOLDER="${OUTPUT_FOLDER_NAME_FOR_DATA_PROCESSED_BY_SPARK:-Placeholder_output_folder}"

# Debug logs
echo "Running as user: $(id)"

exec "$@"
