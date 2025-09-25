#!/bin/bash

# substituir 1001:1001 se o uid/gid do spark for outro
sudo mkdir -p ./shared/utils/processed_output ./shared/utils/datasets
sudo chown -R 1001:1001 ./shared/utils/processed_output ./shared/utils/datasets
sudo chmod -R 0777 ./shared/utils/processed_output ./shared/utils/datasets

# Add the root directory to the PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/applications/data_processing_api/src

# activate the virtual environment
VENV_NAME="captaima_ml_venv"
source "./$VENV_NAME/bin/activate"
source "./$VENV_NAME/Scripts/activate"

# Run the API
python applications/data_processing_api/run.py