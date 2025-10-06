#!/bin/bash

# Add the root directory to the PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/applications/data_processing_api/src

# activate the virtual environment
VENV_NAME="captaima_ml_venv"
source "./$VENV_NAME/bin/activate"
source "./$VENV_NAME/Scripts/activate"

# Run the API
python applications/data_processing_api/run.py