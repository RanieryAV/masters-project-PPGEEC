#!/bin/bash

# Add the root directory to the PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/applications/model_training_api/src

# activate the virtual environment
VENV_NAME="captaima_ml_venv"
source "./$VENV_NAME/bin/activate"
source "./$VENV_NAME/Scripts/activate"

# Run the API
python applications/model_training_api/run.py