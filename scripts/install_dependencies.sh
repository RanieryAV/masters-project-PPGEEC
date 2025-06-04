#!/bin/bash

## Install and configure Python
PYTHON_VERSION=$(cat .python-version)
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Verify if pyenv is installed
if ! command -v pyenv &> /dev/null; then
  echo "pyenv not installed."
fi

echo "Installing Python $PYTHON_VERSION..."
pyenv install -s $PYTHON_VERSION

echo "Defining Python $PYTHON_VERSION as local version..."
pyenv local $PYTHON_VERSION


## Install and configure node
NODE_VERSION=$(cat .nvmrc)
export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
  [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Verify if nvm is installed
if ! command -v nvm &> /dev/null; then
  echo "nvm not installed."
fi

echo "Installing Node $NODE_VERSION..."
nvm install $NODE_VERSION

echo "Defining Node $NODE_VERSION as local version..."
nvm use $NODE_VERSION

## Exporting JAVA_HOME and PATH environment variables (related to OpenJDK 8 requisite)
echo "Exporting JAVA_HOME environment variable and the PATH..."

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"
echo "JAVA_HOME set to $JAVA_HOME"

# Verify if java is installed
if ! command -v java &> /dev/null; then
  echo "Java installation failed."
  exit 1
fi

## Install the required packages
# Add the root directory to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create and activate a virtual environment using pyenv
VENV_NAME="captaima_ml_venv"

# Check if the virtual environment already exists
if [ ! -d "./$VENV_NAME" ]; then
  echo "Creating virtual environment $VENV_NAME..."
  PYENV_VERSION=$PYTHON_VERSION python -m venv "./$VENV_NAME"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "./$VENV_NAME/bin/activate"
if [ "$?" -ne 0 ]; then
  echo "Failed to activate virtual environment"
  exit 1
fi

# Ensure pip is up to date
pip install --upgrade pip

# Install the required Python packages
echo "Installing the required Python packages"
pip install -r requirements.txt
if [ "$?" -ne 0 ]; then
  echo "Failed to install Python packages"
  #exit 1
fi

# Verify installation of Flask
if ! python -c "import flask" &> /dev/null; then
  echo "Flask module not found. Installation failed."
  #exit 1
fi

# Install the required node packages
echo "Installing the required node packages"
npm install