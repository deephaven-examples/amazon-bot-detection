#!/bin/bash

# set the name of the virtual environment
ENV_NAME="dh-amazon-venv"

# create the virtual environment
python3 -m venv $ENV_NAME

# check if the virtual environment was created successfully
if [ -d "$ENV_NAME" ]; then
  echo "Virtual environment '$ENV_NAME' created successfully."
else
  echo "Failed to create virtual environment '$ENV_NAME'."
  exit 1
fi

# activate the virtual environment
source $ENV_NAME/bin/activate

# check if activation was successful
if [ "$VIRTUAL_ENV" != "" ]; then
  echo "Virtual environment '$ENV_NAME' activated."
else
  echo "Failed to activate virtual environment '$ENV_NAME'."
  exit 1
fi

# install packages from the first requirements file
if [ -f ../requirements.txt ]; then
  pip install -r ../requirements.txt
else
  echo "Requirements file not found."
  deactivate
  exit 1
fi

# install packages from the second requirements file
if [ -f jupyter_requirements.txt ]; then
  pip install -r jupyter_requirements.txt
else
  echo "Requirements file not found."
  deactivate
  exit 1
fi

echo "Packages installed successfully."

# install ipykernel and register the virtual environment as a Jupyter kernel
python3 -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"
echo "Virtual environment '$ENV_NAME' registered as a Jupyter kernel."

# deactivate the virtual environment
deactivate
echo "Virtual environment '$ENV_NAME' deactivated."
