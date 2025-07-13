#!/bin/bash

# ARDS Project Virtual Environment Setup Script
# This script creates a virtual environment and sets it up as a Jupyter kernel

echo "=== ARDS Project Environment Setup ==="
echo

# Set environment name
ENV_NAME="ards-env"
KERNEL_NAME="ards-analysis"
KERNEL_DISPLAY_NAME="ARDS Analysis (Python)"

# Check if virtual environment already exists
if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and create a new one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf "$ENV_NAME"
    else
        echo "Exiting without changes."
        exit 1
    fi
fi

# Create virtual environment
echo "Creating virtual environment: $ENV_NAME"
python3 -m venv "$ENV_NAME"

# Activate virtual environment
echo "Activating virtual environment..."
source "$ENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi

# Install ipykernel if not in requirements
pip install ipykernel

# Add virtual environment as Jupyter kernel
echo "Adding Jupyter kernel: $KERNEL_DISPLAY_NAME"
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$KERNEL_DISPLAY_NAME"

# Create activation reminder
echo
echo "=== Setup Complete ==="
echo
echo "To activate this environment in the future, run:"
echo "    source $ENV_NAME/bin/activate"
echo
echo "To use in Jupyter:"
echo "1. Start Jupyter: jupyter notebook or jupyter lab"
echo "2. Select kernel: '$KERNEL_DISPLAY_NAME'"
echo
echo "To remove the kernel later, run:"
echo "    jupyter kernelspec uninstall $KERNEL_NAME"
echo

# Show installed packages
echo "Key packages installed:"
pip list | grep -E "pandas|numpy|scikit-learn|matplotlib|jupyter"