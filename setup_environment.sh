#!/bin/bash

# SVELTE Framework Environment Setup Script for Jules
# This script configures the Python environment for the SVELTE Framework

set -e  # Exit on any error

echo "Setting up SVELTE Framework environment..."

# Update system packages
echo "Updating system packages..."
apt-get update -qq

# Install Python and pip if not present
echo "Installing Python and pip..."
apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0
pip install networkx>=2.6.0
pip install scikit-learn>=1.0.0
pip install pandas>=1.3.0
pip install tqdm>=4.62.0

# Install testing dependencies
echo "Installing testing dependencies..."
pip install pytest>=7.0.0
pip install pytest-cov>=4.0.0

# Install development dependencies
echo "Installing development dependencies..."
pip install black>=22.0.0
pip install flake8>=4.0.0
pip install mypy>=0.910

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models
mkdir -p output
mkdir -p logs
mkdir -p visualizations

# Set environment variables
export SVELTE_LOG_LEVEL=INFO
export SVELTE_OUTPUT_DIR=output
export SVELTE_CACHE_DIR=.cache

echo "Environment setup complete!"
echo "Virtual environment activated"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Verify installations
echo "Verifying installations..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy version: {scipy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"
python -c "import networkx; print(f'NetworkX version: {networkx.__version__}')"

echo "All dependencies installed successfully!"
