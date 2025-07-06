#!/bin/bash

# SVELTE Framework - Jules Agent Optimized Setup
# Fallback setup script for Jules Agent VM environment

set -e

echo "=== Jules Agent - SVELTE Framework Setup ==="

# Check if we're in a container/VM environment
if [ -f /.dockerenv ] || [ -n "${CONTAINER_ID}" ]; then
    echo "Detected containerized environment"
    CONTAINERIZED=true
else
    echo "Standard environment detected"
    CONTAINERIZED=false
fi

# Python environment setup
echo "Setting up Python environment..."
if [ "$CONTAINERIZED" = true ]; then
    # Container environment - assume Python is available
    python3 --version || { echo "Python3 required but not found!"; exit 1; }
else
    # Standard environment
    if command -v python3 &> /dev/null; then
        echo "Python3 found: $(python3 --version)"
    else
        echo "Python3 not found - attempting installation..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y python3 python3-pip python3-venv
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3 python3-pip
        else
            echo "Cannot install Python3 automatically. Please install manually."
            exit 1
        fi
    fi
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install from requirements file if it exists, otherwise install directly
if [ -f "requirements-jules.txt" ]; then
    echo "Installing from requirements-jules.txt..."
    pip install -r requirements-jules.txt
else
    echo "Installing core dependencies directly..."
    pip install numpy scipy matplotlib networkx scikit-learn pandas tqdm
    pip install pytest pytest-cov black flake8 mypy
fi

# Set up environment
echo "Configuring SVELTE environment..."
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"
mkdir -p models output logs visualizations .cache

# Create simple test
echo "Creating environment test..."
python3 -c "
import numpy as np
import scipy
import matplotlib
print('✓ Core dependencies verified')
print('✓ SVELTE environment ready for Jules')
"

echo "=== Jules Setup Complete ==="
echo "Environment is ready for Jules Agent to continue implementation."
