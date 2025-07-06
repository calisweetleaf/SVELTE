#!/bin/sh

# Minimal SVELTE Framework Setup for Jules Agent VM
# Optimized for Jules' containerized environment

set -e

echo "=== SVELTE Framework - Jules Agent Setup ==="

# Verify Python is available
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python3 not found in Jules VM environment"
    exit 1
fi

echo "Python found: $(python3 --version)"

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv

# Activate virtual environment (POSIX compliant)
echo "Activating virtual environment..."
. .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install core scientific dependencies
echo "Installing SVELTE dependencies..."
pip install \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.5.0 \
    networkx>=2.6.0 \
    scikit-learn>=1.0.0 \
    pandas>=1.3.0 \
    tqdm>=4.62.0 \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0

# Set up Python path
export PYTHONPATH="$(pwd):$(pwd)/src"

# Create necessary directories
mkdir -p models output logs visualizations .cache

# Verify installation
echo "Verifying installation..."
python -c "
import numpy, scipy, matplotlib, networkx, sklearn, pandas, pytest
print('✓ All core dependencies installed successfully')
print('✓ SVELTE Framework ready for Jules')
"

echo "=== Jules Setup Complete ==="
echo "Virtual environment: .venv"
echo "Python path configured for SVELTE modules"
