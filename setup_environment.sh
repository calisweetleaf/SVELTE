#!/bin/sh

# This script configures the Python environment for the SVELTE Framework

set -e  # Exit on any error

echo "Setting up SVELTE Framework environment..."

# Check if running as root or with sudo access
if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root, updating system packages..."
    apt-get update -qq
    apt-get install -y python3 python3-pip python3-venv python3-dev build-essential
elif command -v sudo >/dev/null 2>&1; then
    echo "Using sudo for system package installation..."
    sudo apt-get update -qq
    sudo apt-get install -y python3 python3-pip python3-venv python3-dev build-essential
else
    echo "No sudo access, assuming Python is already installed..."
fi

# Check Python version and create virtual environment
echo "Checking Python installation..."
python3 --version || { echo "Python3 not found!"; exit 1; }

# Create virtual environment in the project directory
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment (POSIX-compliant)
echo "Activating virtual environment..."
. venv/bin/activate

# Upgrade pip to latest version
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core scientific computing dependencies
echo "Installing core scientific dependencies..."
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.5.0

# Install additional scientific libraries
echo "Installing additional scientific libraries..."
pip install networkx>=2.6.0 scikit-learn>=1.0.0 pandas>=1.3.0 tqdm>=4.62.0

# Install testing framework
echo "Installing testing dependencies..."
pip install pytest>=7.0.0 pytest-cov>=4.0.0 pytest-xdist>=2.5.0

# Install development tools
echo "Installing development tools..."
pip install black>=22.0.0 flake8>=4.0.0 mypy>=0.910 isort>=5.10.0

# Configure Python path for SVELTE framework
echo "Configuring Python path..."
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

# Create necessary directories if they don't exist
echo "Creating project directories..."
mkdir -p models output logs visualizations .cache

# Set SVELTE-specific environment variables
echo "Setting environment variables..."
export SVELTE_LOG_LEVEL=INFO
export SVELTE_OUTPUT_DIR=output
export SVELTE_CACHE_DIR=.cache
export SVELTE_MODELS_DIR=models
export SVELTE_VISUALIZATIONS_DIR=visualizations

# Create a simple environment activation script for convenience
echo "Creating environment activation script..."
cat > activate_svelte_env.sh << 'EOF'
#!/bin/sh
# SVELTE Environment Activation Script - POSIX compliant
. venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"
export SVELTE_LOG_LEVEL=INFO
export SVELTE_OUTPUT_DIR=output
export SVELTE_CACHE_DIR=.cache
export SVELTE_MODELS_DIR=models
export SVELTE_VISUALIZATIONS_DIR=visualizations
echo "SVELTE environment activated!"
EOF

chmod +x activate_svelte_env.sh

echo "Environment setup complete!"
echo "Virtual environment: $(which python)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Python path: ${PYTHONPATH}"

# Comprehensive dependency verification
echo "Verifying core dependencies..."
python -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print('\\nVerifying imports:')

try:
    import numpy as np
    print(f'✓ NumPy {np.__version__}')
except ImportError as e:
    print(f'✗ NumPy: {e}')

try:
    import scipy
    print(f'✓ SciPy {scipy.__version__}')
except ImportError as e:
    print(f'✗ SciPy: {e}')

try:
    import matplotlib
    print(f'✓ Matplotlib {matplotlib.__version__}')
except ImportError as e:
    print(f'✗ Matplotlib: {e}')

try:
    import networkx as nx
    print(f'✓ NetworkX {nx.__version__}')
except ImportError as e:
    print(f'✗ NetworkX: {e}')

try:
    import sklearn
    print(f'✓ Scikit-learn {sklearn.__version__}')
except ImportError as e:
    print(f'✗ Scikit-learn: {e}')

try:
    import pandas as pd
    print(f'✓ Pandas {pd.__version__}')
except ImportError as e:
    print(f'✗ Pandas: {e}')

try:
    import pytest
    print(f'✓ Pytest {pytest.__version__}')
except ImportError as e:
    print(f'✗ Pytest: {e}')

print('\\nDependency verification complete!')
"

# Test basic SVELTE module imports
echo "Testing SVELTE module structure..."
python -c "
import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

try:
    from src.metadata import metadata_extractor
    print('✓ Metadata extractor import successful')
except ImportError as e:
    print(f'! Metadata extractor: {e}')

try:
    from src.tensor_analysis import gguf_parser
    print('✓ GGUF parser import successful')
except ImportError as e:
    print(f'! GGUF parser: {e}')

try:
    from src.model_architecture import attention_topology
    print('✓ Attention topology import successful')
except ImportError as e:
    print(f'! Attention topology: {e}')

print('\\nSVELTE module structure verification complete!')
"

echo ""
echo "=== SVELTE Framework Environment Setup Complete ==="
echo "To activate this environment in the future, run:"
echo "  . activate_svelte_env.sh"
echo ""
echo "Remember to run this script in the SVELTE project directory."
echo ""
echo "For more information, refer to the SVELTE documentation."
echo "=== Setup Script Finished Successfully ==="
