#!/usr/bin/env python3
"""
SVELTE Framework Environment Setup Script for Jules
This script configures the Python environment for the SVELTE Framework
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and handle errors"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {cmd}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return None

def main():
    """Main setup function"""
    print("Setting up SVELTE Framework environment...")
    
    # Install requirements
    print("\nInstalling Python dependencies...")
    run_command("pip install -r requirements.txt")
    
    # Create necessary directories
    print("\nCreating necessary directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{os.getcwd()}/src"
    os.environ["SVELTE_LOG_LEVEL"] = "INFO"
    os.environ["SVELTE_OUTPUT_DIR"] = "output"
    os.environ["SVELTE_CACHE_DIR"] = ".cache"
    
    print("\nVerifying installations...")
    imports = [
        "numpy",
        "scipy", 
        "matplotlib",
        "networkx",
        "sklearn",
        "pandas",
        "pytest"
    ]
    
    for module in imports:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
    
    print("\nEnvironment setup complete!")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

if __name__ == "__main__":
    main()
