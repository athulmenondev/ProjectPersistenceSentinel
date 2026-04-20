#!/bin/bash
# PSCDL 2026 - Setup Script
# Creates virtual environment and installs dependencies

set -e

echo "======================================"
echo "PSCDL 2026 - Setup"
echo "======================================"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python main.py --help"
echo "======================================"
