#!/bin/bash
# Setup script for Gemini reproduction project

set -e

echo "========================================="
echo "Gemini Project Environment Setup"
echo "========================================="

# Check Python version
echo -e "\n[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo -e "\n[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n[3/5] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo -e "\n[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo -e "\n[5/5] Creating project directories..."
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p results

# Copy config templates
if [ ! -f "configs/training_config.yaml" ]; then
    cp configs/training_config.yaml.template configs/training_config.yaml
    echo "Created configs/training_config.yaml"
fi

if [ ! -f "configs/cluster_config.yaml" ]; then
    cp configs/cluster_config.yaml.template configs/cluster_config.yaml
    echo "Created configs/cluster_config.yaml"
fi

echo -e "\n========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Edit configs/cluster_config.yaml with your cluster details"
echo "3. Edit configs/training_config.yaml with your training settings"
echo ""

