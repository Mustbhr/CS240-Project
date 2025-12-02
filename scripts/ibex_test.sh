#!/bin/bash
#SBATCH --job-name=gemini-test
#SBATCH --output=logs/gemini_test_%j.log
#SBATCH --error=logs/gemini_test_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# ============================================
# GEMINI PROJECT - IBEX TEST SCRIPT
# ============================================
# This script tests the project infrastructure on IBEX.
# 
# Usage:
#   1. Interactive: srun --nodes=1 --gpus-per-node=1 --time=00:30:00 --pty bash
#      Then run: bash scripts/ibex_test.sh
#   
#   2. Batch job: sbatch scripts/ibex_test.sh
#
# ============================================

echo "============================================"
echo "GEMINI PROJECT - IBEX TEST"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "User: $USER"
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR 2>/dev/null || cd /path/to/CS240-Project

# Create logs directory
mkdir -p logs checkpoints

# Load modules (adjust based on IBEX's available modules)
echo "[1/6] Loading modules..."
module purge
module load cuda/11.8  # or cuda/12.x
module load python/3.10  # or available python version
# module load pytorch  # if available as module

echo "Loaded modules:"
module list

# Check Python
echo ""
echo "[2/6] Checking Python environment..."
which python
python --version

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo ""
    echo "[3/6] Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[4/6] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Check GPU
echo ""
echo "[5/6] Checking GPU..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run tests
echo ""
echo "[6/6] Running infrastructure tests..."
echo "============================================"

python scripts/run_baseline_test.py

echo ""
echo "============================================"
echo "TEST COMPLETE"
echo "============================================"
echo "End time: $(date)"

