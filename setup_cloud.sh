#!/bin/bash
# FEVER Baseline - Cloud Setup Script
# Run this script on your cloud instance after uploading the code

set -e  # Exit on error

echo "========================================="
echo "FEVER Baseline - Cloud Setup"
echo "========================================="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "[WARNING] This script is designed for Linux. You may need to adapt it for your OS."
fi

# Create virtual environment
echo "[1/6] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "[2/6] Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "[3/6] Installing requirements..."
pip install -r requirements.txt

# Verify PyTorch and GPU
echo "[4/6] Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo "[5/6] Checking GPU availability..."
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# Check disk space
echo "[6/6] Checking disk space..."
df -h . | grep -v Filesystem

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Make sure DOWNLOAD_WIKIPEDIA=True in config.py"
echo "  2. Run: python run.py all"
echo "  3. Monitor: python monitor_training.py (in separate terminal)"
echo ""
echo "Recommended: Run in screen session to avoid disconnection:"
echo "  screen -S fever"
echo "  python run.py all"
echo "  # Press Ctrl+A then D to detach"
echo "  # Reconnect with: screen -r fever"
echo ""
