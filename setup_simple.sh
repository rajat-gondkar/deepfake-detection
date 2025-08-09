#!/bin/bash

# Deepfake Detection Setup Script for Ubuntu 22.04 with RTX 4070 Ti
# Direct system installation (no virtual environment)

echo "=============================================="
echo "Deepfake Detection Setup (System-wide)"
echo "Ubuntu 22.04 + NVIDIA RTX 4070 Ti"
echo "=============================================="

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3 python3-pip git wget curl build-essential

# Install NVIDIA drivers if not present
echo "Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt install -y nvidia-driver-535
    echo "⚠️  NVIDIA drivers installed. Please REBOOT and run this script again."
    exit 1
fi

echo "NVIDIA driver status:"
nvidia-smi

# Install CUDA if not present
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA 12.1..."
    
    # Download and install CUDA keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    rm cuda-keyring_1.0-1_all.deb
    
    # Install CUDA toolkit
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-1
    
    # Add CUDA to PATH (add to .bashrc for persistence)
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    echo "⚠️  CUDA installed. Please run 'source ~/.bashrc' or restart terminal."
    echo "Then run this script again to continue with Python packages."
    exit 1
fi

echo "CUDA version:"
nvcc --version

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages from requirements.txt
echo "Installing project dependencies..."
python3 -m pip install timm tqdm numpy pandas scikit-learn Pillow matplotlib seaborn

# Verify installation
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA not available - will use CPU')

# Test other imports
try:
    import timm
    print(f'✅ timm version: {timm.__version__}')
except:
    print('❌ timm import failed')

try:
    import tqdm
    print('✅ tqdm imported successfully')
except:
    print('❌ tqdm import failed')

try:
    import numpy as np
    print(f'✅ numpy version: {np.__version__}')
except:
    print('❌ numpy import failed')

try:
    import sklearn
    print(f'✅ sklearn version: {sklearn.__version__}')
except:
    print('❌ sklearn import failed')
"

echo ""
echo "=============================================="
echo "🎉 Setup completed successfully!"
echo "=============================================="
echo ""
echo "You can now start training with:"
echo "python3 train.py --data_dir ./dataset --epochs 15"
echo ""
echo "For help with training options:"
echo "python3 train.py --help"
echo ""
echo "For inference:"
echo "python3 inference.py --model_path ./checkpoints/best_model.pth --image_path test.jpg"
