#!/bin/bash

# Deepfake Detection Setup Script for Ubuntu 22.04 with RTX 4070 Ti
# This script sets up the environment and installs all required dependencies

echo "=============================================="
echo "Deepfake Detection Environment Setup"
echo "Ubuntu 22.04 + NVIDIA RTX 4070 Ti"
echo "=============================================="

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git wget curl

# Install NVIDIA drivers and CUDA (if not already installed)
echo "Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt install -y nvidia-driver-535
    echo "Please reboot the system after installation and run this script again."
    exit 1
fi

# Check CUDA installation
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA 12.1..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-1
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv deepfake_env
source deepfake_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other required packages
echo "Installing additional packages..."
pip install timm tqdm numpy pandas scikit-learn Pillow matplotlib seaborn

# Verify installation
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('CUDA not available - using CPU')
"

echo "=============================================="
echo "Setup completed successfully!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "source deepfake_env/bin/activate"
echo ""
echo "Then you can start training with:"
echo "python train.py --data_dir ./dataset --epochs 15"
echo ""
echo "For help with training options:"
echo "python train.py --help"
