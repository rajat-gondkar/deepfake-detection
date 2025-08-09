# Manual Setup Guide for Ubuntu 22.04 + RTX 4070 Ti

## 🚀 Quick Setup (No Virtual Environment)

### Step 1: System Update
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip git wget curl build-essential
```

### Step 2: NVIDIA Drivers
Check if drivers are installed:
```bash
nvidia-smi
```

If not installed:
```bash
sudo apt install -y nvidia-driver-535
sudo reboot  # Required after driver installation
```

### Step 3: CUDA Installation
Check CUDA:
```bash
nvcc --version
```

If not installed:
```bash
# Download CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Install CUDA 12.1
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Python Packages
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
python3 -m pip install timm tqdm numpy pandas scikit-learn Pillow matplotlib seaborn
```

### Step 5: Verify Installation
```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')
"
```

### Step 6: Start Training
```bash
# Navigate to project directory
cd /path/to/your/deepfake-detection/

# Start training
python3 train.py --data_dir ./dataset --epochs 15
```

## 🔧 Alternative: One-Command Setup

Download and run the automated setup script:
```bash
chmod +x setup_simple.sh
./setup_simple.sh
```

This script will:
1. Update system packages
2. Install NVIDIA drivers (if needed)
3. Install CUDA 12.1 (if needed)
4. Install all Python dependencies
5. Verify the installation

## 📋 Package Versions

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| torchvision | >=0.15.0 | Computer vision utilities |
| timm | >=0.9.0 | EfficientNet-B4 model |
| tqdm | >=4.65.0 | Progress bars |
| numpy | >=1.24.0 | Numerical computing |
| scikit-learn | >=1.3.0 | Metrics and evaluation |
| Pillow | >=10.0.0 | Image processing |
| matplotlib | >=3.7.0 | Plotting |
| seaborn | >=0.12.0 | Advanced plotting |

## 🎯 Training Commands

### Basic Training
```bash
python3 train.py --data_dir ./dataset
```

### Advanced Training
```bash
python3 train.py \
    --data_dir ./dataset \
    --epochs 15 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_dir ./checkpoints \
    --early_stopping \
    --patience 7 \
    --freeze_backbone \
    --unfreeze_epoch 5
```

### Monitor GPU Usage
```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi
```

## 🔍 Inference Commands

### Single Image
```bash
python3 inference.py \
    --model_path ./checkpoints/best_model.pth \
    --image_path /path/to/image.jpg
```

### Batch Processing
```bash
python3 inference.py \
    --model_path ./checkpoints/best_model.pth \
    --directory_path /path/to/images/ \
    --output_file results.csv
```

## 🛠️ Troubleshooting

### CUDA Issues
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
Reduce batch size in training:
```bash
python3 train.py --data_dir ./dataset --batch_size 16  # or 8
```

### Permission Issues
If pip installation fails:
```bash
python3 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install --user timm tqdm numpy pandas scikit-learn Pillow matplotlib seaborn
```

## 📁 Dataset Structure Verification
```bash
# Check dataset structure
tree dataset/ -d

# Should show:
# dataset/
# ├── train/
# │   ├── deepfake/
# │   └── real/
# ├── val/
# │   ├── deepfake/
# │   └── real/
# └── test/
#     ├── deepfake/
#     └── real/
```

## 🚀 Ready to Train!

Once setup is complete, you can start training immediately:
```bash
python3 train.py --data_dir ./dataset --epochs 15 --early_stopping
```

The training will show real-time progress and save the best model automatically!
