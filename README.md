# Deepfake Detection Training Project

This project implements a deep learning model using **EfficientNet-B4** architecture to classify images as either **Real** or **Deepfake**. The model is optimized for training on **NVIDIA RTX 4070 Ti GPU** with **Ubuntu 22.04**.

## 🚀 Quick Start

### 1. Setup Environment (Ubuntu 22.04)

**Option A: Automated Setup**
```bash
# Make setup script executable
chmod +x setup_simple.sh

# Run setup script (installs CUDA, PyTorch, and dependencies)
./setup_simple.sh
```

**Option B: Manual Setup**
See `SETUP_MANUAL.md` for detailed step-by-step instructions.

### 2. Verify Dataset Structure

Ensure your dataset follows this structure:
```
dataset/
├── train/
│   ├── real/          # Real images
│   └── deepfake/      # Deepfake images
├── val/
│   ├── real/          # Validation real images
│   └── deepfake/      # Validation deepfake images
└── test/
    ├── real/          # Test real images
    └── deepfake/      # Test deepfake images
```

### 3. Start Training

```bash
# Basic training with default parameters
python3 train.py --data_dir ./dataset

# Training with custom parameters
python3 train.py \
    --data_dir ./dataset \
    --epochs 15 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_dir ./checkpoints \
    --early_stopping \
    --patience 7
```

### 4. Monitor Training

The training script provides:
- **Real-time progress bars** with loss and accuracy
- **Live metrics display** for each epoch
- **Automatic best model saving**
- **Training plots** generation
- **Comprehensive logging**

## 📊 Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Architecture | EfficientNet-B4 | Pre-trained on ImageNet |
| Input Size | 380×380 | Optimal for EfficientNet-B4 |
| Batch Size | 32 | Optimized for RTX 4070 Ti |
| Learning Rate | 1e-4 | With AdamW optimizer |
| Epochs | 15 | With early stopping |
| Mixed Precision | Enabled | For faster training |

## 🔧 Training Options

```bash
python3 train.py --help
```

**Key Arguments:**
- `--data_dir`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 15)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--freeze_backbone`: Freeze backbone initially for transfer learning
- `--unfreeze_epoch`: Epoch to unfreeze backbone (default: 5)
- `--early_stopping`: Enable early stopping
- `--patience`: Early stopping patience (default: 7)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)

## 🔍 Making Predictions

### Single Image Prediction
```bash
python3 inference.py \
    --model_path ./checkpoints/best_model.pth \
    --image_path /path/to/image.jpg
```

### Batch Prediction (Directory)
```bash
python3 inference.py \
    --model_path ./checkpoints/best_model.pth \
    --directory_path /path/to/images/ \
    --output_file results.csv
```

## 📁 Project Structure

```
deepfake-detection/
├── train.py              # Main training script
├── inference.py          # Inference script
├── model.py              # EfficientNet-B4 model definition
├── data_utils.py         # Data loading and preprocessing
├── training_utils.py     # Training utilities and metrics
├── requirements.txt      # Python dependencies
├── setup_simple.sh       # Simple setup script (no virtual env)
├── SETUP_MANUAL.md       # Detailed manual setup guide
├── helper.md             # Original specifications
└── dataset/              # Your dataset directory
    ├── train/
    ├── val/
    └── test/
```

## 📈 Training Output

The training script generates:

1. **Checkpoints/** directory containing:
   - `best_model.pth` - Best model based on validation accuracy
   - `final_model.pth` - Final model after training
   - `checkpoint_epoch_X.pth` - Periodic checkpoints
   - `training_metrics.png` - Training/validation curves
   - `test_evaluation.png` - Test set confusion matrix
   - `test_results.txt` - Final test metrics

2. **Console Output:**
   - Real-time progress bars with `tqdm`
   - Epoch summaries with train/val metrics
   - Learning rate updates
   - Best model notifications
   - Final evaluation results

## 🎯 Expected Performance

With the provided configuration:
- **Training Time**: ~6-8 hours for 15 epochs (140k images)
- **GPU Memory Usage**: ~8-10 GB (RTX 4070 Ti has 12 GB)
- **Expected Accuracy**: 90-95% on validation set
- **Batch Processing**: ~100-120 images/second

## 🔧 Troubleshooting

### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Reduce `batch_size` from 32 to 16 or 8
- Reduce `num_workers` if CPU bottleneck
- Enable `torch.cuda.empty_cache()` in training loop

### Dataset Issues
- Verify folder structure matches expected format
- Check image file extensions (.jpg, .png)
- Ensure sufficient disk space for checkpoints

## 📚 Additional Features

- **Mixed Precision Training**: Automatic optimization for RTX GPUs
- **Class Weight Balancing**: Automatic handling of imbalanced datasets
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Learning Rate Scheduling**: Cosine annealing scheduler
- **Early Stopping**: Prevent overfitting with patience
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Training curves and confusion matrices

## 🔄 Resume Training

```bash
# Resume from checkpoint
python3 train.py \
    --data_dir ./dataset \
    --resume ./checkpoints/checkpoint_epoch_10.pth
```

## 💡 Tips for Best Results

1. **Data Quality**: Ensure high-quality, diverse images
2. **Preprocessing**: Images are automatically resized and normalized
3. **Transfer Learning**: Start with frozen backbone, then fine-tune
4. **Monitoring**: Watch for overfitting using validation metrics
5. **Patience**: Allow early stopping to prevent overfitting
6. **Hardware**: Ensure adequate cooling for sustained training

---

**Happy Training! 🚀**

For detailed specifications, refer to `helper.md`.
