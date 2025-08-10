# Deepfake Detection with EfficientNet-B4

This project implements a deep learning model using **EfficientNet-B4** architecture to classify images as either **Real** or **Deepfake**. The model includes face detection preprocessing and multi-face detection capabilities.

## 🚀 Quick Setup

### Install Dependencies
```bash
pip install torch torchvision timm pillow numpy tqdm scikit-learn matplotlib seaborn facenet-pytorch opencv-python streamlit
```

## 📂 Dataset Structure

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

## 🔧 Training the Model

### Option 1: Train on Original Images
```bash
python train.py --data_dir ./dataset --epochs 50 --batch_size 32
```

### Option 2: Train with Face Preprocessing (Recommended)
```bash
# Step 1: Extract faces from dataset
python preprocess_faces.py --input_dir ./dataset --output_dir ./dataset_faces

# Step 2: Train on face-cropped images
python train.py --data_dir ./dataset_faces --epochs 50 --batch_size 32
```

### Training with Custom Parameters
```bash
python train.py \
    --data_dir ./dataset_faces \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_dir ./checkpoints \
    --early_stopping \
    --patience 10 \
    --freeze_backbone \
    --unfreeze_epoch 10
```

### Training Options
- `--data_dir`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--freeze_backbone`: Freeze backbone initially for transfer learning
- `--unfreeze_epoch`: Epoch to unfreeze backbone (default: 10)
- `--early_stopping`: Enable early stopping (recommended)
- `--patience`: Early stopping patience (default: 10)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)

## 🔍 Testing the Trained Model

### Option 1: Single Image Inference
```bash
python inference.py \
    --model_path ./checkpoints/best_model.pth \
    --image_path /path/to/image.jpg
```

### Option 2: Batch Inference (Directory)
```bash
python inference.py \
    --model_path ./checkpoints/best_model.pth \
    --directory_path /path/to/images/ \
    --output_file results.csv
```

### Option 3: Streamlit Web App (Single Face)
```bash
streamlit run streamlit_app.py
```
- Upload an image through the web interface
- Get real-time predictions with confidence scores

### Option 4: Streamlit Multi-Face Detection App
```bash
streamlit run streamlit_face_app.py
```
- Detects and analyzes ALL faces in an image
- Shows individual predictions for each face
- Draws bounding boxes with color-coded results
- Provides summary statistics

## 📁 Project Structure

```
deepfake-detection/
├── train.py                  # Main training script
├── inference.py              # Command-line inference
├── model.py                  # EfficientNet-B4 model definition
├── data_utils.py             # Data loading and preprocessing
├── training_utils.py         # Training utilities and metrics
├── preprocess_faces.py       # Face detection and cropping
├── streamlit_app.py          # Single image web interface
├── streamlit_face_app.py     # Multi-face detection web interface
├── requirements.txt          # Python dependencies
├── helper.md                 # Original specifications
└── dataset/                  # Your dataset directory
    ├── train/
    ├── val/
    └── test/
```

## 📊 Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Architecture | EfficientNet-B4 | Pre-trained on ImageNet |
| Input Size | 380×380 | Optimal for EfficientNet-B4 |
| Batch Size | 32 | Adjustable based on GPU memory |
| Learning Rate | 1e-4 | With AdamW optimizer |
| Mixed Precision | Enabled | For faster training |
| Early Stopping | Enabled | Prevents overfitting |

## 📈 Training Output

The training script generates:

**Checkpoints/** directory containing:
- `best_model.pth` - Best model based on validation accuracy
- `final_model.pth` - Final model after training
- `checkpoint_epoch_X.pth` - Periodic checkpoints
- `training_metrics.png` - Training/validation curves
- `test_evaluation.png` - Test set confusion matrix
- `test_results.txt` - Final test metrics

**Console Output:**
- Real-time progress bars with loss and accuracy
- Epoch summaries with train/val metrics
- Learning rate updates and early stopping notifications
- Final evaluation results on test set

## 🎯 Face Preprocessing Benefits

Using `preprocess_faces.py` before training provides:
- **Better Focus**: Model trains only on facial features
- **Improved Accuracy**: Eliminates background noise
- **Faster Training**: Smaller, more relevant data
- **Multi-Face Handling**: Extracts all faces from images
- **Consistent Input**: Standardized face crops

## 🖥️ Web Interface Features

### Single Face App (`streamlit_app.py`)
- Simple drag-and-drop interface
- Real-time predictions
- Confidence scores and probabilities

### Multi-Face App (`streamlit_face_app.py`)
- Detects all faces in an image using MTCNN
- Individual classification for each face
- Visual annotations with color-coded bounding boxes
- Summary statistics (real vs. deepfake counts)
- Detailed confidence scores per face

## 🔧 Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Reduce `batch_size` from 32 to 16 or 8
- Use `--num_workers 0` if encountering data loading issues

### Dataset Issues
- Verify folder structure matches expected format
- Check image file extensions (.jpg, .png supported)
- Ensure sufficient disk space for checkpoints

## 💡 Tips for Best Results

1. **Use Face Preprocessing**: Always preprocess your dataset for better accuracy
2. **Monitor Training**: Watch validation metrics to prevent overfitting
3. **Early Stopping**: Enable early stopping to save time and prevent overfitting
4. **Transfer Learning**: Start with frozen backbone, then fine-tune
5. **Web Testing**: Use Streamlit apps for easy model testing and demonstration

## 🔄 Resume Training

```bash
# Resume from checkpoint
python train.py \
    --data_dir ./dataset_faces \
    --resume ./checkpoints/checkpoint_epoch_25.pth
```

---

**Happy Training! 🚀**

For original specifications, refer to `helper.md`.
