# Deepfake Detection Model using EfficientNet-B4

## Overview
This project builds a deep learning model using the **EfficientNet-B4** architecture to classify images as either **Real** or **Deepfake**.  
The dataset consists of ~140,000 images (balanced: 50% real, 50% deepfake).  
The model is trained for **15 epochs** with optimized parameters for an **NVIDIA RTX 4070 Ti GPU**.  
During training, **live progress** will be displayed in the terminal using `tqdm` progress bars.

---

## Steps to Build the Model

### 1. Environment Setup
- **Python version:** >= 3.9
- **Recommended libraries:**
  - `torch` (PyTorch) with CUDA support
  - `torchvision`
  - `tqdm` (for progress bar display during training)
  - `timm` (for EfficientNet-B4 and other architectures)
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `Pillow` (for image processing)


### 2. Dataset Preparation
1. **Folder structure:**
    ```
    dataset/
    ├── train/
    │   ├── real/
    │   └── deepfake/
    ├── val/
    │   ├── real/
    │   └── deepfake/
    ├── test/
    │   ├── real/
    │   └── deepfake/
    ```
2. **Image format:** JPG or PNG.
3. **Image preprocessing:**
    - Resize to **380x380** (EfficientNet-B4 input size).
    - Normalize with ImageNet mean & std:
      - Mean: `[0.485, 0.456, 0.406]`
      - Std: `[0.229, 0.224, 0.225]`

---

### 3. Data Loading
- Use **`torchvision.datasets.ImageFolder`** for dataset loading.
- Use **`torch.utils.data.DataLoader`** with:
  - `batch_size = 32` (optimal for 4070 Ti)
  - `num_workers = 4` (adjust if bottleneck occurs)
  - `pin_memory = True` for GPU training speedup
- Apply **data augmentation** for training:
  - Random horizontal flip
  - Random rotation
  - Random brightness/contrast adjustment

---

### 4. Model Architecture
- Use **EfficientNet-B4** pre-trained on ImageNet.
- Modify the **final fully connected layer** to output 2 classes (real, deepfake).
- Optionally freeze base layers initially for transfer learning, then unfreeze for fine-tuning.

---

### 5. Training Configuration
- **Loss function:** CrossEntropyLoss
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-5`)
- **Scheduler:** CosineAnnealingLR or StepLR (optional)
- **Epochs:** 15
- **Device:** CUDA
- **Mixed Precision Training:** Enable with `torch.cuda.amp` for faster computation.

---

### 6. Training Loop with Progress Display
- Wrap data loaders with `tqdm` to show progress bars:
  - Show **batch progress** within each epoch.
  - Show **loss** and **accuracy** updates in the progress bar.
- Save the **best model weights** (based on validation accuracy) using `torch.save`.

Example pseudocode structure for training loop with `tqdm`:
```python
from tqdm import tqdm

for epoch in range(epochs):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch")
    for images, labels in train_loader_tqdm:
        # Training step code here
        train_loader_tqdm.set_postfix(loss=current_loss, acc=current_accuracy)

    model.eval()
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", unit="batch")
    for images, labels in val_loader_tqdm:
        # Validation step code here
        val_loader_tqdm.set_postfix(val_loss=current_val_loss, val_acc=current_val_accuracy)
```

---

### 7. Evaluation
- After training, run model on validation set.
- Compute:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion matrix
- Optionally, visualize a few predictions with their probabilities.

---

### 8. Inference
- Load the trained model weights.
- Preprocess a single image and predict its class.
- Output the class label (`Real` or `Deepfake`) and prediction probability.

---

## Training Parameters Summary
| Parameter        | Value                     |
|------------------|---------------------------|
| Architecture     | EfficientNet-B4 (pretrained) |
| Image Size       | 380x380                   |
| Batch Size       | 32                        |
| Epochs           | 15                        |
| Learning Rate    | 1e-4                      |
| Optimizer        | AdamW                     |
| Loss Function    | CrossEntropyLoss          |
| GPU              | NVIDIA RTX 4070 Ti        |
| Progress Display | `tqdm`                    |

---

## Final Notes
- Enable **mixed precision training** for performance on RTX GPUs.
- Implement **early stopping** to avoid overfitting.
- Maintain a log file for tracking accuracy and loss for each epoch.
