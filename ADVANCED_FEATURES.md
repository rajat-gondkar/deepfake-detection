# 🚀 Advanced Training Features Guide

## 🎯 **NEW DEFAULT BEHAVIOR**

**Early Stopping is now ENABLED by default!**

### Basic Training (Enhanced):
```bash
python3 train.py --data_dir ./dataset --epochs 15
```
Now includes:
- ✅ **Early stopping** with patience=7 (enabled by default)
- ✅ **Cosine annealing** learning rate scheduler
- ✅ **Automatic best model saving**
- ✅ **Enhanced progress bars**

---

## 🛠️ **All Available Features**

### **1. Early Stopping Control**
```bash
# Early stopping enabled by default (NEW!)
python3 train.py --data_dir ./dataset

# Disable early stopping (train all epochs)
python3 train.py --data_dir ./dataset --no_early_stopping

# Custom patience (default: 7)
python3 train.py --data_dir ./dataset --patience 10
```

### **2. Transfer Learning Options**
```bash
# Freeze backbone for first 5 epochs, then fine-tune
python3 train.py --data_dir ./dataset --freeze_backbone --unfreeze_epoch 5

# Custom unfreeze epoch
python3 train.py --data_dir ./dataset --freeze_backbone --unfreeze_epoch 3
```

### **3. Learning Rate Schedulers**
```bash
# Cosine annealing (default)
python3 train.py --data_dir ./dataset --scheduler cosine

# Step scheduler (reduce LR every few epochs)
python3 train.py --data_dir ./dataset --scheduler step

# Exponential decay
python3 train.py --data_dir ./dataset --scheduler exponential

# Reduce on plateau (adaptive)
python3 train.py --data_dir ./dataset --scheduler plateau --scheduler_patience 3 --scheduler_factor 0.5
```

### **4. Advanced Regularization**
```bash
# Gradient clipping (prevents exploding gradients)
python3 train.py --data_dir ./dataset --gradient_clip 1.0

# Label smoothing (prevents overconfidence)
python3 train.py --data_dir ./dataset --label_smoothing 0.1

# Combine multiple regularization techniques
python3 train.py --data_dir ./dataset \
    --gradient_clip 1.0 \
    --label_smoothing 0.1 \
    --weight_decay 1e-4
```

### **5. Data Augmentation (Advanced)**
```bash
# MixUp augmentation (blends images and labels)
python3 train.py --data_dir ./dataset --mixup_alpha 0.2

# CutMix augmentation (patches from different images)
python3 train.py --data_dir ./dataset --cutmix_alpha 1.0

# Both augmentations
python3 train.py --data_dir ./dataset --mixup_alpha 0.2 --cutmix_alpha 1.0
```

### **6. Enhanced Monitoring**
```bash
# More frequent logging
python3 train.py --data_dir ./dataset --log_every 50

# Validate every 2 epochs (saves time)
python3 train.py --data_dir ./dataset --validate_every 2

# Keep best 5 model checkpoints
python3 train.py --data_dir ./dataset --save_best_k 5

# Test-time augmentation for better validation accuracy
python3 train.py --data_dir ./dataset --test_time_augmentation
```

### **7. Weights & Biases Logging** (Optional)
```bash
# Enable W&B logging (requires: pip install wandb)
python3 train.py --data_dir ./dataset --wandb_project "deepfake-detection"
```

### **8. Learning Rate Warmup**
```bash
# Gradual learning rate increase for first few epochs
python3 train.py --data_dir ./dataset --warmup_epochs 3
```

---

## 🏆 **Recommended Configurations**

### **🚀 Fast Training (Quick Results)**
```bash
python3 train.py \
    --data_dir ./dataset \
    --epochs 10 \
    --batch_size 64 \
    --patience 5 \
    --freeze_backbone \
    --unfreeze_epoch 3
```

### **🎯 High Accuracy (Competition Setup)**
```bash
python3 train.py \
    --data_dir ./dataset \
    --epochs 25 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --patience 10 \
    --gradient_clip 1.0 \
    --label_smoothing 0.1 \
    --mixup_alpha 0.2 \
    --cutmix_alpha 1.0 \
    --test_time_augmentation \
    --scheduler plateau \
    --warmup_epochs 3
```

### **💪 Robust Training (Production Ready)**
```bash
python3 train.py \
    --data_dir ./dataset \
    --epochs 20 \
    --batch_size 32 \
    --patience 7 \
    --gradient_clip 0.5 \
    --label_smoothing 0.05 \
    --scheduler cosine \
    --save_best_k 3 \
    --log_every 100 \
    --wandb_project "deepfake-production"
```

### **🔬 Research & Experimentation**
```bash
python3 train.py \
    --data_dir ./dataset \
    --epochs 30 \
    --patience 15 \
    --no_early_stopping \
    --mixup_alpha 0.4 \
    --cutmix_alpha 1.2 \
    --gradient_clip 2.0 \
    --label_smoothing 0.2 \
    --warmup_epochs 5 \
    --save_every 2 \
    --validate_every 1
```

---

## 📊 **Feature Impact Guide**

| Feature | Impact | When to Use |
|---------|--------|-------------|
| **Early Stopping** | Prevents overfitting | Always (enabled by default) |
| **Gradient Clipping** | Stabilizes training | When loss spikes occur |
| **Label Smoothing** | Better generalization | For better real-world performance |
| **MixUp/CutMix** | Improved robustness | When you have sufficient data |
| **Freeze Backbone** | Faster initial training | When starting with small datasets |
| **Warmup Epochs** | Stable training start | With high learning rates |
| **Test-Time Augmentation** | Higher validation accuracy | For final model evaluation |
| **Plateau Scheduler** | Adaptive learning | When training plateaus |

---

## ⚡ **Performance Tips**

### **For RTX 4070 Ti (12GB VRAM):**
```bash
# Optimal batch sizes
--batch_size 32    # Safe default
--batch_size 48    # If memory allows
--batch_size 64    # Maximum (with smaller models)

# Optimal worker counts
--num_workers 4    # Default
--num_workers 6    # If CPU has 8+ cores
--num_workers 8    # If CPU has 12+ cores
```

### **Memory Optimization:**
```bash
# Reduce memory usage
python3 train.py --data_dir ./dataset --batch_size 16 --num_workers 2

# Increase throughput
python3 train.py --data_dir ./dataset --batch_size 48 --num_workers 6
```

---

## 🔄 **Migration from Old Version**

**Old Command:**
```bash
python3 train.py --data_dir ./dataset --epochs 15 --early_stopping
```

**New Equivalent (Early stopping now default):**
```bash
python3 train.py --data_dir ./dataset --epochs 15
```

**To disable early stopping (old behavior):**
```bash
python3 train.py --data_dir ./dataset --epochs 15 --no_early_stopping
```

---

## 🎭 **Example Training Output**

```
✅ Early stopping enabled with patience=7
✅ Gradient clipping enabled: 1.0
✅ Label smoothing enabled: 0.1
✅ MixUp augmentation enabled: alpha=0.2

🚂 Epoch  1/15 [Train]: 100%|██████████| 4375/4375 [12:34<00:00, 5.8batch/s] Loss=0.6543, Acc=0.723, Samples=140000, Speed=185/s
🔍 Epoch  1/15 [Valid]: 100%|██████████| 625/625 [01:45<00:00, 5.9batch/s] Loss=0.5432, Acc=0.789, Samples=20000, Speed=190/s

Epoch 1/15 Summary:
Train Loss: 0.6543 | Train Acc: 0.7234
Val Loss: 0.5432 | Val Acc: 0.7891
Learning Rate: 1.00e-04
Epoch Time: 14.3s
New best model saved! Val Acc: 0.7891

🛑 Early stopping triggered at epoch 12 - Best Val Acc: 0.9456
```

All these features are now available and ready to use! 🚀
