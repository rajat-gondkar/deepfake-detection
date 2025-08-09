#!/usr/bin/env python3
"""
Deepfake Detection Training Script
Based on EfficientNet-B4 architecture for binary classification (Real vs Deepfake)
Optimized for NVIDIA RTX 4070 Ti GPU with Ubuntu 22.04
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import os
import time
import argparse
from tqdm import tqdm

# Local imports
from model import get_model
from data_utils import get_dataloaders, calculate_class_weights
from training_utils import (
    train_one_epoch, validate_one_epoch, evaluate_model,
    EarlyStopping, MetricsTracker, save_checkpoint
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Deepfake Detection Training')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Path to dataset directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    
    # Model arguments
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone initially for transfer learning')
    parser.add_argument('--unfreeze_epoch', type=int, default=5,
                        help='Epoch to unfreeze backbone (if frozen initially)')
    
    # Hardware arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    
    # Checkpoint arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Class names: {class_names}")
    
    # Calculate class weights for handling imbalance
    class_weights = calculate_class_weights(args.data_dir)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Create model
    print("Creating model...")
    model = get_model(
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Unfreeze backbone if specified
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            print(f"\nUnfreezing backbone at epoch {epoch + 1}")
            model.unfreeze_backbone()
            # Update optimizer with new parameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, scaler, 
            device, epoch, args.epochs
        )
        
        # Validation phase
        val_loss, val_acc, val_predictions, val_labels = validate_one_epoch(
            model, dataloaders['val'], criterion, device, epoch, args.epochs
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics_tracker.update(train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{args.epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"Epoch Time: {epoch_time:.1f}s")
        print("-" * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Early stopping check
        if early_stopping:
            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs - 1, val_loss, val_acc, final_checkpoint_path)
    
    # Plot training metrics
    print("\nGenerating training plots...")
    metrics_plot_path = os.path.join(args.save_dir, 'training_metrics.png')
    metrics_tracker.plot_metrics(save_path=metrics_plot_path)
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation on test set
    if 'test' in dataloaders:
        print("\nEvaluating on test set...")
        test_results = evaluate_model(
            model, dataloaders['test'], device, class_names,
            save_path=os.path.join(args.save_dir, 'test_evaluation.png')
        )
        
        # Save test results
        results_path = os.path.join(args.save_dir, 'test_results.txt')
        with open(results_path, 'w') as f:
            f.write("Final Test Results:\n")
            f.write(f"Accuracy: {test_results['accuracy']:.4f}\n")
            f.write(f"Precision: {test_results['precision']:.4f}\n")
            f.write(f"Recall: {test_results['recall']:.4f}\n")
            f.write(f"F1-Score: {test_results['f1_score']:.4f}\n")
        
        print(f"Test results saved to {results_path}")
    
    print(f"\nAll outputs saved to: {args.save_dir}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
