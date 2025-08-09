import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_transforms():
    """
    Get data transforms for training, validation, and testing
    
    Returns:
        dict: Dictionary containing transforms for each split
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Validation and test transforms (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return {
        'train': train_transforms,
        'val': val_test_transforms,
        'test': val_test_transforms
    }


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        dict: Dictionary containing data loaders and dataset sizes
    """
    transforms_dict = get_transforms()
    
    # Create datasets
    datasets_dict = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            datasets_dict[split] = datasets.ImageFolder(
                root=split_dir,
                transform=transforms_dict[split]
            )
    
    # Create data loaders
    dataloaders = {}
    dataset_sizes = {}
    
    for split, dataset in datasets_dict.items():
        shuffle = True if split == 'train' else False
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True if split == 'train' else False
        )
        
        dataset_sizes[split] = len(dataset)
    
    # Get class names
    class_names = datasets_dict['train'].classes if 'train' in datasets_dict else None
    
    return dataloaders, dataset_sizes, class_names


def calculate_class_weights(data_dir):
    """
    Calculate class weights for handling class imbalance
    
    Args:
        data_dir (str): Path to the dataset directory
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    train_dir = os.path.join(data_dir, 'train')
    
    if not os.path.exists(train_dir):
        print("Train directory not found. Using equal weights.")
        return torch.tensor([1.0, 1.0])
    
    # Count samples in each class
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    
    print(f"Class distribution: {class_counts}")
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = []
    for class_name in sorted(class_counts.keys()):
        weight = total_samples / (num_classes * class_counts[class_name])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)
