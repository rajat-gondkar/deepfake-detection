import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    def __init__(self, model_name='efficientnet_b4', num_classes=2, pretrained=True):
        """
        EfficientNet-B4 based deepfake detection model
        
        Args:
            model_name (str): Model architecture name
            num_classes (int): Number of output classes (2 for real/deepfake)
            pretrained (bool): Whether to use pretrained weights
        """
        super(DeepfakeDetector, self).__init__()
        
        # Load EfficientNet-B4 backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove the classifier head
            global_pool='avg'
        )
        
        # Get the number of features from the backbone
        self.num_features = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def freeze_backbone(self):
        """Freeze the backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_model(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Create and return the deepfake detection model
    
    Args:
        num_classes (int): Number of classes
        pretrained (bool): Use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone initially
        
    Returns:
        model: DeepfakeDetector model
    """
    model = DeepfakeDetector(
        model_name='efficientnet_b4',
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    if freeze_backbone:
        model.freeze_backbone()
        
    return model
