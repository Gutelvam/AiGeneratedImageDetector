# src/models.py
import torch
import torch.nn as nn
import timm

class FakeImageDetectorCNN(nn.Module):
    def __init__(self, num_classes=2, model_name='resnet18', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # Adjust the final classification layer
        if hasattr(self.model, 'fc'): # For ResNet, EfficientNet
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'head'): # For some other models
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} does not have a recognizable classification head.")

    def forward(self, x):
        return self.model(x)

class FakeImageDetectorViT(nn.Module):
    def __init__(self, num_classes=2, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # Adjust the final classification layer
        if hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} does not have a recognizable classification head.")

    def forward(self, x):
        return self.model(x)

def get_model(model_type='cnn', model_name='resnet18', num_classes=2, pretrained=True):
    if model_type == 'cnn':
        return FakeImageDetectorCNN(num_classes, model_name, pretrained)
    elif model_type == 'vit':
        return FakeImageDetectorViT(num_classes, model_name, pretrained)
    else:
        raise ValueError("model_type must be 'cnn' or 'vit'")