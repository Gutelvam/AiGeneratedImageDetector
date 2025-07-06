# src/data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image # <--- ADD THIS IMPORT
import numpy as np # <--- ADD THIS IMPORT

# Define a custom transform class or function for Albumentations
# This makes the transform object picklable
class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img_np): # Expects a NumPy array (HWC, RGB, uint8)
        # Apply the Albumentations transform
        return self.transform(image=img_np)['image']


def get_transforms(image_size=(224, 224)):
    train_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet stats
        ToTensorV2(), # Converts to PyTorch Tensor, normalizes by 255.0, and permutes to C, H, W
    ])

    val_test_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    # Return instances of the picklable wrapper
    return AlbumentationsTransform(train_transform), AlbumentationsTransform(val_test_transform)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]

        # 1. Read image using PIL (Pillow)
        image_pil = Image.open(img_path).convert("RGB") # Ensure RGB mode

        # 2. Convert PIL image to NumPy array (HWC, RGB, uint8)
        image_np = np.array(image_pil)

        if self.transform:
            image_tensor = self.transform(image_np) # Pass numpy array
        else:
            # If no transform, still need to convert to tensor and normalize if necessary
            # For simplicity, ensure transform is always applied
            raise ValueError("Transform must be provided to ImageDataset.")

        return image_tensor, label

def get_dataloaders(data_dir="./data", image_size=(224, 224), batch_size=32):
    train_transforms, val_test_transforms = get_transforms(image_size)

    # Note: datasets.ImageFolder itself doesn't apply transforms directly here.
    # It just sets up the structure (paths and labels).
    # Our custom ImageDataset then takes these paths and applies our Albumentations transforms.
    train_dataset_raw = datasets.ImageFolder(f"{data_dir}/Train")
    val_dataset_raw = datasets.ImageFolder(f"{data_dir}/Validation")
    test_dataset_raw = datasets.ImageFolder(f"{data_dir}/Test")

    train_dataset = ImageDataset(train_dataset_raw, transform=train_transforms)
    val_dataset = ImageDataset(val_dataset_raw, transform=val_test_transforms)
    test_dataset = ImageDataset(test_dataset_raw, transform=val_test_transforms)

    # It's generally recommended to keep num_workers > 0 on multi-core systems for performance,
    # but 0 can be used for debugging if you suspect multiprocessing issues
    # that are not related to pickling (like the lambda issue we fixed earlier).
    # For now, let's keep it at 4 or 8 if you have enough CPU cores.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_dataset_raw.classes