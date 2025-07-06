# main.py
import torch
import torch.nn as nn
import argparse
from src.data_loader import get_dataloaders
from src.models import get_model
from src.train import train_model, evaluate_model
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser(description="Fake Image Detection Project")
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "vit"], help="Type of model to use (cnn or vit)")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Specific model architecture name (e.g., resnet18, vit_base_patch16_224)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size (height width)")
    parser.add_argument("--pretrained", type=bool, default=True, help="Use pretrained weights")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Acquisition and Processing
    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        image_size=tuple(args.image_size),
        batch_size=args.batch_size
    )
    print(f"Classes: {classes}")

    # 2. Model Development
    print(f"Initializing {args.model_type} model: {args.model_name}...")
    model = get_model(
        model_type=args.model_type,
        model_name=args.model_name,
        num_classes=len(classes),
        pretrained=args.pretrained
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Optimization (Training)
    print("Starting training...")
    model_save_path = os.path.join(args.save_dir, f"best_{args.model_type}_{args.model_name}.pth")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, model_save_path)

    # Load best model for evaluation
    model.load_state_dict(torch.load(model_save_path))
    print(f"Loaded best model from {model_save_path}")

    # 4. Evaluation
    print("Evaluating on test set...")
    test_loss, test_accuracy, precision, recall, f1, roc_auc, all_labels, all_predictions = \
        evaluate_model(model, test_loader, criterion, device)

    print("\n--- Test Set Results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision (Fake): {precision:.4f}")
    print(f"Recall (Fake): {recall:.4f}")
    print(f"F1-Score (Fake): {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.results_dir, f"confusion_matrix_{args.model_type}_{args.model_name}.png"))
    plt.show()

    # 5. Interpretability (Grad-CAM)
    print("Applying Grad-CAM for interpretability...")
    # Get a few sample images from the test set
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Choose a target layer for Grad-CAM. This needs to be a convolutional layer before the global pooling/classification head.
    # For ResNet: model.model.layer4[-1]
    # For ViT: model.model.blocks[-1].norm1 (or similar, requires inspection of model architecture)
    if args.model_type == 'cnn':
        # Example for ResNet18: The last convolutional block
        target_layers = [model.model.layer4[-1]]
    elif args.model_type == 'vit':
        # Example for ViT: The last block's attention output or MLP output
        # You might need to inspect the specific ViT model's structure using `print(model)`
        target_layers = [model.model.blocks[-1].norm1] # This is a common choice for ViT
        # Or try: target_layers = [model.model.norm]

    try:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        targets = [ClassifierOutputTarget(label.item()) for label in labels[:5]] # For top 5 samples

        for i in range(min(5, len(images))):
            input_tensor = images[i].unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[targets[i]])
            grayscale_cam = grayscale_cam[0, :]

            # Denormalize image for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_img = (images[i].cpu().numpy().transpose(1, 2, 0) * std + mean).astype(np.float32)
            rgb_img = np.clip(rgb_img, 0, 1) # Ensure values are between 0 and 1

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            plt.imshow(cam_image)
            plt.title(f"Grad-CAM - True: {classes[labels[i]]}, Pred: {classes[all_predictions[i]]}")
            plt.axis('off')
            plt.savefig(os.path.join(args.results_dir, f"grad_cam_{args.model_type}_{args.model_name}_sample_{i}.png"))
            plt.show()
    except Exception as e:
        print(f"Error during Grad-CAM visualization: {e}")
        print("Make sure target_layers are correctly identified for your specific model.")


if __name__ == "__main__":
    main()