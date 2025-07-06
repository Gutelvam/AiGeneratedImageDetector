# inference.py
import torch
import torch.nn as nn
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Import necessary components from your src directory
from src.models import get_model
from src.data_loader import AlbumentationsTransform, get_transforms # Reusing the transform logic

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def predict_image(model, image_path, device, image_size=(224, 224)):
    """
    Loads an image, preprocesses it using the same pipeline as training/validation,
    and makes a prediction using the model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        image_path (str): Path to the image file.
        device (torch.device): The device (CPU/GPU) to run inference on.
        image_size (tuple): The (height, width) the model was trained with.
                            The input image will be resized to this.

    Returns:
        tuple: (predicted_label_name, confidence, probabilities, original_image_np, input_tensor)
    """
    # Get the same validation/test transforms used during training
    # This includes resizing to image_size, normalization, and ToTensorV2
    _, val_test_transforms = get_transforms(image_size)

    # Load image using PIL
    image_pil = Image.open(image_path).convert("RGB")
    original_image_np = np.array(image_pil) # Keep original for Grad-CAM visualization (HWC, RGB, uint8)

    # Apply the transforms. The AlbumentationsTransform expects a NumPy array.
    # It will handle resizing, normalization, and conversion to PyTorch tensor.
    input_tensor = val_test_transforms(image=original_image_np).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(device)

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0] # Get probabilities for the single image

    # IMPORTANT: Verify this order matches your training data's class mapping!
    # ImageFolder typically sorts alphabetically, so 'fake' (0) and 'real' (1).
    class_names = ["fake", "real"] # Example: Adjust if your classes are ordered differently
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_label = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()

    return predicted_label, confidence, probabilities, original_image_np, input_tensor, class_names

def plot_probabilities(probabilities, class_names, output_dir, filename="probabilities.png"):
    """
    Plots the probabilities for each class.

    Args:
        probabilities (torch.Tensor): Tensor of probabilities for each class.
        class_names (list): List of class names (e.g., ['fake', 'real']).
        output_dir (str): Directory to save the plot.
        filename (str): Name of the file to save the plot as.
    """
    plt.figure(figsize=(6, 4))
    plt.bar(class_names, probabilities.cpu().numpy(), color=['skyblue', 'lightcoral'])
    plt.ylabel("Probability")
    plt.title("Class Probabilities")
    plt.ylim(0, 1)
    for i, prob in enumerate(probabilities):
        plt.text(i, prob + 0.02, f"{prob:.2f}", ha='center', va='bottom')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Fake Image Detection Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file for inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights (.pth file).")
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "vit"], help="Type of model (cnn or vit).")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Specific model architecture name (e.g., resnet18, vit_base_patch16_224).")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size (height width) model was trained on.")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save inference plots and Grad-CAM images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    print(f"Loading {args.model_type} model: {args.model_name} from {args.model_path}...")
    # Assuming 2 classes (fake, real)
    model = get_model(model_type=args.model_type, model_name=args.model_name, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode

    # Perform prediction
    print(f"Performing inference on {args.image_path}...")
    predicted_label, confidence, probabilities, original_image_np, input_tensor, class_names = \
        predict_image(model, args.image_path, device, tuple(args.image_size))

    print("\n--- Inference Results ---")
    print(f"Image: {args.image_path}")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: {probabilities.cpu().numpy()}")

    # Plot probabilities
    print(f"Saving probability plot to {args.output_dir}...")
    plot_probabilities(probabilities, class_names, args.output_dir,
                       filename=f"probabilities_{os.path.basename(args.image_path)}.png")

    # Grad-CAM Visualization
    print("Applying Grad-CAM for interpretability...")
    try:
        # Choose a target layer for Grad-CAM. This needs to be a convolutional layer
        # before the global pooling/classification head.
        # For ResNet: model.model.layer4[-1]
        # For ViT: model.model.blocks[-1].norm1 (or similar, requires inspection)
        if args.model_type == 'cnn':
            target_layers = [model.model.layer4[-1]] # Example for ResNet
        elif args.model_type == 'vit':
            target_layers = [model.model.blocks[-1].norm1] # Example for ViT

        cam = GradCAM(model=model, target_layers=target_layers)
        # Target the predicted class for Grad-CAM
        target_class_idx = class_names.index(predicted_label)
        targets = [ClassifierOutputTarget(target_class_idx)]

        # The input_tensor is already normalized. For visualization, we need to denormalize the original image.
        # The show_cam_on_image function expects a 0-1 float numpy array.
        # original_image_np is uint8 (0-255), convert to float (0-1)
        rgb_img_for_cam = original_image_np.astype(np.float32) / 255.0

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Remove batch dimension

        cam_image = show_cam_on_image(rgb_img_for_cam, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(8, 6))
        plt.imshow(cam_image)
        plt.title(f"Grad-CAM - Predicted: {predicted_label} ({confidence:.2f})")
        plt.axis('off')
        cam_filename = f"grad_cam_{os.path.basename(args.image_path)}.png"
        plt.savefig(os.path.join(args.output_dir, cam_filename))
        plt.show()
        print(f"Grad-CAM visualization saved to {os.path.join(args.output_dir, cam_filename)}")

    except Exception as e:
        print(f"Error during Grad-CAM visualization: {e}")
        print("Ensure 'target_layers' are correctly identified for your specific model architecture.")
        print("You might need to inspect your model's layers (e.g., by printing the model structure).")


if __name__ == "__main__":
    main()
