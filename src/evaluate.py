# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.models import get_model
from src.data_loader import get_dataloaders
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, model_save_path="models/best_model.pth"):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}")

        # Validation Phase
        val_loss, val_accuracy, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation accuracy: {best_val_accuracy:.4f}")

    print("Training complete!")

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of being fake

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.5 # Default if only one class is present

    return total_loss, accuracy, precision, recall, f1, roc_auc, all_labels, all_predictions