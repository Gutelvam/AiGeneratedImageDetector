# Fake Image Detection using Deep Learning

## 1. Background
The proliferation of artificial image generation technologies necessitates robust methods for detecting synthetic images to combat misinformation and fraud. This project aims to develop deep learning models for this crucial task.

## 2. Objectives
- Develop deep learning models (CNNs and ViTs) for fake image detection.
- Compare the performance of CNNs and ViTs.
- Create an end-to-end pipeline for image authenticity evaluation.
- Apply interpretability techniques (e.g., Grad-CAM).

## 3. Methodology Overview
The project will involve data acquisition and processing, model development (CNNs and ViTs), optimization, and evaluation.

## 4. Folder Structure
- `data/`: Contains training, validation, and test datasets.
  - `Train/`: Training images (both real and fake).
  - `Validation/`: Validation images (both real and fake).
  - `Test/`: Test images (both real and fake).
- `models/`: (To be created) Stores trained model weights.
- `results/`: (To be created) Stores evaluation metrics, plots, and interpretability outputs.
- `src/`: (To be created) Contains Python scripts for data processing, model definitions, training, and evaluation.
- `notebooks/`: (Optional, to be created) For experimentation and exploration.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.
- `main.py` or `train.py`: Main script to run the project.

# STEPS
- 1 - create virtual enviroment
 `python -m venv venv`
- 2 - Activate virtual enviroment
`source venv/bin/activate`
- 3- Install dependences
`pip install -r requirements.txt`


```
.
├── readme.md
├── main.py                   # Main script to run the project
├── requirements.txt          # Python dependencies
├── data/
│   ├── Train/
│   │   ├── real/
│   │   └── fake/
│   ├── Validation/
│   │   ├── real/
│   │   └── fake/
│   └── Test/
│       ├── real/
│       └── fake/
├── src/
│   ├── __init__.py           # Empty file to make 'src' a Python package
│   ├── data_loader.py        # Handles data loading and augmentation
│   ├── models.py             # Defines CNN and ViT architectures
│   └── train.py              # Contains training and evaluation logic
├── models/                   # Directory to save trained model weights
│   └── best_cnn_resnet18.pth # Example saved model
├── results/                  # Directory to save evaluation plots and Grad-CAM images
│   ├── confusion_matrix_cnn_resnet18.png
│   ├── grad_cam_cnn_resnet18_sample_0.png
│   └── ...
└── (Optional) app.py         # Streamlit or Flask app for demo
```