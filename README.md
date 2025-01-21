# PCOS Classification using Deep Learning

## 📌 Project Overview
This project is a deep learning-based image classification system designed to detect **Polycystic Ovary Syndrome (PCOS)** from medical images. The model is built using **PyTorch** and follows a modular structure to ensure scalability, maintainability, and efficiency.

## 📁 Directory Structure
```
PCOS-Classification/
│── data/
│── models/
│   ├── model.py      # Model architecture
│   ├── train.py      # Training pipeline
│   ├── predict.py    # Prediction script
│── utils/
│   ├── dataset.py    # Data loading & preprocessing
│   ├── train_utils.py  # Training helper functions
│   ├── test_utils.py   # Testing helper functions
│── requirements.txt    # Dependencies
│── README.md           # Documentation
```

## ⚙️ Setup and Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-repo/PCOS-Classification.git
cd PCOS-Classification
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Prepare Dataset
Dataset Link : [ https://github.com/xajeel/lite-skip-bridgeNet.git ]

Place your images inside the `data/train/` and `data/test/` directories following this structure:
```
data/
│── train/
│   ├── class_0/
│   ├── class_1/
│── test/
│   ├── class_0/
│   ├── class_1/
```

## 🚀 Training the Model
Run the following command to start training:
```sh
python models/train.py
```
This will:
- Load the dataset
- Train the model for the specified number of epochs
- Save the trained model

## 🔍 Making Predictions
To make predictions on new images:
```sh
python models/predict.py --image_path test_image.jpg --model_path best_model.pth
```

## 🏗️ Modular Breakdown
### **1️⃣ Model Architecture (`models/model.py`)**
Defines the deep learning model using convolutional layers and fully connected layers.

### **2️⃣ Data Loader (`utils/dataset.py`)**
Handles dataset loading, transformations, and splitting into training/testing sets.

### **3️⃣ Training Pipeline (`models/train.py`)**
Implements training logic, loss calculation, and accuracy tracking.

### **5️⃣ Prediction Script (`models/predict.py`)**
Loads the trained model and predicts the class of a new image.

## 📊 Training Details
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Augmentations**: Resizing, Rotation, Color Jitter, Normalization

## 🤖 Future Improvements
- Implement **Grad-CAM** to visualize model decisions
- Train on a larger dataset
- Experiment with different architectures (ResNet, EfficientNet)

## 🏆 Credits
Developed by **M sajeel & Amna Khan** as part of a research project on PCOS classification.

## 📜 License
This project is licensed under the MIT License. Feel free to use and modify it!

