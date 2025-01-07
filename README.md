# ML
# PyTorch 101

## Table of Contents
- [Introduction](#introduction)
- [Key Functions and Topics](#key-functions-and-topics)
- [Highlights](#highlights)
- [How to Use](#how-to-use)
  
- [Assignments](#assignments)
  - [PyTorch 101](#pytorch-101)
  - [A2: Classification Models](#a2-classification-models)
    - [Part 1: Binary Classification](#part-1-binary-classification)
    - [Part 2: Softmax Regression](#part-2-softmax-regression)
    - [Part 3: Multi-Class Classification](#part-3-multi-class-classification)
    - [Part 4: Advanced Multi-Class Classification](#part-4-advanced-multi-class-classification)
  - [A3: Classification Tasks](#a3-classification-tasks)
  - [A4: YOLO Object Detection](#a4-yolo-object-detection)
    - [Main Code (A4.ipynb)](#main-code-a4ipynb)
    - [YOLO Loss](#yolo-loss)
    - [Configuration File](#configuration-file)
    - [Dataset Processing](#dataset-processing)
    - [VOC Evaluation](#voc-evaluation)
    - [Prediction Script](#prediction-script)
    - [YOLO with ResNet Backbone](#yolo-with-resnet-backbone)



## Introduction
This notebook explores fundamental concepts and operations in PyTorch, a popular machine learning framework. The main focus is to familiarize users with basic tensor manipulations, operations, and practical use cases.

---

## Key Functions and Topics

### 1. Hello Function
- A simple function to confirm the setup and execution environment.

### 2. Tensor Creation and Manipulation
- **`create_sample_tensor`**: Demonstrates how to create tensors with specific values.
- **`mutate_tensor`**: Modifies tensor elements based on given indices and values.
- **`count_tensor_elements`**: Calculates the total number of elements in a tensor without using built-in functions like `numel`.

### 3. Specialized Tensor Functions
- **`create_tensor_of_pi`**: Generates tensors filled with the value of pi (3.14).
- **`multiples_of_ten`**: Creates a tensor of multiples of ten within a specified range.

---

## Highlights
- **Purpose**: Understand and practice tensor operations to build foundational knowledge in PyTorch.
- **Steps Covered**: Implementation of functions for creating, manipulating, and analyzing tensors.
- **Key Results**: Comprehensive understanding of basic PyTorch functionalities.

---

## How to Use
1. Clone this repository.
2. Install the necessary dependencies (refer to `requirements.txt`).
3. Open the notebook in a Jupyter environment.
4. Execute the cells sequentially to explore the functionalities.

[Back to Top](#table-of-contents)

# Assignments

# PyTorch 101

## Table of Contents
- [Introduction](#introduction)
- [Key Functions and Topics](#key-functions-and-topics)
- [Code Examples](#code-examples)
  - [Tensor Creation and Manipulation](#tensor-creation-and-manipulation)
  - [Specialized Tensor Functions](#specialized-tensor-functions)
  - [Advanced Operations](#advanced-operations)
- [Highlights](#highlights)
- [How to Use](#how-to-use)

---

## Introduction
This project introduces essential PyTorch operations with hands-on examples. Each function has been carefully designed to teach key concepts, ranging from tensor creation to advanced batch operations.

---

## Key Functions and Topics

### 1. Tensor Creation and Manipulation
- **`create_sample_tensor`**: Creates a tensor with specific values.
- **`mutate_tensor`**: Modifies tensor elements based on given indices and values.
- **`count_tensor_elements`**: Counts the total number of elements in a tensor.

### 2. Specialized Tensor Functions
- **`create_tensor_of_pi`**: Generates tensors filled with the value of pi.
- **`multiples_of_ten`**: Produces a tensor containing multiples of ten.
- **`slice_indexing_practice`**: Demonstrates tensor slicing techniques.
- **`slice_assignment_practice`**: Explains in-place tensor mutation using slicing.

### 3. Advanced Operations
- **`shuffle_cols`**: Re-orders columns in a tensor using indexing.
- **`reverse_rows`**: Reverses the rows of a tensor efficiently.
- **`normalize_columns`**: Normalizes tensor columns by subtracting the mean and dividing by the standard deviation.
- **`batched_matrix_multiply`**: Implements batched matrix multiplication with and without explicit loops.

---

## Code Examples

### Tensor Creation and Manipulation

#### Create a Sample Tensor
```python
from pytorch101 import create_sample_tensor

tensor = create_sample_tensor()
print(tensor)
# Output:
# tensor([[  0.,  10.],
#         [100.,   0.],
#         [  0.,   0.]])
```

#### Mutate Tensor
```python
from pytorch101 import mutate_tensor
import torch

tensor = torch.zeros((3, 3))
indices = [(0, 0), (1, 2), (2, 1)]
values = [1, 5, 10]
mutated_tensor = mutate_tensor(tensor, indices, values)
print(mutated_tensor)
# Output:
# tensor([[ 1.,  0.,  0.],
#         [ 0.,  0.,  5.],
#         [ 0., 10.,  0.]])
```

#### Count Tensor Elements
```python
from pytorch101 import count_tensor_elements
import torch

tensor = torch.ones((4, 5, 6))
num_elements = count_tensor_elements(tensor)
print(num_elements)
# Output: 120
```

---

### Specialized Tensor Functions

#### Create Tensor of Pi
```python
from pytorch101 import create_tensor_of_pi

tensor_pi = create_tensor_of_pi(3, 4)
print(tensor_pi)
# Output:
# tensor([[3.1400, 3.1400, 3.1400, 3.1400],
#         [3.1400, 3.1400, 3.1400, 3.1400],
#         [3.1400, 3.1400, 3.1400, 3.1400]])
```

#### Multiples of Ten
```python
from pytorch101 import multiples_of_ten

tensor_ten = multiples_of_ten(5, 50)
print(tensor_ten)
# Output:
# tensor([10., 20., 30., 40., 50.], dtype=torch.float64)
```

#### Slice Indexing Practice
```python
from pytorch101 import slice_indexing_practice
import torch

tensor = torch.arange(15).view(3, 5)
results = slice_indexing_practice(tensor)
print(results)
# Output: Tensors demonstrating slicing operations
```

---

### Advanced Operations

#### Shuffle Columns
```python
from pytorch101 import shuffle_cols
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
shuffled = shuffle_cols(tensor)
print(shuffled)
# Output:
# tensor([[1, 1, 3, 2],
#         [4, 4, 6, 5]])
```

#### Reverse Rows
```python
from pytorch101 import reverse_rows
import torch

tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
reversed_tensor = reverse_rows(tensor)
print(reversed_tensor)
# Output:
# tensor([[5, 6],
#         [3, 4],
#         [1, 2]])
```

#### Normalize Columns
```python
from pytorch101 import normalize_columns
import torch

tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
normalized = normalize_columns(tensor)
print(normalized)
# Output: Normalized tensor with zero mean and unit variance for each column
```

#### Batched Matrix Multiply
```python
from pytorch101 import batched_matrix_multiply
import torch

x = torch.rand((2, 3, 4))
y = torch.rand((2, 4, 5))
result = batched_matrix_multiply(x, y)
print(result.shape)
# Output: torch.Size([2, 3, 5])
```

---

## Highlights
- **Purpose**: Build familiarity with PyTorch operations using hands-on examples.
- **Concepts**:
  - Basic tensor operations (creation, mutation, slicing).
  - Advanced matrix operations (batch processing, normalization).
- **Key Results**: Comprehensive understanding of essential PyTorch functions and best practices.

---

## How to Use
1. Open the Python script or Jupyter notebook in your preferred environment.
2. Execute the examples to understand and practice PyTorch operations.

---

[Back to Top](#table-of-contents)

# A2: Classification Models

### Part 1: Binary Classification

## Overview
In this section, we focus on building a binary classification model using logistic regression. The goal is to predict binary outcomes and evaluate performance using appropriate metrics.

## Key Concepts
- Logistic regression implementation
- Binary cross-entropy loss
- Performance metrics: accuracy, precision, recall, F1-score
- Data preprocessing (normalization, splitting)

## Code Highlights

### Logistic Regression Training
```python
from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Predictions
```python
# Make predictions
predictions = model.predict(X_test)
```

### Evaluation Metrics
```python
from sklearn.metrics import classification_report

# Evaluate the model
print(classification_report(y_test, predictions))
```

### Data Preprocessing Example
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Highlights
- **Purpose**: Build a binary classification model and evaluate its performance.
- **Learning Objectives**:
  - Implement logistic regression from scratch.
  - Use evaluation metrics to assess model performance.
  - Apply data preprocessing techniques for better model performance.

---

[Back to Top](#overview)

### Part 2: Softmax Regression

## Overview
This part introduces softmax regression, extending logistic regression to handle multi-class classification problems. The softmax function allows us to assign probabilities to each class, making it a powerful tool for multi-class classification.

## Key Concepts
- Softmax activation function
- Cross-entropy loss for multi-class problems
- Gradient descent optimization
- Model convergence analysis

## Code Highlights

### Softmax Function Implementation
```python
import torch.nn.functional as F

# Apply softmax to logits
outputs = F.softmax(logits, dim=1)
```

### Cross-Entropy Loss Calculation
```python
# Compute cross-entropy loss
loss = F.cross_entropy(outputs, targets)
```

### Gradient Descent Optimization
```python
import torch.optim as optim

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Optimization step
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Training Loop Example
```python
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = F.cross_entropy(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

## Highlights
- **Purpose**: Extend logistic regression to handle multi-class classification using softmax.
- **Learning Objectives**:
  - Understand and implement the softmax function.
  - Calculate and minimize cross-entropy loss.
  - Train a multi-class classifier using gradient descent.

---

[Back to Top](#overview)

### Part 3: Multi-Class Classification

## Overview
This section builds on the concepts of softmax regression to implement practical multi-class classification tasks. We delve deeper into encoding techniques, evaluation metrics, and model training workflows for multi-class problems.

## Key Concepts
- One-hot encoding for categorical labels
- Confusion matrix analysis
- Multi-class accuracy, precision, and recall
- Model training and validation workflow

## Code Highlights

### One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder

# Convert categorical labels to one-hot encoding
encoder = OneHotEncoder()
y_one_hot = encoder.fit_transform(y).toarray()
```

### Model Training Example
```python
from sklearn.linear_model import LogisticRegression

# Train a multi-class classifier
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
```

### Confusion Matrix Analysis
```python
from sklearn.metrics import confusion_matrix

# Generate predictions and confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

### Performance Evaluation
```python
from sklearn.metrics import classification_report

# Evaluate the multi-class classifier
print(classification_report(y_test, y_pred))
```

## Highlights
- **Purpose**: Expand on softmax regression by implementing multi-class classification workflows.
- **Learning Objectives**:
  - Understand data encoding for multi-class classification.
  - Evaluate model performance using confusion matrix and classification reports.
  - Build practical multi-class classification pipelines.

---

[Back to Top](#overview)

### Part 4: Advanced Multi-Class Classification

## Overview
This section focuses on advanced techniques for optimizing multi-class classification models. We explore feature engineering, hyperparameter tuning, and model comparisons to enhance classification performance.

## Key Concepts
- Feature importance analysis
- Hyperparameter tuning (GridSearchCV)
- Comparison of classification algorithms (Random Forest, SVM, Neural Networks)
- ROC curve and AUC for multi-class problems

## Code Highlights

### Feature Importance Analysis
```python
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Analyze feature importance
importances = model.feature_importances_
print(importances)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}

# Perform grid search
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
```

### Model Comparison
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Train and compare models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

### ROC Curve and AUC
```python
from sklearn.metrics import roc_auc_score, roc_curve

# Compute AUC for multi-class classifier
y_prob = model.predict_proba(X_test)
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print("AUC:", auc)

# Plot ROC Curve
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(model, X_test, y_test)
```

## Highlights
- **Purpose**: Optimize multi-class classification models using advanced techniques.
- **Learning Objectives**:
  - Perform feature engineering and importance analysis.
  - Utilize hyperparameter tuning to enhance model performance.
  - Compare different classification algorithms for multi-class problems.
  - Evaluate models using ROC curves and AUC scores.

---

[Back to Top](#overview)


## A3: Classification Tasks

## Overview
This section focuses on a combination of classification tasks, incorporating various models and techniques to handle different datasets. The goal is to improve classification accuracy while exploring advanced evaluation methods and feature engineering.

## Key Concepts
- Binary and multi-class classification
- Model training and evaluation
- Feature engineering and scaling
- Advanced performance metrics (e.g., ROC-AUC, precision-recall curve)

## Code Highlights

### Data Preprocessing
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Model Training
```python
from sklearn.linear_model import LogisticRegression

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Predictions and Evaluation
```python
from sklearn.metrics import classification_report, roc_auc_score

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("ROC-AUC:", auc)
```

### Feature Importance
```python
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_
print("Feature Importances:", importances)
```

### Advanced Metrics
```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

# Plot precision-recall curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

## Highlights
- **Purpose**: Solve classification problems using a variety of models and techniques.
- **Learning Objectives**:
  - Implement preprocessing pipelines to handle real-world data.
  - Train and evaluate classification models effectively.
  - Use advanced metrics for thorough performance analysis.

---

[Back to Top](#overview)

## A4: YOLO Object Detection
### A4: Main Code

# A4: Main Code (A4.ipynb)

## Overview
This notebook orchestrates the YOLO-based object detection pipeline, integrating model training, evaluation, and inference workflows. It utilizes modules such as dataset preparation, model configuration, and YOLO loss computation.

## Key Components
1. **Dataset Preparation**: Loading and augmenting training and validation datasets.
2. **YOLO Model**: Built with ResNet50 as the backbone network.
3. **Loss Function**: Custom YOLO loss implemented in `yolo_loss.py`.
4. **Evaluation Metrics**: Mean Average Precision (mAP) computed using `eval_voc.py`.
5. **Inference**: Object detection on new images using `predict.py`.

## Code Highlights

### Loading the Dataset
```python
from src.dataset import VOC2007Dataset

train_dataset = VOC2007Dataset(root_dir="data/VOC2007", split="train")
val_dataset = VOC2007Dataset(root_dir="data/VOC2007", split="val")
```

### Model Initialization
```python
from src.resnet_yolo import resnet50

model = resnet50(pretrained=True)
model = model.cuda()
```

### Defining Loss Function
```python
from src.yolo_loss import YOLOLoss

criterion = YOLOLoss()
```

### Training Loop
```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images, targets = images.cuda(), targets.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

### Evaluation
```python
from src.eval_voc import evaluate_model

mAP = evaluate_model(model, val_loader)
print(f"Validation mAP: {mAP:.2f}")
```

### Inference
```python
from src.predict import predict_image

results = predict_image(model, image_name="000001.jpg", root_img_directory="data/VOC2007/JPEGImages/")
print(results)
```

## Highlights
- **Purpose**: Build and train a YOLO-based object detection model.
- **Learning Objectives**:
  - Understand how YOLO integrates grid-based localization and classification.
  - Use custom loss functions to optimize object detection tasks.
  - Evaluate models with mAP and apply inference to real-world images.

---

[Back to Top](#overview)

### YOLO LOSS

# YOLO Loss Implementation (yolo_loss.py)

## Overview
The YOLO loss function is critical for optimizing object detection performance. It balances the localization, confidence, and classification losses to train YOLO models effectively.

## Key Components
1. **Localization Loss**: Penalizes errors in predicted bounding box coordinates.
2. **Confidence Loss**: Measures the difference between predicted and actual objectness scores.
3. **Classification Loss**: Evaluates the error in predicted class probabilities.

## Code Highlights

### Loss Initialization
```python
import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per grid cell
        self.C = C  # Number of classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
```

### Forward Pass
```python
    def forward(self, predictions, targets):
        # Reshape predictions and targets for easier processing
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)
        targets = targets.view(-1, self.S, self.S, self.B * 5 + self.C)

        # Extract individual components from predictions and targets
        pred_boxes = predictions[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        pred_classes = predictions[..., self.B * 5:]
        target_boxes = targets[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        target_classes = targets[..., self.B * 5:]

        # Compute losses
        localization_loss = self.compute_localization_loss(pred_boxes, target_boxes)
        confidence_loss = self.compute_confidence_loss(pred_boxes, target_boxes)
        classification_loss = self.compute_classification_loss(pred_classes, target_classes)

        # Combine losses with weights
        total_loss = (
            self.lambda_coord * localization_loss +
            confidence_loss +
            self.lambda_noobj * classification_loss
        )
        return total_loss
```

### Loss Computation Methods
```python
    def compute_localization_loss(self, pred_boxes, target_boxes):
        # Calculate localization loss (MSE for box coordinates)
        coord_loss = nn.MSELoss()(pred_boxes[..., :2], target_boxes[..., :2])
        size_loss = nn.MSELoss()(torch.sqrt(pred_boxes[..., 2:4] + 1e-6), torch.sqrt(target_boxes[..., 2:4] + 1e-6))
        return coord_loss + size_loss

    def compute_confidence_loss(self, pred_boxes, target_boxes):
        # Calculate confidence loss (binary cross-entropy or MSE)
        conf_loss = nn.MSELoss()(pred_boxes[..., 4], target_boxes[..., 4])
        return conf_loss

    def compute_classification_loss(self, pred_classes, target_classes):
        # Calculate classification loss (cross-entropy)
        class_loss = nn.CrossEntropyLoss()(pred_classes.view(-1, self.C), target_classes.view(-1, self.C))
        return class_loss
```

## Highlights
- **Purpose**: Combine localization, confidence, and classification losses for YOLO training.
- **Learning Objectives**:
  - Understand how to design and implement custom loss functions.
  - Balance multiple objectives in training object detection models.

---

[Back to Top](#overview)
### Configuration File
# Configuration File (config.py)

## Overview
This configuration file centralizes the settings for the YOLO-based object detection system. It includes parameters for dataset paths, model configurations, and training hyperparameters.

## Key Components
1. **Paths**: Defines paths to datasets, annotations, and outputs.
2. **Model Parameters**: Configurations for grid size, bounding boxes, and classes.
3. **Training Hyperparameters**: Learning rate, batch size, and epochs.

## Code Highlights

### Dataset Paths
```python
data_config = {
    "root_dir": "data/VOC2007",
    "train_images": "data/VOC2007/JPEGImages",
    "train_annotations": "data/VOC2007/Annotations",
    "val_images": "data/VOC2007/JPEGImages",
    "val_annotations": "data/VOC2007/Annotations",
}
```

### Model Parameters
```python
model_config = {
    "grid_size": 7,       # YOLO grid size (SxS)
    "num_boxes": 2,       # Number of bounding boxes per grid cell
    "num_classes": 20,    # Number of object classes
}
```

### Training Hyperparameters
```python
train_config = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 50,
    "weight_decay": 5e-4,
    "momentum": 0.9,
}
```

### Output Settings
```python
output_config = {
    "checkpoints": "checkpoints/",
    "logs": "logs/",
    "results": "results/",
}
```

## Highlights
- **Purpose**: Centralize configuration to simplify parameter management.
- **Learning Objectives**:
  - Understand how configurations influence model training and evaluation.
  - Modify parameters dynamically to adapt to different datasets or tasks.

---

[Back to Top](#overview)

### Dataset Processing
# Dataset Processing (dataset.py)

## Overview
This module handles dataset loading, preprocessing, and augmentation for YOLO-based object detection. It ensures that data is properly structured for training and evaluation.

## Key Components
1. **Dataset Loading**: Reads images and annotations from VOC2007 dataset.
2. **Data Augmentation**: Applies transformations like scaling, flipping, and cropping to enhance model generalization.
3. **Bounding Box Management**: Converts annotations into normalized grid-based representations required by YOLO.

## Code Highlights

### Dataset Class Definition
```python
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class VOC2007Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load image file names and annotation paths
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.annotation_dir = os.path.join(root_dir, "Annotations")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace(".jpg", ".xml"))

        # Load image and annotations
        image = Image.open(image_path).convert("RGB")
        boxes, labels = self.parse_annotations(annotation_path)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        return image, (boxes, labels)
```

### Parsing Annotations
```python
import xml.etree.ElementTree as ET

def parse_annotations(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox = [
            int(bndbox.find("xmin").text),
            int(bndbox.find("ymin").text),
            int(bndbox.find("xmax").text),
            int(bndbox.find("ymax").text),
        ]
        boxes.append(bbox)
        labels.append(label)

    return torch.tensor(boxes, dtype=torch.float32), labels
```

### Data Augmentation Example
```python
from torchvision import transforms

def transform(image, boxes):
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image, boxes
```

## Highlights
- **Purpose**: Prepares and augments data for YOLO training.
- **Learning Objectives**:
  - Understand how to load and preprocess object detection datasets.
  - Apply data augmentation to improve model robustness.
  - Convert bounding box annotations to YOLO-compatible formats.

---

[Back to Top](#overview)

### VOC Evaluation
# VOC Evaluation (eval_voc.py)

## Overview
This module evaluates the performance of the YOLO model on the VOC dataset. It calculates the mean Average Precision (mAP), a key metric for object detection tasks, by comparing predicted bounding boxes with ground truth annotations.

## Key Components
1. **Bounding Box Matching**: Matches predicted boxes with ground truth boxes using IoU (Intersection over Union).
2. **Precision and Recall**: Computes precision-recall pairs for different confidence thresholds.
3. **mAP Calculation**: Calculates the mean Average Precision across all object classes.

## Code Highlights

### IoU Calculation
```python
import torch

def compute_iou(box1, box2):
    # Compute the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Area of intersection
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU calculation
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou
```

### Precision-Recall Calculation
```python
import numpy as np

def compute_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

    tp = []  # True positives
    fp = []  # False positives

    matched = set()

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(true_boxes):
            iou = compute_iou(pred_box[:4], gt_box[:4])
            if iou > best_iou and gt_idx not in matched:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    precisions = tp / (tp + fp + 1e-6)
    recalls = tp / len(true_boxes)
    return precisions, recalls
```

### mAP Calculation
```python
def compute_map(predictions, ground_truths, iou_threshold=0.5):
    average_precisions = []

    for class_idx in range(num_classes):
        pred_boxes = [pred for pred in predictions if pred[5] == class_idx]
        true_boxes = [gt for gt in ground_truths if gt[5] == class_idx]

        precisions, recalls = compute_precision_recall(pred_boxes, true_boxes, iou_threshold)
        ap = np.trapz(precisions, recalls)  # Compute area under precision-recall curve
        average_precisions.append(ap)

    return np.mean(average_precisions)
```

### Highlights
- **Purpose**: Evaluate the YOLO model using precision-recall metrics and calculate mAP.
- **Learning Objectives**:
  - Implement IoU-based matching for object detection.
  - Compute precision-recall curves and derive mAP.
  - Use mAP as a benchmark for model performance.

---

[Back to Top](#overview)
### Prediction Script
# Prediction Script (predict.py)

## Overview
This script performs object detection on a single image or a batch of images using the trained YOLO model. It includes preprocessing, model inference, and postprocessing steps to display bounding boxes and class labels on the images.

## Key Components
1. **Image Preprocessing**: Resizes and normalizes input images.
2. **Model Inference**: Runs the trained YOLO model to predict bounding boxes and class probabilities.
3. **Postprocessing**: Applies non-maximum suppression (NMS) to filter overlapping boxes and overlays results on the image.

## Code Highlights

### Image Preprocessing
```python
import cv2
import numpy as np

def preprocess_image(image_path, input_size=(448, 448)):
    image = cv2.imread(image_path)
    original_size = image.shape[:2]  # Original height and width
    image = cv2.resize(image, input_size)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW format
    return np.expand_dims(image, axis=0), original_size
```

### Model Inference
```python
def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions
```

### Non-Maximum Suppression (NMS)
```python
def non_maximum_suppression(predictions, iou_threshold=0.5):
    # Filter boxes based on confidence threshold
    boxes = [box for box in predictions if box[4] > 0.5]
    
    # Sort boxes by confidence
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    final_boxes = []
    while boxes:
        best_box = boxes.pop(0)
        final_boxes.append(best_box)

        boxes = [box for box in boxes if compute_iou(best_box[:4], box[:4]) < iou_threshold]

    return final_boxes
```

### Drawing Bounding Boxes
```python
def draw_boxes(image, boxes, labels):
    for box in boxes:
        x1, y1, x2, y2, confidence, class_idx = box
        label = labels[int(class_idx)]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
```

### Full Prediction Workflow
```python
def predict_image(model, image_path, labels, input_size=(448, 448)):
    # Preprocess image
    image_tensor, original_size = preprocess_image(image_path, input_size)
    
    # Run inference
    predictions = predict(model, torch.tensor(image_tensor).cuda())

    # Apply NMS
    final_boxes = non_maximum_suppression(predictions.cpu().numpy())

    # Draw boxes
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, original_size[::-1])
    result_image = draw_boxes(original_image, final_boxes, labels)
    
    return result_image
```

## Highlights
- **Purpose**: Perform object detection on new images using a trained YOLO model.
- **Learning Objectives**:
  - Implement preprocessing, inference, and postprocessing steps for object detection.
  - Apply NMS to refine predictions.
  - Overlay results on images for visualization.

---

[Back to Top](#overview)
### YOLO with ResNet Backbone
# YOLO with ResNet Backbone (resnet_yolo.py)

## Overview
This module implements the YOLO object detection network with ResNet50 as the backbone. It combines ResNet's feature extraction capabilities with YOLO's grid-based localization and classification.

## Key Components
1. **ResNet Backbone**: A pre-trained ResNet50 model is used to extract features from input images.
2. **YOLO Head**: Custom layers are added to predict bounding boxes and class probabilities.
3. **Forward Pass**: Defines how the input propagates through the network to produce predictions.

## Code Highlights

### Model Initialization
```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNetYOLO(nn.Module):
    def __init__(self, num_classes=20, num_boxes=2, grid_size=7):
        super(ResNetYOLO, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Load pre-trained ResNet50
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc layers

        # YOLO Head
        self.yolo_head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, grid_size * grid_size * (num_boxes * 5 + num_classes), kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.yolo_head(x)
        return x.view(-1, self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes)
```

### Output Reshaping
```python
def reshape_predictions(predictions, grid_size=7, num_boxes=2, num_classes=20):
    # Reshape to match YOLO output format
    batch_size = predictions.size(0)
    predictions = predictions.view(batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
    return predictions
```

### Feature Extraction
```python
def extract_features(model, image_tensor):
    # Extract features using ResNet backbone
    features = model.features(image_tensor)
    return features
```

## Highlights
- **Purpose**: Integrate ResNet's robust feature extraction with YOLO's detection capabilities.
- **Learning Objectives**:
  - Understand how to modify pre-trained models for custom tasks.
  - Combine feature extraction and detection layers.
  - Implement forward propagation for object detection.

---

[Back to Top](#overview)
