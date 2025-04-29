Improving Heart Failure Prediction Accuracy with PCA, Information Gain, and Recursive Feature Elimination

This repository contains the complete implementation and findings from a research project focused on enhancing the accuracy of heart failure prediction. The study evaluates the impact of feature selection techniques—Principal Component Analysis (PCA), Information Gain (IG), and Recursive Feature Elimination (RFE)—on both traditional machine learning (ML) and deep learning (DL) models.

## 📊 Project Overview

**Goal**: Improve heart failure prediction by applying PCA, IG, and RFE for feature selection and evaluating their impact on ML (Logistic Regression, Random Forest) and DL (LSTM, ANN) models.

**Dataset**: [ECG Arrhythmia Classification Dataset] data source for this dissertation is obtained from the below link
(https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset) 


**Key Findings**:
- **Best Model**: LSTM without feature selection achieved 0.99 accuracy and 0.9992 AUC.
- **Best Feature Selector**: PCA maintained strong performance with DL models while reducing dimensionality.
- Traditional ML models performed worse with feature selection, especially with IG.

## 🧠 Models Used

- Logistic Regression (LR)
- Random Forest (RF)
- Long Short-Term Memory (LSTM)
- Artificial Neural Network (ANN)

## 🔍 Feature Selection Techniques

- **PCA** – Dimensionality reduction while preserving variance.
- **Information Gain** – Selects features that provide the most information about the target.
- **Recursive Feature Elimination (RFE)** – Iteratively removes less important features.

## 🧪 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

## 🛠 Tools and Libraries

- Python (Pandas, NumPy, Scikit-learn, TensorFlow, Keras)
- SMOTE (imbalanced-learn) for class balancing
- Matplotlib & Seaborn for visualization

## 🧬 Project Structure

```
📁 Heart-Failure-Prediction/
├── data/
│   └── HeartFailure2Processed.csv
├── notebooks/
│   ├── a)Preprocess of Heart Failure.ipynb
│   ├── b)Without Feature Selection of Heart Failure.ipynb
│   ├── c)PCA Feature Selection of Heart Failure.ipynb
│   ├── d)Information Gain Feature Selection of Heart Failure.ipynb
│   └── e)RFE Feature Selection of Heart Failure.ipynb
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Running the Project

1. Preprocess the data:
   ```bash
   python preprocess.py
   ```

2. Train models with and without feature selection:
   ```bash
   python train_models.py
   ```

3. Evaluate model performance:
   ```bash
   python evaluate.py
   ```

## 📈 Results

| Feature Selection | Model | Validation Accuracy | Testing Accuracy | Validation AUC | Testing AUC |
|-------------------|-------|----------------------|-------------------|----------------|-------------|
| No FS             | LR    | 0.86                 | 0.86              | 0.9706         | 0.9710      |
|                   | RF    | 0.86                 | 0.86              | 0.9706         | 0.9710      |
|                   | LSTM  | 0.99                 | 0.99              | 0.9992         | 0.9992      |
|                   | ANN   | 0.97                 | 0.98              | 0.9984         | 0.9985      |
| PCA               | LR    | 0.83                 | 0.83              | 0.9476         | 0.9487      |
|                   | RF    | 0.83                 | 0.83              | 0.9476         | 0.9487      |
|                   | LSTM  | 0.98                 | 0.98              | 0.9988         | 0.9989      |
|                   | ANN   | 0.98                 | 0.98              | 0.9981         | 0.9983      |
| IG                | LR    | 0.64                 | 0.64              | 0.8782         | 0.8793      |
|                   | RF    | 0.64                 | 0.64              | 0.8782         | 0.8793      |
|                   | LSTM  | 0.90                 | 0.90              | 0.9876         | 0.9877      |
|                   | ANN   | 0.88                 | 0.88              | 0.9833         | 0.9835      |
| RFE               | LR    | 0.73                 | 0.73              | 0.9265         | 0.9256      |
|                   | RF    | 0.73                 | 0.73              | 0.9265         | 0.9256      |
|                   | LSTM  | 0.89                 | 0.89              | 0.9855         | 0.9855      |
|                   | ANN   | 0.94                 | 0.94              | 0.9927         | 0.9926      |


