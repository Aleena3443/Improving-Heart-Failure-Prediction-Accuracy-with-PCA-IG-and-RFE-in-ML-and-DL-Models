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
│   ├── 1_preprocessing.ipynb
│   ├── 2_model_no_fs.ipynb
│   ├── 3_model_pca.ipynb
│   ├── 4_model_ig.ipynb
│   └── 5_model_rfe.ipynb
├── models/
│   └── saved_models/
├── results/
│   └── performance_metrics.csv
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

| Model | Feature Selection | Accuracy | AUC |
|-------|-------------------|----------|-----|
| LSTM  | None              | 0.99     | 0.9992 |
| ANN   | PCA               | 0.98     | 0.9983 |
| LR    | IG                | 0.64     | 0.8793 |


