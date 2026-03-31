# 💳 Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📌 Problem Statement

Credit card fraud detection is a **highly imbalanced classification problem**, where fraudulent transactions account for only **~0.17%** of the dataset.

This creates several challenges:

* Accuracy becomes **misleading**
* Models tend to bias toward the majority class (non-fraud)
* Difficult trade-off between:

  * **High Recall** → catching as many fraud cases as possible
  * **High Precision** → avoiding false alarms for legitimate users

👉 **Objective**:
Build a robust fraud detection system that:

* Maximizes **fraud detection (Recall)**
* Maintains **high alert accuracy (Precision)**

---

## 📊 Dataset

* **Source**: European cardholders dataset
* **Total transactions**: 284,807
* **Fraud ratio**: 0.17%

### Features:

* 28 anonymized features (V1–V28) transformed using PCA
* `Amount`: transaction value
* `Class`: target label (0 = normal, 1 = fraud)

👉 Key characteristics:

* Features are **already PCA-transformed**
* Dataset is **extremely imbalanced**

---

## 🏗️ Project Structure

```
project/
├── data/
│   └── raw/                  # Raw dataset (creditcard.csv)
├── models/                   # Trained artifacts
│   ├── scaler.joblib         # StandardScaler
│   └── best_model.joblib     # Random Forest model
├── notebooks/
│   └── exploration.ipynb     # EDA and experiments
├── src/
│   ├── train.py              # Training pipeline
│   └── predict.py            # Inference pipeline
└── README.md
```

---

## ⚙️ Approach & Methodology

### 1. Exploratory Data Analysis (EDA)

* Analyzed class distribution → confirmed severe imbalance
* Explored `Amount` and PCA features
* Identified abnormal patterns in fraudulent transactions

---

### 2. Data Preprocessing

* Applied **StandardScaler**
* Used **Stratified Train-Test Split**

---

### 3. Sampling Strategy

Used:

* **SMOTE (Synthetic Minority Over-sampling Technique)**

👉 Purpose:

* Increase minority class samples (fraud)
* Preserve information from majority class

⚠️ **Important Lesson**:

* DO NOT apply SMOTE before Cross-validation
* Use:

```python
from imblearn.pipeline import Pipeline
```

→ to avoid **Data Leakage**

---

### 4. Model Selection

Tested models:

* Logistic Regression
* Random Forest (Best)
* (Future) XGBoost

👉 Final model:

* **Random Forest Classifier**
* Tuned using **GridSearchCV**

---

### 5. Evaluation Metrics

Due to imbalance, we use:

* Precision
* Recall
* F1-score
* **AUC-PR (Average Precision Score)**

👉 Accuracy is **not used**

---

## 📈 Results

| Model               | Sampling                  | Precision | Recall   | F1       | AUC-PR   |
| ------------------- | ------------------------- | --------- | -------- | -------- | -------- |
| Logistic Regression | Undersampling             | 0.49      | 0.12     | 0.19     | 0.49     |
| Random Forest       | SMOTE                     | **0.90**  | **0.78** | **0.83** | **0.82** |
| XGBoost             | scale_pos_weight          | (future)  | (future) | (future) | (future) |
| **Best Model**      | **Random Forest + SMOTE** | **0.90**  | **0.78** | **0.83** | **0.82** |

---

## 📊 Key Visualizations

Include in notebook or README:

* Confusion Matrix
* Precision-Recall Curve
* Feature Importance

---

## 🚀 How to Run

### 1. Installation

```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib matplotlib seaborn
```

---

### 2. Training

```bash
python src/train.py
```

---

### 3. Inference

Update input data in `predict.py`, then run:

```bash
python src/predict.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn
* Joblib
* Matplotlib, Seaborn

---

## 💡 Key Insights

* ⚠️ **Data Leakage Pitfall**
  Applying SMOTE before Cross-validation leads to incorrect evaluation
  → Fixed using Pipeline

* 🎯 **Metric Selection Matters**
  Accuracy is misleading → switched to **Precision-Recall & AUC-PR**

* 🔍 **Feature Importance**

  * `V14` and `V12` are the most influential features

* 🧠 **Model Behavior**
  Random Forest captures non-linear fraud patterns effectively

* 📉 **Baseline vs Model**

  * Baseline AUC-PR: ~0.0017
  * Model AUC-PR: **0.82**

---

## 🔬 Error Analysis

Observed weakness:

* **Micro-fraud transactions**
* Average Amount ≈ **-0.33 (scaled)**

👉 Insight:

* Small transactions are harder to distinguish from normal behavior

---

## 🔮 Future Improvements

* Add **behavioral features**:

  * Transaction frequency
  * Time-based patterns

* Try advanced models:

  * XGBoost
  * LightGBM

* Apply **cost-sensitive learning**

  * Penalize False Negatives more heavily

* Build **real-time detection pipeline**

---

## 👤 Author

**Le Quang Anh Tuan**
AI Engineer Roadmap

GitHub: *lequanganhtuan*

---

## 📌 Summary

> This project highlights that in extreme imbalance scenarios, **data handling, evaluation metrics, and pipeline design are more critical than the model itself**.
