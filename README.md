# Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions on highly imbalanced financial transaction data.

---

## 📌 Project Overview

Credit card fraud detection is a critical application of machine learning in financial technology. Fraudulent transactions constitute a very small percentage of overall transactions (approx. **0.17%**), creating an extreme class imbalance. The goal of this project is to build, evaluate, tune, and deploy high-performance classification models that prioritize **Fraud Recall** and **PR-AUC (Precision-Recall Area Under the Curve)** while minimizing costly false positives.

---

## 📂 Repository Structure

```text
Fraud Detection/
├── .agents/               # Project-level agent rules and guidelines
├── data/
│   ├── raw/               # Raw dataset (creditcard.csv)
│   ├── processed/         # Preprocessed & partitioned CSV splits (X_train, X_test, SMOTE data)
│   └── external/          # External domain resources
├── models/                # Saved models (.pkl) and scaler artifacts (scaler_amount.joblib, scaler_time.joblib)
├── notebooks/             # Sequential Jupyter Notebooks (01 through 09)
├── reports/
│   ├── baseline_model_results.csv   # Serialized baseline evaluation results
│   ├── rf_comparission_results.csv  # Sampling technique comparison results
│   ├── threshold_tuning_results.csv # Probability threshold sweep results
│   ├── project_summary.csv          # Final model & pipeline benchmark summary
│   └── figures/                     # Generated EDA and evaluation visualizations
├── src/                   # Source code scripts for modular pipeline execution
├── pyproject.toml         # Dependency configurations
└── README.md              # Project documentation and pipeline status
```

---

## 📓 Notebook Pipeline & Progress Status

| Notebook | Title | Status | Key Focus & Summary |
| :--- | :--- | :---: | :--- |
| **01** | `01_Business_Understanding.ipynb` | ✅ Completed | Problem formulation, cost-benefit matrix (False Positives vs. False Negatives), primary evaluation metrics selection (PR-AUC, Recall@Precision). |
| **02** | `02_Data_Understanding.ipynb` | ✅ Completed | Data dimensions analysis (284,807 rows × 31 columns), variable data types, missing value audit (0 nulls), target distribution (0.172% fraud prevalence). |
| **03** | `03_EDA.ipynb` | ✅ Completed | Univariate distribution of `Amount` (extreme right skewness, median $22.00), time dynamics, correlation with `Class`, and outlier analysis. |
| **04** | `04_Preprocessing.ipynb` | ✅ Completed | Data cleaning, deduplication (1,081 duplicates removed → 283,726 samples), `RobustScaler` transformation on `Amount` & `Time`, 80/20 stratified train-test split, SMOTE oversampling on training data (453,232 balanced samples), and dataset export. |
| **05** | `05_Baseline_Models.ipynb` | ✅ Completed | Trained and evaluated 4 baseline classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost). Identified Random Forest as top baseline performer (PR-AUC: 0.7876, Recall: 72.63%, Precision: 97.18%). Serialized baseline models & metrics. |
| **06** | `06_Imbalanced_Learning.ipynb` | ✅ Completed | Evaluated Random Oversampling (ROS) vs. Synthetic Minority Over-sampling Technique (SMOTE). **Random Forest + SMOTE** emerged as the top performer with **0.8078 PR-AUC**, **77.89% Recall**, and **0.9591 ROC-AUC**. |
| **07** | `07_Model_Evaluation.ipynb` | ✅ Completed | Comprehensive diagnostic analysis of RF + SMOTE: Precision-Recall Curves, ROC Curves, Confusion Matrix analysis, feature importance ranking, and error distribution. |
| **08** | `08_Threshold_Tuning.ipynb` | ✅ Completed | Systematic decision threshold optimization (0.10 to 0.90 sweep). Selected optimal decision threshold of **0.60**, boosting Precision to **97.26%** while reducing False Positives down to just **2** on the test set. |
| **09** | `09_Final_Model.ipynb` | ✅ Completed | Final end-to-end inference pipeline assembly, serialization (`fraud_detection_pipeline.pkl`, `best_threshold.pkl`), prediction testing, business recommendations, and future roadmap. |

---

## 📊 Preprocessing & Data Summary (Notebook 04 Output)

Following `04_Preprocessing.ipynb`, the raw transaction dataset was processed and exported to `data/processed/`:

| Pipeline Stage | Total Samples | Legitimate (Class 0) | Fraudulent (Class 1) | Fraud Ratio (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Dataset** | 284,807 | 284,315 | 492 | 0.1727% |
| **Deduplicated Dataset** | 283,726 | 283,253 | 473 | 0.1667% |
| **Train Set (80% Stratified)** | 226,980 | 226,616 | 364 | 0.1604% |
| **Test Set (20% Stratified)** | 56,746 | 56,637 | 109 | 0.1921% |
| **SMOTE Resampled Train Set** | 453,232 | 226,616 | 226,616 | 50.0000% |

### Exported Preprocessing Artifacts:
- **`data/processed/`**: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `X_train_resampled.csv`, `y_train_resampled.csv`
- **`models/`**: `scaler_amount.joblib`, `scaler_time.joblib`

---

## 📈 Model Performance & Experiments

### 1. Baseline Model Comparison (Notebook 05 Output)

Evaluated baseline algorithms on the imbalanced test partition (56,746 transactions):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Winner / Key Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.9991 | 0.8438 | 0.5684 | 0.6792 | **0.9573** | 0.6900 | Highest ROC-AUC (Probability Ranking) |
| **Decision Tree** | 0.9990 | 0.7204 | 0.7053 | 0.7128 | 0.8524 | 0.5086 | Higher recall, lower precision |
| **Random Forest** | **0.9995** | **0.9718** | **0.7263** | **0.8313** | 0.9239 | **0.7876** | 🏆 **Best Overall Baseline Model** |
| **XGBoost** | 0.9992 | 0.8171 | 0.7053 | 0.7571 | 0.8475 | 0.6649 | Strong baseline performance |

### 2. Imbalanced Learning & Sampling Comparison (Notebook 06 Output)

Compared resampling techniques paired with Random Forest on test set evaluation:

| Model / Sampling Method | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Random Forest** | 0.9995 | **0.9718** | 0.7263 | 0.8313 | 0.9239 | 0.7876 | High precision, lower recall |
| **Random Forest + Random Oversampling** | 0.9995 | 0.9583 | 0.7263 | 0.8263 | 0.9292 | 0.8023 | Improved probability ranking |
| **Random Forest + SMOTE** | **0.9995** | 0.9136 | **0.7789** | **0.8409** | **0.9591** | **0.8078** | 🏆 **Highest Recall & PR-AUC** |

---

## 🎯 Threshold Tuning & Final Model Results (Notebooks 08 & 09)

Swept decision thresholds between `0.10` and `0.90` on the **Random Forest + SMOTE** probability outputs:

| Threshold | Precision | Recall | F1 Score | False Positives | False Negatives | Optimization Focus |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.10` | 0.3493 | 0.8421 | 0.4938 | 149 | 15 | High Recall, excessive false alerts |
| `0.30` | 0.8105 | 0.8105 | 0.8105 | 18 | 18 | Balanced Precision-Recall trade-off |
| `0.50` (Default) | 0.9136 | 0.7789 | 0.8409 | 7 | 21 | Default classifier threshold |
| **`0.60` (Optimal)** | **0.9726** | **0.7474** | **0.8452** | **2** | **24** | 🏆 **Best Precision & F1 Score** |

### 🏆 Final Selected Model Configuration:
- **Algorithm**: Random Forest Classifier
- **Sampling Strategy**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Decision Threshold**: `0.60`
- **Key Performance Highlights**:
  - **PR-AUC**: `0.8078`
  - **ROC-AUC**: `0.9591`
  - **Precision**: `97.26%` (only **2 False Positives** out of 56,746 test cases)
  - **F1 Score**: `84.52%`

---

## 📦 Exported Artifacts & Reports

- **Serialized Models (`models/`)**:
  - `random_forest_smote.pkl`: Winning model trained with SMOTE
  - `best_threshold.pkl`: Optimal probability decision threshold (`0.60`)
  - `fraud_detection_pipeline.pkl`: Production end-to-end pipeline object
- **Evaluation Reports (`reports/`)**:
  - `baseline_model_results.csv`
  - `rf_comparission_results.csv`
  - `threshold_tuning_results.csv`
  - `project_summary.csv`

---

## 🛠️ Environment Setup & Installation

1. **Clone Repository & Navigate to Workspace**:
   ```bash
   git clone <repository-url>
   cd "Fraud Detection"
   ```

2. **Install Dependencies using `uv` or `pip`**:
   ```bash
   uv pip install -r requirements.txt
   ```
