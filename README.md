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
├── models/                # Saved models and scaler artifacts (scaler_amount.joblib, scaler_time.joblib)
├── notebooks/             # Sequential Jupyter Notebooks (01 through 09)
├── reports/
│   └── figures/           # Generated EDA and evaluation visualizations
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
| **05** | `05_Baseline_Models.ipynb` | ⏳ Pending | Training initial baseline models (Logistic Regression, Random Forest, XGBoost) on un-resampled vs. SMOTE training data. |
| **06** | `06_Imbalanced_Learning.ipynb` | ⏳ Pending | Exploring advanced imbalanced techniques (Cost-Sensitive Learning, Focal Loss, Balanced Random Forest, Random UnderSampling). |
| **07** | `07_Model_Evaluation.ipynb` | ⏳ Pending | Comprehensive evaluation: Precision-Recall Curves, ROC Curves, Confusion Matrices, and business impact analysis. |
| **08** | `08_Threshold_Tuning.ipynb` | ⏳ Pending | Optimal decision threshold selection based on financial loss minimization and target precision/recall trade-offs. |
| **09** | `09_Final_Model.ipynb` | ⏳ Pending | Final model pipeline assembly, feature importance analysis, model serialization, and deployment readiness review. |

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

### Exported Artifacts:
- **`data/processed/`**: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `X_train_resampled.csv`, `y_train_resampled.csv`
- **`models/`**: `scaler_amount.joblib`, `scaler_time.joblib`

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
