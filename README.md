# Kaggle: Predicting Airline Passenger Satisfaction

An end-to-end Machine Learning solution for Kaggle's **Airline Passenger Satisfaction** binary classification competition. This repository provides a complete pipeline with domain-specific feature engineering, systematic hyperparameter tuning for diverse model families (**Random Forest**, **XGBoost**, and **LightGBM**), stratified cross-validation, and multi-model ensembling designed to maximize the **ROC-AUC** evaluation metric.

---

## 📌 Project Overview

- **Task**: Binary Classification (`satisfaction` $\in \{0, 1\}$)
- **Dataset Size**:
  - `train.csv`: 699,635 rows $\times$ 23 columns
  - `test.csv`: 299,844 rows $\times$ 22 columns
- **Target Distribution**: ~44.36% Satisfied (`1`), ~55.64% Neutral/Dissatisfied (`0`)
- **Evaluation Metric**: Area Under the ROC Curve (**ROC-AUC**)
- **Best Out-of-Fold (OOF) Score**: **0.95868** (Tuned LightGBM Deep & SLSQP Optimal Blend)

---

## 🏗️ Architecture & Pipeline Workflow

```
Raw Data (train.csv & test.csv)
   │
   ▼
[1] Contextual Imputation & Missing Value Tracking
   │  └── Arrival Delay missingness indicator & contextual imputation from Departure Delay
   ▼
[2] Domain Feature Engineering (42 Features)
   │  ├── Delay dynamics (Total Delay, Delay Diff, Log Delays, Delay Ratios)
   │  ├── Survey score aggregations (Mean, Std, Min, Max, Zero-counts, Sentiment extremes)
   │  ├── Composite sub-dimension indices (Comfort, Inflight Service, Booking Convenience)
   │  └── Demographic ratios & strict leak-free Ordinal Encoding
   ▼
[3] Systematic Hyperparameter Tuning (Stratified ROC-AUC Search)
   │  ├── Random Forest: Depth, min_samples_split, bagging subsamples
   │  ├── XGBoost: Max depth, learning rate, subsample, colsample_bytree, L1/L2 penalties
   │  └── LightGBM: Number of leaves, learning rate, colsample_bytree, min_child_samples
   ▼
[4] Stratified K-Fold Cross-Validation (5 Folds)
   │  ├── Model 1: Tuned LightGBM GBDT
   │  ├── Model 2: Deep Leaves LightGBM (num_leaves=127)
   │  ├── Model 3: Tuned XGBoost (Histogram Tree Method)
   │  ├── Model 4: Regularized XGBoost (L1/L2 Shrinkage)
   │  ├── Model 5: Tuned Random Forest (Bagged Ensembles)
   │  └── Model 6: HistGradientBoostingClassifier (Scikit-Learn)
   ▼
[5] Advanced Ensembling & ROC-AUC Optimization
   │  ├── Ensemble 7: Weighted Top 2 GBDT Blend (LightGBM + XGBoost)
   │  ├── Ensemble 8: Uniform Multi-Family Average (LGBM + XGB + RF + HistGB)
   │  ├── Ensemble 9: Percentile Rank-Averaged Ensemble
   │  └── Ensemble 10: SLSQP Out-of-Fold Convex Optimization
   ▼
[6] Generation of 10 Distinct, Validated Submission Files
```

---

## 🛠️ Feature Engineering Details

1. **Contextual Delay Imputation**:
   - `Arrival Delay in Minutes` is the only column with missing records (~0.04%).
   - Missing values are imputed using `Departure Delay in Minutes` (correlation $> 0.96$), followed by the training-set median fallback.
   - Added a dedicated missingness indicator flag: `Arrival_Delay_isna`.

2. **Delay Metrics**:
   - `Total_Delay = Departure Delay + Arrival Delay`
   - `Delay_Diff = Arrival Delay - Departure Delay` (delay mitigation or worsening in flight)
   - `Log_Dep_Delay = ln(1 + max(0, Departure Delay))`
   - `Log_Arr_Delay = ln(1 + max(0, Arrival Delay))`
   - `Is_Delayed = 1 if Departure or Arrival Delay > 0 else 0`
   - `Delay_Ratio = (Total_Delay + 1) / (Flight Distance + 1)`

3. **Survey Rating Aggregations (13 Survey Categories, 0–5 scale)**:
   - `Rating_Mean`, `Rating_Std`, `Rating_Sum`, `Rating_Min`, `Rating_Max`
   - `Rating_Zero_Count`: Count of `0` ratings (represents "Not Applicable" / unrated experiences)
   - `Rating_Low_Count`: Count of ratings $\le 2$ (dissatisfaction flags)
   - `Rating_High_Count`: Count of ratings $\ge 4$ (satisfaction signals)
   - **Composite Sub-indices**:
     - `Comfort_Mean`: Mean of *Seat comfort*, *Leg room service*, *Cleanliness*
     - `Service_Mean`: Mean of *Inflight wifi service*, *Food and drink*, *Inflight entertainment*, *On-board service*, *Baggage handling*, *Checkin service*
     - `Convenience_Mean`: Mean of *Departure/Arrival time convenient*, *Ease of Online booking*, *Gate location*, *Online boarding*

4. **Categorical & Demographic Interactions**:
   - `Flight_Distance_per_Age = Flight Distance / (Age + 1)`
   - `Class_TravelType = Class + '_' + Type of Travel` (e.g. `Business_Business travel` vs `Eco_Personal Travel`)
   - `Customer_Class = Customer Type + '_' + Class`

---

## 📊 Models & 10 Distinct Submissions Summary

Each submission file contains all **299,844 test rows** formatted with columns `id` and `satisfaction` (predicted probability of satisfaction $\in [0.0, 1.0]$):

| Filename | OOF ROC-AUC | Mean Pred | Description |
| :--- | :---: | :---: | :--- |
| `submission_01_lightgbm_tuned.csv` | **0.95787** | 0.4439 | Tuned LightGBM GBDT with optimized leaves and feature subsampling |
| `submission_02_lightgbm_deep.csv` | **0.95868** | 0.4437 | LightGBM with deep leaves (`num_leaves=127`) capturing high-order non-linearities |
| `submission_03_xgboost_tuned.csv` | **0.95702** | 0.4439 | Tuned XGBoost with histogram tree method and depth-wise splitting |
| `submission_04_xgboost_regularized.csv` | **0.95831** | 0.4438 | XGBoost with conservative learning rate, higher $L_1/L_2$ penalties & subsampling |
| `submission_05_random_forest_tuned.csv` | **0.95532** | 0.4445 | Tuned Random Forest with bagged tree sub-sampling and leaf regularization |
| `submission_06_hist_gradient_boosting.csv` | **0.95831** | 0.4439 | Scikit-Learn `HistGradientBoostingClassifier` with native histogram binning |
| `submission_07_blend_lightgbm_xgboost.csv` | **0.95766** | 0.4439 | Weighted blend of top gradient boosted trees ($55\%\text{ LGBM} + 45\%\text{ XGB}$) |
| `submission_08_multi_family_average.csv` | **0.95765** | 0.4440 | Equal uniform average of 4 distinct model families ($\text{LGBM} + \text{XGB} + \text{RF} + \text{HistGB}$) |
| `submission_09_rank_averaged_ensemble.csv` | **0.95834** | 0.5000 | Rank-averaged ensemble of all models (robust against probability calibration skew) |
| `submission_10_optimal_roc_auc_blend.csv` | **0.95868** | 0.4438 | Directly optimized SLSQP ensemble weights maximizing Out-Of-Fold ROC-AUC |

---

## 🚀 Quickstart & Usage

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/nikhilk2151/Kaggle-Predicting-Airline-Satisfaction.git
cd Kaggle-Predicting-Airline-Satisfaction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data
Download `train.csv` and `test.csv` from Kaggle and place them in the root directory.

### 3. Run Pipeline

The script is fully optimized for **Kaggle Competition Environments, Colab Notebooks, and Local Environments**. It automatically detects and formats final outputs using `sample_submission.csv`:
- Competition Mount: `/kaggle/input/competitions/playground-series-s6e10/`
- Dataset Mount: `/kaggle/input/playground-series-s6e10/`
- Output Directory: `/kaggle/working/submissions/` (or `./submissions/` locally)
- Zero `argparse` conflicts with Jupyter/Colab notebook kernels.

```python
# In Kaggle / Colab Notebook cell:
%run airline_satisfaction_pipeline.py

# Or directly in Python:
from airline_satisfaction_pipeline import main
main(quick_mode=True, n_splits=3)
```

Or from terminal:
```bash
python airline_satisfaction_pipeline.py
```

Generated submissions and the summary report will be saved automatically in the `submissions/` directory.

---

## 📦 Repository Structure
```
.
├── airline_satisfaction_pipeline.py  # Complete end-to-end training and inference script
├── requirements.txt                  # Python package dependencies
├── README.md                         # Project documentation and benchmarks
├── .gitignore                        # Git ignore file
└── submissions/
    └── submissions_summary.csv       # Summary table with OOF ROC-AUC scores & statistics
```
