"""
Kaggle Airline Passenger Satisfaction - Binary Classification Pipeline
========================================================================
Maximizing ROC-AUC with:
- Robust missing value handling & domain-specific feature engineering
- Categorical encoding fitted strictly on training data
- Systematic hyperparameter tuning for Random Forest, XGBoost, and LightGBM
- 5-Fold Stratified Cross-Validation with Out-Of-Fold (OOF) evaluation
- Generation of 10 distinct, validated submission files
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------
# Global Constants & Configurations
# ---------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COL = 'satisfaction'
ID_COL = 'id'

SURVEY_COLS = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Cleanliness'
]

CATEGORICAL_COLS = [
    'Gender',
    'Customer Type',
    'Type of Travel',
    'Class'
]

# ---------------------------------------------------------
# 1. Data Loading & Inspection
# ---------------------------------------------------------
def load_data(train_path='train.csv', test_path='test.csv'):
    """
    Loads train and test datasets, handles target type conversion,
    and reports core statistics.
    """
    print("=" * 70)
    print("STEP 1: LOADING DATASETS")
    print("=" * 70)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing train or test CSV: '{train_path}', '{test_path}'")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"[*] Train shape: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"[*] Test shape:  {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
    
    # Standardize target to binary integers (0 and 1)
    if train_df[TARGET_COL].dtype == bool or train_df[TARGET_COL].dtype == object:
        # Check mapping if strings
        if train_df[TARGET_COL].dtype == object:
            mapping = {'satisfied': 1, 'neutral or dissatisfied': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}
            train_df[TARGET_COL] = train_df[TARGET_COL].map(mapping).fillna(train_df[TARGET_COL].astype(int))
        else:
            train_df[TARGET_COL] = train_df[TARGET_COL].astype(int)
            
    pos_rate = train_df[TARGET_COL].mean()
    print(f"[*] Target balance: {pos_rate:.4%} positive ({train_df[TARGET_COL].sum():,} / {len(train_df):,})")
    return train_df, test_df

# ---------------------------------------------------------
# 2. Missing Value Handling & Feature Engineering
# ---------------------------------------------------------
def engineer_features(train_df, test_df):
    """
    Handles missing values and engineers high-signal domain features:
    - Contextual imputation for Arrival Delay using Departure Delay
    - Delay interactions (Total Delay, Delay Diff, Log Delays)
    - Survey aggregations (Mean, Std, Min, Max, Zero-counts, High/Low counts)
    - Domain service indices (Comfort, Service, Booking Convenience)
    - Categorical interactions and strict Ordinal Encoding
    """
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)
    
    train = train_df.copy()
    test = test_df.copy()
    
    # 2.1 Missing Value Analysis & Contextual Imputation
    train_nulls = train['Arrival Delay in Minutes'].isnull().sum()
    test_nulls = test['Arrival Delay in Minutes'].isnull().sum()
    print(f"[*] Missing values in 'Arrival Delay in Minutes': {train_nulls} (train), {test_nulls} (test)")
    
    # Missing indicator
    train['Arrival_Delay_isna'] = train['Arrival Delay in Minutes'].isnull().astype(int)
    test['Arrival_Delay_isna'] = test['Arrival Delay in Minutes'].isnull().astype(int)
    
    # Learn median arrival delay strictly from training data
    train_delay_median = train['Arrival Delay in Minutes'].median()
    
    # Impute: Arrival delay closely tracks Departure delay; fallback to train median
    train['Arrival Delay in Minutes'] = (
        train['Arrival Delay in Minutes']
        .fillna(train['Departure Delay in Minutes'])
        .fillna(train_delay_median)
    )
    test['Arrival Delay in Minutes'] = (
        test['Arrival Delay in Minutes']
        .fillna(test['Departure Delay in Minutes'])
        .fillna(train_delay_median)
    )
    
    # 2.2 Delay Interactions
    for df in [train, test]:
        df['Total_Delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
        df['Delay_Diff'] = df['Arrival Delay in Minutes'] - df['Departure Delay in Minutes']
        df['Log_Dep_Delay'] = np.log1p(np.maximum(0, df['Departure Delay in Minutes']))
        df['Log_Arr_Delay'] = np.log1p(np.maximum(0, df['Arrival Delay in Minutes']))
        df['Is_Delayed'] = ((df['Departure Delay in Minutes'] > 0) | (df['Arrival Delay in Minutes'] > 0)).astype(int)
        df['Delay_Ratio'] = (df['Total_Delay'] + 1.0) / (df['Flight Distance'] + 1.0)

    # 2.3 Survey Score Aggregations (13 rating categories)
    print("[*] Generating survey rating aggregations and composite indices...")
    for df in [train, test]:
        df['Rating_Mean'] = df[SURVEY_COLS].mean(axis=1)
        df['Rating_Std'] = df[SURVEY_COLS].std(axis=1)
        df['Rating_Sum'] = df[SURVEY_COLS].sum(axis=1)
        df['Rating_Min'] = df[SURVEY_COLS].min(axis=1)
        df['Rating_Max'] = df[SURVEY_COLS].max(axis=1)
        df['Rating_Zero_Count'] = (df[SURVEY_COLS] == 0).sum(axis=1) # 0 = Not Applicable / Unrated
        df['Rating_Low_Count'] = (df[SURVEY_COLS] <= 2).sum(axis=1)
        df['Rating_High_Count'] = (df[SURVEY_COLS] >= 4).sum(axis=1)
        
        # Sub-dimension domain indices
        df['Comfort_Mean'] = df[['Seat comfort', 'Leg room service', 'Cleanliness']].mean(axis=1)
        df['Service_Mean'] = df[['Inflight wifi service', 'Food and drink', 'Inflight entertainment',
                                 'On-board service', 'Baggage handling', 'Checkin service']].mean(axis=1)
        df['Convenience_Mean'] = df[['Departure/Arrival time convenient', 'Ease of Online booking',
                                     'Gate location', 'Online boarding']].mean(axis=1)
        
        # Flight & demographic ratios
        df['Flight_Distance_per_Age'] = df['Flight Distance'] / (df['Age'] + 1)
        df['Class_TravelType'] = df['Class'].astype(str) + '_' + df['Type of Travel'].astype(str)
        df['Customer_Class'] = df['Customer Type'].astype(str) + '_' + df['Class'].astype(str)

    # 2.4 Categorical Encoding (strictly fit on train)
    cat_columns_to_encode = CATEGORICAL_COLS + ['Class_TravelType', 'Customer_Class']
    print(f"[*] Encoding categorical features: {cat_columns_to_encode}")
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[cat_columns_to_encode] = encoder.fit_transform(train[cat_columns_to_encode].astype(str))
    test[cat_columns_to_encode] = encoder.transform(test[cat_columns_to_encode].astype(str))
    
    # Feature columns list
    feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
    print(f"[*] Final feature count: {len(feature_cols)} features engineered.")
    
    return train, test, feature_cols

# ---------------------------------------------------------
# 3. Hyperparameter Tuning
# ---------------------------------------------------------
def run_hyperparameter_tuning(X, y, sample_size=50000):
    """
    Performs randomized hyperparameter search on a representative stratified
    subsample to discover optimal hyperparameter sets for:
    - Random Forest
    - XGBoost
    - LightGBM
    """
    print("\n" + "=" * 70)
    print("STEP 3: SYSTEMATIC HYPERPARAMETER TUNING (ROC-AUC)")
    print("=" * 70)
    
    if len(X) > sample_size:
        print(f"[*] Subsampling {sample_size:,} instances for fast, robust hyperparameter tuning...")
        skf_sub = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        idx, _ = next(skf_sub.split(X, y))
        idx_sample = np.random.RandomState(RANDOM_STATE).choice(idx, size=sample_size, replace=False)
        X_tune, y_tune = X[idx_sample], y[idx_sample]
    else:
        X_tune, y_tune = X, y

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # 3.1 Random Forest Tuning
    print("[*] Tuning Random Forest...")
    rf_param_dist = {
        'n_estimators': [100, 150],
        'max_depth': [14, 18, 22],
        'min_samples_split': [10, 20],
        'max_samples': [0.25, 0.40]
    }
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=4,
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    rf_search.fit(X_tune, y_tune)
    best_rf_params = rf_search.best_params_
    print(f"    --> Best RF ROC-AUC: {rf_search.best_score_:.5f} with params: {best_rf_params}")
    
    # 3.2 LightGBM Tuning
    print("[*] Tuning LightGBM...")
    lgb_param_dist = {
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.03, 0.06, 0.10],
        'n_estimators': [250, 400],
        'subsample': [0.75, 0.90],
        'colsample_bytree': [0.75, 0.90],
        'min_child_samples': [20, 50]
    }
    lgb_search = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
        param_distributions=lgb_param_dist,
        n_iter=5,
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    lgb_search.fit(X_tune, y_tune)
    best_lgb_params = lgb_search.best_params_
    print(f"    --> Best LightGBM ROC-AUC: {lgb_search.best_score_:.5f} with params: {best_lgb_params}")
    
    # 3.3 XGBoost Tuning
    print("[*] Tuning XGBoost...")
    xgb_param_dist = {
        'max_depth': [5, 6, 8],
        'learning_rate': [0.03, 0.06, 0.10],
        'n_estimators': [250, 400],
        'subsample': [0.75, 0.90],
        'colsample_bytree': [0.75, 0.90],
        'reg_alpha': [0.0, 1.0],
        'reg_lambda': [1.0, 3.0]
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(tree_method='hist', eval_metric='auc', random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=xgb_param_dist,
        n_iter=4,
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    xgb_search.fit(X_tune, y_tune)
    best_xgb_params = xgb_search.best_params_
    print(f"    --> Best XGBoost ROC-AUC: {xgb_search.best_score_:.5f} with params: {best_xgb_params}")
    
    return best_rf_params, best_lgb_params, best_xgb_params

# ---------------------------------------------------------
# 4. Model Training & Out-of-Fold (OOF) Prediction
# ---------------------------------------------------------
def train_cross_validated_models(X, y, X_test, best_rf_params, best_lgb_params, best_xgb_params, n_splits=N_SPLITS):
    """
    Trains diverse model architectures using Stratified K-Fold CV:
    1. Tuned LightGBM (GBDT)
    2. Deep Leaves LightGBM (Alternative Architecture)
    3. Tuned XGBoost (Histogram Method)
    4. Regularized XGBoost (Conservative L1/L2 shrinkage)
    5. Tuned Random Forest (Bagged Ensembles)
    6. HistGradientBoosting (Scikit-Learn Gradient Boosting)
    
    Returns:
    - oof_predictions: dict mapping model_name -> 1D array of OOF probabilities
    - test_predictions: dict mapping model_name -> 1D array of test probabilities
    """
    print("\n" + "=" * 70)
    print(f"STEP 4: STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    n_train = len(y)
    n_test = len(X_test)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    # Define model factory configurations
    def get_model_instances():
        return {
            'LightGBM_Tuned': lgb.LGBMClassifier(
                **best_lgb_params,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1
            ),
            'LightGBM_Deep': lgb.LGBMClassifier(
                n_estimators=max(best_lgb_params.get('n_estimators', 350), 350),
                learning_rate=0.04,
                num_leaves=127,
                min_child_samples=60,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_lambda=2.5,
                random_state=RANDOM_STATE + 1,
                verbose=-1,
                n_jobs=-1
            ),
            'XGBoost_Tuned': xgb.XGBClassifier(
                **best_xgb_params,
                tree_method='hist',
                eval_metric='auc',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'XGBoost_Regularized': xgb.XGBClassifier(
                n_estimators=max(best_xgb_params.get('n_estimators', 300), 300),
                learning_rate=0.035,
                max_depth=7,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=2.0,
                reg_lambda=5.0,
                tree_method='hist',
                eval_metric='auc',
                random_state=RANDOM_STATE + 2,
                n_jobs=-1
            ),
            'Random_Forest_Tuned': RandomForestClassifier(
                **best_rf_params,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'HistGradientBoosting_Tuned': HistGradientBoostingClassifier(
                max_iter=280,
                learning_rate=0.05,
                max_leaf_nodes=63,
                l2_regularization=1.5,
                random_state=RANDOM_STATE
            )
        }
    
    # Initialize OOF and test accumulators
    model_names = list(get_model_instances().keys())
    oof_predictions = {name: np.zeros(n_train, dtype=np.float64) for name in model_names}
    test_predictions = {name: np.zeros(n_test, dtype=np.float64) for name in model_names}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- [ FOLD {fold} / {n_splits} ] ---")
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]
        
        fold_models = get_model_instances()
        
        for name, clf in fold_models.items():
            t0 = time.time()
            clf.fit(X_train_fold, y_train_fold)
            
            val_probs = clf.predict_proba(X_val_fold)[:, 1]
            oof_predictions[name][val_idx] = val_probs
            
            test_probs = clf.predict_proba(X_test)[:, 1]
            test_predictions[name] += test_probs / n_splits
            
            fold_auc = roc_auc_score(y_val_fold, val_probs)
            elapsed = time.time() - t0
            print(f"[{name:28s}] Fold {fold} AUC: {fold_auc:.5f} ({elapsed:.1f}s)")
            
    print("\n" + "-" * 70)
    print("OUT-OF-FOLD (OOF) BASELINE SUMMARY")
    print("-" * 70)
    for name in model_names:
        total_auc = roc_auc_score(y, oof_predictions[name])
        print(f"[*] {name:28s} Overall OOF ROC-AUC: {total_auc:.5f}")
        
    return oof_predictions, test_predictions

# ---------------------------------------------------------
# 5. Advanced Ensembling & ROC-AUC Optimization
# ---------------------------------------------------------
def create_ensembles(oof_preds, test_preds, y):
    """
    Creates multiple diverse ensemble strategies:
    - Top GBDT Weighted Blends
    - Multi-Family Uniform Average
    - Rank-Averaged Ensemble
    - Scipy-Optimized Weights directly maximizing ROC-AUC
    """
    print("\n" + "=" * 70)
    print("STEP 5: ENSEMBLE MODELING & ROC-AUC MAXIMIZATION")
    print("=" * 70)
    
    # Ensemble 7: Weighted Top 2 GBDTs (LightGBM + XGBoost)
    oof_blend_top2 = 0.55 * oof_preds['LightGBM_Tuned'] + 0.45 * oof_preds['XGBoost_Tuned']
    test_blend_top2 = 0.55 * test_preds['LightGBM_Tuned'] + 0.45 * test_preds['XGBoost_Tuned']
    auc_top2 = roc_auc_score(y, oof_blend_top2)
    print(f"[*] Ensemble 7 (LGBM + XGBoost Blend)      OOF ROC-AUC: {auc_top2:.5f}")
    
    # Ensemble 8: Equal Multi-Family Average (LightGBM + XGBoost + RF + HistGB)
    oof_multi_avg = (
        oof_preds['LightGBM_Tuned'] +
        oof_preds['XGBoost_Tuned'] +
        oof_preds['Random_Forest_Tuned'] +
        oof_preds['HistGradientBoosting_Tuned']
    ) / 4.0
    test_multi_avg = (
        test_preds['LightGBM_Tuned'] +
        test_preds['XGBoost_Tuned'] +
        test_preds['Random_Forest_Tuned'] +
        test_preds['HistGradientBoosting_Tuned']
    ) / 4.0
    auc_multi_avg = roc_auc_score(y, oof_multi_avg)
    print(f"[*] Ensemble 8 (Multi-Family Average)       OOF ROC-AUC: {auc_multi_avg:.5f}")
    
    # Ensemble 9: Rank Averaging across all 6 base models
    print("[*] Computing Rank-Averaged Ensemble across all models...")
    oof_ranks = np.zeros_like(oof_blend_top2)
    test_ranks = np.zeros_like(test_blend_top2)
    all_models = list(oof_preds.keys())
    
    for name in all_models:
        oof_ranks += rankdata(oof_preds[name]) / len(oof_ranks)
        test_ranks += rankdata(test_preds[name]) / len(test_ranks)
        
    oof_ranks /= len(all_models)
    test_ranks /= len(all_models)
    auc_ranks = roc_auc_score(y, oof_ranks)
    print(f"[*] Ensemble 9 (Rank-Averaged Ensemble)    OOF ROC-AUC: {auc_ranks:.5f}")
    
    # Ensemble 10: Nelder-Mead / SLSQP Optimization on OOF to directly maximize ROC-AUC
    print("[*] Optimizing blend weights to maximize ROC-AUC on OOF validation...")
    model_keys = ['LightGBM_Tuned', 'LightGBM_Deep', 'XGBoost_Tuned', 'XGBoost_Regularized', 'Random_Forest_Tuned', 'HistGradientBoosting_Tuned']
    oof_matrix = np.column_stack([oof_preds[m] for m in model_keys])
    test_matrix = np.column_stack([test_preds[m] for m in model_keys])
    
    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blended = np.dot(oof_matrix, w)
        # Minimize negative ROC-AUC
        return -roc_auc_score(y, blended)
    
    initial_weights = [1.0 / len(model_keys)] * len(model_keys)
    bounds = [(0.0, 1.0) for _ in range(len(model_keys))]
    constraints = {'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)}
    
    res = minimize(loss_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x)
    
    oof_opt_blend = np.dot(oof_matrix, best_weights)
    test_opt_blend = np.dot(test_matrix, best_weights)
    auc_opt = roc_auc_score(y, oof_opt_blend)
    print(f"[*] Ensemble 10 (Optimal ROC-AUC Blend)    OOF ROC-AUC: {auc_opt:.5f}")
    print(f"    --> Optimized Weights:")
    for m, w in zip(model_keys, best_weights):
        print(f"        - {m:28s}: {w:6.2%}")
        
    ensembles = {
        'submission_07_blend_lightgbm_xgboost.csv': test_blend_top2,
        'submission_08_multi_family_average.csv': test_multi_avg,
        'submission_09_rank_averaged_ensemble.csv': test_ranks,
        'submission_10_optimal_roc_auc_blend.csv': test_opt_blend
    }
    oof_ensembles = {
        'submission_07_blend_lightgbm_xgboost.csv': oof_blend_top2,
        'submission_08_multi_family_average.csv': oof_multi_avg,
        'submission_09_rank_averaged_ensemble.csv': oof_ranks,
        'submission_10_optimal_roc_auc_blend.csv': oof_opt_blend
    }
    return ensembles, oof_ensembles

# ---------------------------------------------------------
# Helper: Path Resolution for Kaggle & Local
# ---------------------------------------------------------
def find_data_file(filename):
    """
    Resolves data file path across Kaggle competition mounts, Kaggle dataset mounts, and local directory.
    Prioritizes:
    1. /kaggle/input/competitions/playground-series-s6e10/{filename}
    2. /kaggle/input/playground-series-s6e10/{filename}
    3. ./{filename}
    """
    candidates = [
        f"/kaggle/input/competitions/playground-series-s6e10/{filename}",
        f"/kaggle/input/playground-series-s6e10/{filename}",
        filename
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Default to user-specified Kaggle competition path
    return f"/kaggle/input/competitions/playground-series-s6e10/{filename}"


# ---------------------------------------------------------
# 6. Generate 10 Distinct Submission Files
# ---------------------------------------------------------
def generate_10_submissions(
    test_ids,
    base_test_preds,
    ensemble_preds,
    y,
    base_oof_preds,
    oof_ensembles,
    sample_sub_path=None,
    output_dir='submissions'
):
    """
    Saves and validates exactly 10 distinct submission files with:
    - Uses sample_submission.csv template for exact Kaggle formatting if available
    - Required format: 'id', 'satisfaction'
    - Test IDs matching test.csv and sample_submission.csv in identical sequence
    - Calibrated predicted probabilities between 0.0 and 1.0
    - Verification that no NaNs or formatting errors exist
    """
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING 10 DISTINCT SUBMISSION FILES")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    template_df = None
    if sample_sub_path and os.path.exists(sample_sub_path):
        print(f"[*] Formatting final outputs using template: '{sample_sub_path}'")
        template_df = pd.read_csv(sample_sub_path)
    else:
        print("[*] Formatting outputs using test IDs directly...")
    
    submission_map = {
        'submission_01_lightgbm_tuned.csv': (
            base_test_preds['LightGBM_Tuned'],
            base_oof_preds['LightGBM_Tuned'],
            "Tuned LightGBM GBDT with optimized leaves and feature subsampling"
        ),
        'submission_02_lightgbm_deep.csv': (
            base_test_preds['LightGBM_Deep'],
            base_oof_preds['LightGBM_Deep'],
            "LightGBM with deep leaves (num_leaves=127) capturing high-order non-linearities"
        ),
        'submission_03_xgboost_tuned.csv': (
            base_test_preds['XGBoost_Tuned'],
            base_oof_preds['XGBoost_Tuned'],
            "Tuned XGBoost with histogram tree method and depth-wise splitting"
        ),
        'submission_04_xgboost_regularized.csv': (
            base_test_preds['XGBoost_Regularized'],
            base_oof_preds['XGBoost_Regularized'],
            "XGBoost with conservative learning rate, higher L1/L2 penalties & subsampling"
        ),
        'submission_05_random_forest_tuned.csv': (
            base_test_preds['Random_Forest_Tuned'],
            base_oof_preds['Random_Forest_Tuned'],
            "Tuned Random Forest with bagged tree sub-sampling and leaf regularization"
        ),
        'submission_06_hist_gradient_boosting.csv': (
            base_test_preds['HistGradientBoosting_Tuned'],
            base_oof_preds['HistGradientBoosting_Tuned'],
            "Scikit-Learn HistGradientBoostingClassifier with native histogram binning"
        ),
        'submission_07_blend_lightgbm_xgboost.csv': (
            ensemble_preds['submission_07_blend_lightgbm_xgboost.csv'],
            oof_ensembles['submission_07_blend_lightgbm_xgboost.csv'],
            "Weighted blend of top gradient boosted trees (55% LightGBM + 45% XGBoost)"
        ),
        'submission_08_multi_family_average.csv': (
            ensemble_preds['submission_08_multi_family_average.csv'],
            oof_ensembles['submission_08_multi_family_average.csv'],
            "Equal uniform average of all 4 distinct model families (LGBM + XGB + RF + HistGB)"
        ),
        'submission_09_rank_averaged_ensemble.csv': (
            ensemble_preds['submission_09_rank_averaged_ensemble.csv'],
            oof_ensembles['submission_09_rank_averaged_ensemble.csv'],
            "Rank-averaged ensemble of all models (robust against probability calibration skew)"
        ),
        'submission_10_optimal_roc_auc_blend.csv': (
            ensemble_preds['submission_10_optimal_roc_auc_blend.csv'],
            oof_ensembles['submission_10_optimal_roc_auc_blend.csv'],
            "Directly optimized SLSQP ensemble weights maximizing Out-Of-Fold ROC-AUC score"
        ),
    }
    
    summary_records = []
    
    for filename, (probs, oof_probs, description) in submission_map.items():
        filepath = os.path.join(output_dir, filename)
        
        # Clip probabilities strictly to [0.0, 1.0]
        clean_probs = np.clip(probs, 0.0, 1.0)
        
        if template_df is not None:
            sub_df = template_df.copy()
            sub_df[TARGET_COL] = clean_probs
        else:
            sub_df = pd.DataFrame({
                ID_COL: test_ids,
                TARGET_COL: clean_probs
            })
        
        # Strict validation
        assert sub_df.shape[0] == len(test_ids), f"Row mismatch for {filename}"
        assert sub_df[TARGET_COL].isnull().sum() == 0, f"NaNs detected in {filename}"
        assert sub_df[TARGET_COL].min() >= 0.0, f"Negative probabilities in {filename}"
        assert sub_df[TARGET_COL].max() <= 1.0, f"Probabilities > 1.0 in {filename}"
        
        sub_df.to_csv(filepath, index=False)
        oof_auc = roc_auc_score(y, oof_probs) if oof_probs is not None else np.nan
        
        summary_records.append({
            'Filename': filename,
            'OOF_ROC_AUC': oof_auc,
            'Mean_Pred': clean_probs.mean(),
            'Std_Pred': clean_probs.std(),
            'Min_Pred': clean_probs.min(),
            'Max_Pred': clean_probs.max(),
            'Description': description
        })
        print(f"[✓] Generated: {filename:45s} | OOF ROC-AUC: {oof_auc:.5f}")
        
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(output_dir, 'submissions_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[*] All 10 submission files saved in directory: '{output_dir}'")
    return summary_df

# ---------------------------------------------------------
# 7. Main Execution Flow
# ---------------------------------------------------------
def main(
    train_path=None,
    test_path=None,
    sample_sub_path=None,
    output_dir=None,
    tuning_samples=50000,
    n_splits=5,
    quick_mode=False
):
    """
    Main execution pipeline for Kaggle Airline Passenger Satisfaction.
    Directly callable and auto-detects:
    - /kaggle/input/competitions/playground-series-s6e10/
    - /kaggle/input/playground-series-s6e10/
    - local workspace paths
    """
    # 0. Automatically resolve paths across Kaggle and Local environments
    if train_path is None:
        train_path = find_data_file('train.csv')

    if test_path is None:
        test_path = find_data_file('test.csv')

    if sample_sub_path is None:
        sample_sub_path = find_data_file('sample_submission.csv')

    if output_dir is None:
        if os.path.exists('/kaggle/working'):
            output_dir = '/kaggle/working/submissions'
        else:
            output_dir = 'submissions'

    total_start = time.time()
    print("=" * 70)
    print("AIRLINE PASSENGER SATISFACTION: ROC-AUC MAXIMIZATION PIPELINE")
    print("=" * 70)
    print(f"[*] Train Path:       {train_path}")
    print(f"[*] Test Path:        {test_path}")
    print(f"[*] Sample Sub Path:  {sample_sub_path}")
    print(f"[*] Output Directory: {output_dir}")
    print(f"[*] Folds (CV):       {n_splits}")
    print(f"[*] Tuning Samples:   {tuning_samples:,}")
    print(f"[*] Quick Mode:       {quick_mode}")

    # 1. Load Data
    train_df, test_df = load_data(train_path, test_path)
    test_ids = test_df[ID_COL].values

    # 2. Feature Engineering & Preprocessing
    train_df, test_df, feature_cols = engineer_features(train_df, test_df)

    X = train_df[feature_cols].values
    y = train_df[TARGET_COL].values
    X_test = test_df[feature_cols].values

    # 3. Hyperparameter Tuning
    if not quick_mode:
        best_rf_params, best_lgb_params, best_xgb_params = run_hyperparameter_tuning(
            X, y, sample_size=tuning_samples
        )
    else:
        print("[*] Quick mode active: Using pre-tuned optimal parameters...")
        best_rf_params = {'n_estimators': 120, 'max_depth': 18, 'min_samples_split': 12, 'max_samples': 0.35}
        best_lgb_params = {'num_leaves': 63, 'learning_rate': 0.05, 'n_estimators': 350, 'subsample': 0.85, 'colsample_bytree': 0.80, 'min_child_samples': 30}
        best_xgb_params = {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 350, 'subsample': 0.85, 'colsample_bytree': 0.80, 'reg_alpha': 0.5, 'reg_lambda': 1.5}

    # 4. Stratified Cross-Validation
    oof_preds, test_preds = train_cross_validated_models(
        X, y, X_test,
        best_rf_params, best_lgb_params, best_xgb_params,
        n_splits=n_splits
    )

    # 5. Ensembling
    ensembles, oof_ensembles = create_ensembles(oof_preds, test_preds, y)

    # 6. Generate 10 Submissions
    summary_df = generate_10_submissions(
        test_ids, test_preds, ensembles, y, oof_preds, oof_ensembles,
        sample_sub_path=sample_sub_path,
        output_dir=output_dir
    )

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE IN {total_elapsed / 60:.2f} MINUTES")
    print("=" * 70)
    print(summary_df[['Filename', 'OOF_ROC_AUC', 'Description']].to_string(index=False))
    return summary_df


# ---------------------------------------------------------
# Direct execution (Kaggle / Colab Notebook & Script friendly)
# Runs immediately without argparse to prevent Jupyter kernel errors
# ---------------------------------------------------------
summary_results = main(
    train_path=find_data_file('train.csv'),
    test_path=find_data_file('test.csv'),
    sample_sub_path=find_data_file('sample_submission.csv'),
    output_dir='/kaggle/working/submissions' if os.path.exists('/kaggle/working') else 'submissions',
    tuning_samples=50000,
    n_splits=5,
    quick_mode=False
)
