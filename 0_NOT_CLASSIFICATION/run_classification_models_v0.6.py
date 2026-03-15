"""
Classification Models v0.6 - 福知山市橋梁健全度分類

目的:
1. 健全度（I〜IV）の分類モデル構築
2. 複数モデルの比較: GAM-like (LR) + XGBoost + LightGBM + CatBoost
3. 方法論の実証: 解釈可能性とアンサンブル

実行方法:
    python run_classification_models_v0.6.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing_classification import (
    load_and_preprocess_classification_data,
    create_stratified_splits
)
from src.evaluation import (
    calculate_classification_metrics,
    evaluate_classification_model,
    print_classification_results
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

from sklearn.model_selection import cross_val_score
import json


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    ロジスティック回帰（GAM-like: 解釈可能なベースラインモデル）
    
    Parameters
    ----------
    X_train : pd.DataFrame
        訓練データ特徴量
    y_train : np.array
        訓練データターゲット
    X_val : pd.DataFrame
        検証データ特徴量
    y_val : np.array
        検証データターゲット
        
    Returns
    -------
    model : LogisticRegression
        学習済みモデル
    results : dict
        評価結果
    """
    print("\n" + "="*60)
    print("Training Logistic Regression (GAM-like Baseline)")
    print("="*60)
    
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    
    model.fit(X_train, y_train)
    
    results = evaluate_classification_model(
        model, X_train, y_train, X_val, y_val,
        model_name='LogisticRegression'
    )
    
    print_classification_results(results)
    
    return model, results


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    ランダムフォレスト分類器
    
    Parameters
    ----------
    X_train : pd.DataFrame
        訓練データ特徴量
    y_train : np.array
        訓練データターゲット
    X_val : pd.DataFrame
        検証データ特徴量
    y_val : np.array
        検証データターゲット
        
    Returns
    -------
    model : RandomForestClassifier
        学習済みモデル
    results : dict
        評価結果
    """
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    results = evaluate_classification_model(
        model, X_train, y_train, X_val, y_val,
        model_name='RandomForest'
    )
    
    print_classification_results(results)
    
    return model, results


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    XGBoost分類器（GBAM-like: 勾配ブースティング）
    
    Parameters
    ----------
    X_train : pd.DataFrame
        訓練データ特徴量
    y_train : np.array
        訓練データターゲット
    X_val : pd.DataFrame
        検証データ特徴量
    y_val : np.array
        検証データターゲット
        
    Returns
    -------
    model : XGBClassifier
        学習済みモデル
    results : dict
        評価結果
    """
    print("\n" + "="*60)
    print("Training XGBoost (GBAM-like)")
    print("="*60)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    results = evaluate_classification_model(
        model, X_train, y_train, X_val, y_val,
        model_name='XGBoost'
    )
    
    print_classification_results(results)
    
    return model, results


def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    LightGBM分類器（高速・高精度）
    
    Parameters
    ----------
    X_train : pd.DataFrame
        訓練データ特徴量
    y_train : np.array
        訓練データターゲット
    X_val : pd.DataFrame
        検証データ特徴量
    y_val : np.array
        検証データターゲット
        
    Returns
    -------
    model : LGBMClassifier
        学習済みモデル
    results : dict
        評価結果
    """
    print("\n" + "="*60)
    print("Training LightGBM")
    print("="*60)
    
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    results = evaluate_classification_model(
        model, X_train, y_train, X_val, y_val,
        model_name='LightGBM'
    )
    
    print_classification_results(results)
    
    return model, results


def train_catboost(X_train, y_train, X_val, y_val):
    """
    CatBoost分類器（カテゴリ特徴量対応）
    
    Parameters
    ----------
    X_train : pd.DataFrame
        訓練データ特徴量
    y_train : np.array
        訓練データターゲット
    X_val : pd.DataFrame
        検証データ特徴量
    y_val : np.array
        検証データターゲット
        
    Returns
    -------
    model : CatBoostClassifier
        学習済みモデル
    results : dict
        評価結果
    """
    if not CATBOOST_AVAILABLE:
        print("CatBoost is not available. Skipping...")
        return None, None
    
    print("\n" + "="*60)
    print("Training CatBoost")
    print("="*60)
    
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=20,
        verbose=False
    )
    
    results = evaluate_classification_model(
        model, X_train, y_train, X_val, y_val,
        model_name='CatBoost'
    )
    
    print_classification_results(results)
    
    return model, results


def create_ensemble_predictions(models, X_val, y_val, weights=None):
    """
    アンサンブル予測（ソフト投票）
    
    Parameters
    ----------
    models : list
        学習済みモデルのリスト
    X_val : pd.DataFrame
        検証データ特徴量
    y_val : np.array
        検証データターゲット
    weights : list, optional
        各モデルの重み
        
    Returns
    -------
    y_pred : np.array
        アンサンブル予測
    y_proba : np.array
        アンサンブル確率予測
    results : dict
        評価結果
    """
    print("\n" + "="*60)
    print("Creating Ensemble Predictions")
    print("="*60)
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # 確率予測の加重平均
    proba_list = []
    for model in models:
        if model is not None and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_val)
            proba_list.append(proba)
    
    if len(proba_list) == 0:
        print("No models with predict_proba available")
        return None, None, None
    
    # 加重平均
    ensemble_proba = np.average(proba_list, axis=0, weights=weights[:len(proba_list)])
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    
    results = calculate_classification_metrics(
        y_val, ensemble_pred, ensemble_proba, prefix='ensemble_'
    )
    results['model'] = 'Ensemble'
    
    print(f"\nEnsemble Results:")
    for key, value in results.items():
        if key != 'model':
            print(f"  {key}: {value:.4f}")
    
    return ensemble_pred, ensemble_proba, results


def main():
    """メイン実行関数"""
    print("\n" + "="*80)
    print(" 福知山市橋梁健全度分類モデル v0.6")
    print("="*80)
    
    # データパス
    data_path = Path(__file__).parent / 'data' / 'fukuchiyama_bridge_sample_v0.6.csv'
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    # データの読み込みと前処理
    print("\n" + "="*60)
    print("Loading and Preprocessing Data")
    print("="*60)
    
    X, y, feature_names, preprocessor = load_and_preprocess_classification_data(
        str(data_path),
        scale_features=True
    )
    
    # 交差検証分割
    splits = create_stratified_splits(X, y, n_splits=5, random_state=42)
    
    # 最初のFoldで評価
    train_idx, val_idx = splits[0]
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"\nUsing Fold 1 for model training:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    
    # モデル訓練
    all_models = []
    all_results = []
    
    # 1. Logistic Regression (GAM-like)
    lr_model, lr_results = train_logistic_regression(X_train, y_train, X_val, y_val)
    all_models.append(lr_model)
    all_results.append(lr_results)
    
    # 2. Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_val, y_val)
    all_models.append(rf_model)
    all_results.append(rf_results)
    
    # 3. XGBoost (GBAM-like)
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
    all_models.append(xgb_model)
    all_results.append(xgb_results)
    
    # 4. LightGBM
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_val, y_val)
    all_models.append(lgb_model)
    all_results.append(lgb_results)
    
    # 5. CatBoost
    cb_model, cb_results = train_catboost(X_train, y_train, X_val, y_val)
    if cb_model is not None:
        all_models.append(cb_model)
        all_results.append(cb_results)
    
    # アンサンブル
    ensemble_pred, ensemble_proba, ensemble_results = create_ensemble_predictions(
        all_models, X_val, y_val
    )
    if ensemble_results is not None:
        all_results.append(ensemble_results)
    
    # 結果の比較
    print("\n" + "="*80)
    print(" Model Comparison Summary")
    print("="*80)
    
    comparison_df = pd.DataFrame(all_results)
    
    # 重要な指標のみ表示
    key_metrics = ['model', 'test_accuracy', 'test_f1_weighted', 'test_kappa']
    available_metrics = [m for m in key_metrics if m in comparison_df.columns]
    
    if len(available_metrics) > 0:
        print("\n" + comparison_df[available_metrics].to_string(index=False))
    
    # 結果の保存
    output_dir = Path(__file__).parent / 'outputs' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'classification_models_v0.6.csv'
    comparison_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    
    # ベストモデルの表示
    if 'test_f1_weighted' in comparison_df.columns:
        best_idx = comparison_df['test_f1_weighted'].idxmax()
        best_model = comparison_df.loc[best_idx, 'model']
        best_f1 = comparison_df.loc[best_idx, 'test_f1_weighted']
        print(f"\n🏆 Best Model: {best_model} (F1-weighted: {best_f1:.4f})")
    
    print("\n" + "="*80)
    print(" Classification Model Training Completed!")
    print("="*80)


if __name__ == '__main__':
    main()
