"""
モデル評価用の共通関数
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     prefix: str = '') -> Dict[str, float]:
    """
    回帰評価指標の計算
    
    Parameters
    ----------
    y_true : np.ndarray
        真の値
    y_pred : np.ndarray
        予測値
    prefix : str, optional
        指標名のプレフィックス（例: 'train_', 'test_'）
        
    Returns
    -------
    dict
        評価指標の辞書
    """
    metrics = {}
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics[f'{prefix}rmse'] = rmse
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    metrics[f'{prefix}mae'] = mae
    
    # R² (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)
    metrics[f'{prefix}r2'] = r2
    
    # MAPE (Mean Absolute Percentage Error) - ゼロ除算を避ける
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics[f'{prefix}mape'] = mape
    
    # 平均予測値と真値
    metrics[f'{prefix}mean_true'] = np.mean(y_true)
    metrics[f'{prefix}mean_pred'] = np.mean(y_pred)
    
    # 標準偏差
    metrics[f'{prefix}std_true'] = np.std(y_true)
    metrics[f'{prefix}std_pred'] = np.std(y_pred)
    
    return metrics


def calculate_quantile_loss(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           quantile: float = 0.5) -> float:
    """
    分位損失（Pinball Loss）の計算
    
    Parameters
    ----------
    y_true : np.ndarray
        真の値
    y_pred : np.ndarray
        予測値
    quantile : float
        分位点（0 < quantile < 1）
        
    Returns
    -------
    float
        分位損失
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, 
                    quantile * errors,
                    (quantile - 1) * errors)
    return np.mean(loss)


def calculate_quantile_coverage(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               quantile: float = 0.9) -> float:
    """
    分位カバレッジの計算
    
    Parameters
    ----------
    y_true : np.ndarray
        真の値
    y_pred : np.ndarray
        予測分位値
    quantile : float
        分位点
        
    Returns
    -------
    float
        カバレッジ率（理想は quantile に近い値）
    """
    coverage = np.mean(y_true <= y_pred)
    return coverage


def evaluate_regression_model(model,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              model_name: str = 'Model') -> Dict[str, float]:
    """
    回帰モデルの訓練・テスト評価
    
    Parameters
    ----------
    model : object
        学習済みモデル（predictメソッドを持つ）
    X_train : np.ndarray
        訓練データ特徴量
    y_train : np.ndarray
        訓練データターゲット
    X_test : np.ndarray
        テストデータ特徴量
    y_test : np.ndarray
        テストデータターゲット
    model_name : str
        モデル名
        
    Returns
    -------
    dict
        評価指標辞書
    """
    results = {'model': model_name}
    
    # 訓練データの予測
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred, prefix='train_')
    results.update(train_metrics)
    
    # テストデータの予測
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, prefix='test_')
    results.update(test_metrics)
    
    # 過学習の指標
    results['overfit_rmse'] = results['train_rmse'] - results['test_rmse']
    results['overfit_r2'] = results['train_r2'] - results['test_r2']
    
    return results


def print_evaluation_results(results: Dict[str, float]):
    """
    評価結果の表示
    
    Parameters
    ----------
    results : dict
        評価指標辞書
    """
    print("=" * 60)
    print(f"Evaluation Results: {results.get('model', 'Unknown')}")
    print("=" * 60)
    
    print("\n[Training Set]")
    print(f"  RMSE: {results.get('train_rmse', 0):.6f}")
    print(f"  MAE:  {results.get('train_mae', 0):.6f}")
    print(f"  R²:   {results.get('train_r2', 0):.6f}")
    
    print("\n[Test Set]")
    print(f"  RMSE: {results.get('test_rmse', 0):.6f}")
    print(f"  MAE:  {results.get('test_mae', 0):.6f}")
    print(f"  R²:   {results.get('test_r2', 0):.6f}")
    
    print("\n[Overfitting Check]")
    print(f"  RMSE difference: {results.get('overfit_rmse', 0):.6f}")
    print(f"  R² difference:   {results.get('overfit_r2', 0):.6f}")
    
    print("=" * 60)


def cross_validate_model(model_class,
                        model_params: Dict,
                        X: np.ndarray,
                        y: np.ndarray,
                        fold_splits: list,
                        model_name: str = 'Model',
                        fit_params: Dict = None) -> Tuple[list, Dict]:
    """
    Cross-validationの実行
    
    Parameters
    ----------
    model_class : class
        モデルクラス
    model_params : dict
        モデルのパラメータ
    X : np.ndarray
        特徴量
    y : np.ndarray
        ターゲット
    fold_splits : list
        (train_idx, test_idx)のリスト
    model_name : str
        モデル名
    fit_params : dict, optional
        fitメソッドに渡す追加パラメータ
        
    Returns
    -------
    tuple
        (fold_results: list, avg_results: dict)
    """
    fold_results = []
    
    if fit_params is None:
        fit_params = {}
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation: {model_name}")
    print(f"{'='*60}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits, 1):
        print(f"\nFold {fold_idx}/{len(fold_splits)}...")
        
        # データ分割
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # モデル学習
        model = model_class(**model_params)
        model.fit(X_train, y_train, **fit_params)
        
        # 評価
        results = evaluate_regression_model(
            model, X_train, y_train, X_test, y_test, 
            model_name=f"{model_name}_fold{fold_idx}"
        )
        results['fold'] = fold_idx
        fold_results.append(results)
        
        print(f"  Test RMSE: {results['test_rmse']:.6f}, R²: {results['test_r2']:.6f}")
    
    # 平均の計算
    df = pd.DataFrame(fold_results)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    avg_results = df[numeric_cols].mean().to_dict()
    avg_results['model'] = model_name
    avg_results['n_folds'] = len(fold_splits)
    
    print(f"\n{'='*60}")
    print(f"Average Results:")
    print(f"  Test RMSE: {avg_results['test_rmse']:.6f} ± {df['test_rmse'].std():.6f}")
    print(f"  Test MAE:  {avg_results['test_mae']:.6f} ± {df['test_mae'].std():.6f}")
    print(f"  Test R²:   {avg_results['test_r2']:.6f} ± {df['test_r2'].std():.6f}")
    print(f"{'='*60}")
    
    return fold_results, avg_results


def calculate_residual_statistics(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, float]:
    """
    残差の統計量計算
    
    Parameters
    ----------
    y_true : np.ndarray
        真の値
    y_pred : np.ndarray
        予測値
        
    Returns
    -------
    dict
        残差統計量
    """
    residuals = y_true - y_pred
    
    stats = {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'median_residual': np.median(residuals),
        'q25_residual': np.percentile(residuals, 25),
        'q75_residual': np.percentile(residuals, 75),
    }
    
    return stats


def calculate_classification_metrics(y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None,
                                    prefix: str = '') -> Dict[str, float]:
    """
    分類評価指標の計算
    
    Parameters
    ----------
    y_true : np.ndarray
        真のクラスラベル
    y_pred : np.ndarray
        予測クラスラベル
    y_pred_proba : np.ndarray, optional
        各クラスの予測確率（shape: [n_samples, n_classes]）
    prefix : str, optional
        指標名のプレフィックス（例: 'train_', 'test_'）
        
    Returns
    -------
    dict
        評価指標の辞書
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, confusion_matrix, cohen_kappa_score,
                                 log_loss)
    
    metrics = {}
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    metrics[f'{prefix}accuracy'] = accuracy
    
    # Precision, Recall, F1 (macro average)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{prefix}precision_macro'] = precision
    
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{prefix}recall_macro'] = recall
    
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{prefix}f1_macro'] = f1
    
    # Weighted average (クラス不均衡対応)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics[f'{prefix}precision_weighted'] = precision_weighted
    
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics[f'{prefix}recall_weighted'] = recall_weighted
    
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics[f'{prefix}f1_weighted'] = f1_weighted
    
    # Cohen's Kappa (クラス不均衡に頑健)
    kappa = cohen_kappa_score(y_true, y_pred)
    metrics[f'{prefix}kappa'] = kappa
    
    # Log Loss (確率が利用可能な場合)
    if y_pred_proba is not None:
        try:
            logloss = log_loss(y_true, y_pred_proba)
            metrics[f'{prefix}logloss'] = logloss
        except:
            pass
    
    # Confusion Matrix（各クラスの予測正解率）
    cm = confusion_matrix(y_true, y_pred)
    for i in range(cm.shape[0]):
        if cm[i, :].sum() > 0:
            class_accuracy = cm[i, i] / cm[i, :].sum()
            metrics[f'{prefix}class_{i}_accuracy'] = class_accuracy
    
    return metrics


def print_classification_results(results: Dict[str, float]):
    """
    分類結果のプリント
    
    Parameters
    ----------
    results : dict
        評価指標辞書
    """
    print("\n" + "="*60)
    print("Classification Evaluation Results")
    print("="*60)
    
    if 'model' in results:
        print(f"\nModel: {results['model']}")
    
    # 訓練データ指標
    print("\n[Training Set]")
    train_keys = [k for k in results.keys() if k.startswith('train_')]
    for key in sorted(train_keys):
        metric_name = key.replace('train_', '')
        print(f"  {metric_name:25s}: {results[key]:8.4f}")
    
    # テストデータ指標
    print("\n[Test Set]")
    test_keys = [k for k in results.keys() if k.startswith('test_')]
    for key in sorted(test_keys):
        metric_name = key.replace('test_', '')
        print(f"  {metric_name:25s}: {results[key]:8.4f}")
    
    print("="*60)


def evaluate_classification_model(model,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  model_name: str = 'Model') -> Dict[str, float]:
    """
    分類モデルの訓練・テスト評価
    
    Parameters
    ----------
    model : object
        学習済みモデル（predict, predict_probaメソッドを持つ）
    X_train : np.ndarray
        訓練データ特徴量
    y_train : np.ndarray
        訓練データターゲット
    X_test : np.ndarray
        テストデータ特徴量
    y_test : np.ndarray
        テストデータターゲット
    model_name : str
        モデル名
        
    Returns
    -------
    dict
        評価指標辞書
    """
    results = {'model': model_name}
    
    # 訓練データの予測
    y_train_pred = model.predict(X_train)
    
    # 確率予測（利用可能な場合）
    y_train_proba = None
    try:
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)
    except:
        pass
    
    train_metrics = calculate_classification_metrics(
        y_train, y_train_pred, y_train_proba, prefix='train_'
    )
    results.update(train_metrics)
    
    # テストデータの予測
    y_test_pred = model.predict(X_test)
    
    y_test_proba = None
    try:
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)
    except:
        pass
    
    test_metrics = calculate_classification_metrics(
        y_test, y_test_pred, y_test_proba, prefix='test_'
    )
    results.update(test_metrics)
    
    return results


if __name__ == '__main__':
    # テスト
    print("Evaluation functions loaded successfully!")
    
    # 回帰用ダミーデータでテスト
    y_true_reg = np.random.rand(100)
    y_pred_reg = y_true_reg + np.random.randn(100) * 0.1
    
    metrics_reg = calculate_metrics(y_true_reg, y_pred_reg, prefix='test_')
    print("\nRegression Test Metrics:")
    for k, v in metrics_reg.items():
        print(f"  {k}: {v:.6f}")
    
    # 分類用ダミーデータでテスト
    y_true_clf = np.random.randint(0, 4, 100)
    y_pred_clf = np.random.randint(0, 4, 100)
    
    metrics_clf = calculate_classification_metrics(y_true_clf, y_pred_clf, prefix='test_')
    print("\nClassification Test Metrics:")
    for k, v in metrics_clf.items():
        print(f"  {k}: {v:.6f}")
