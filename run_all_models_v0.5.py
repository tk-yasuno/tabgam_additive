"""
HRAM v0.5 - Agentic Ensemble Weight Tuning

目的:
1. Diverse Ensemble: GAM + FAM + GBAM + TabNAM の異種アンサンブル
2. Agentic AI: モデル個別ハイパーパラメータ + アンサンブル重みの自動探索
3. Planner-Executor-Evaluator パターンによる最適化

注意:
- FOBAM v2は除外（FAMと本質的に同じため）
- 4つの多様なモデルによるアンサンブル構築

実行方法:
    python run_all_models_v0.5.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import load_and_preprocess_data
from src.models.glm_gam import GLMGAMModel
from src.models.fam import FAMModel
from src.models.gbam import GBAMModel
from src.models.tabnam import TabNAMModel
from src.agentic_tuner import AgenticTuner, create_diverse_ensemble_predictions
from src.evaluation import calculate_metrics
from sklearn.model_selection import KFold


def train_base_models(X_train, y_train, X_val, y_val):
    """
    4つのベースモデルを訓練
    
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
    models_dict : dict
        訓練済みモデルの辞書
    """
    print("=" * 80)
    print("STEP 1: Training Base Models")
    print("=" * 80)
    
    models_dict = {}
    
    # 1. GLM-GAM
    print("\n[1/4] Training GLM-GAM...")
    glm_gam = GLMGAMModel(
        n_splines=10,
        spline_order=3,
        lam=0.6,
        max_iter=100
    )
    glm_gam.fit(X_train, y_train)
    y_val_pred = glm_gam.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)
    print(f"  Validation R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.6f}")
    models_dict['GAM'] = glm_gam
    
    # 2. FAM
    print("\n[2/4] Training FAM...")
    fam = FAMModel(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=50,
        min_samples_split=100,
        max_features='sqrt'
    )
    fam.fit(X_train, y_train)
    y_val_pred = fam.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)
    print(f"  Validation R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.6f}")
    models_dict['FAM'] = fam
    
    # 3. GBAM
    print("\n[3/4] Training GBAM...")
    gbam = GBAMModel(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        bagging_fraction=0.8,
        feature_fraction=0.8,
        reg_lambda=1.0
    )
    gbam.fit(X_train, y_train)
    y_val_pred = gbam.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)
    print(f"  Validation R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.6f}")
    models_dict['GBAM'] = gbam
    
    # 4. TabNAM
    print("\n[4/4] Training TabNAM...")
    tabnam = TabNAMModel(
        n_d=8,
        n_a=8,
        n_steps=3,
        lambda_sparse=1e-3,
        seed=42,
        verbose=0
    )
    # max_epochs, patience, batch_sizeをfit前に設定（インスタンス変数）
    tabnam.max_epochs = 50
    tabnam.patience = 10
    tabnam.batch_size = 256
    
    tabnam.fit(X_train, y_train)
    y_val_pred = tabnam.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)
    print(f"  Validation R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.6f}")
    models_dict['TabNAM'] = tabnam
    
    print("\nBase models training completed!")
    return models_dict


def optimize_ensemble_weights(models_dict, X_val, y_val):
    """
    Agentic Tuner でアンサンブル重みを最適化
    
    Parameters
    ----------
    models_dict : dict
        訓練済みモデルの辞書
    X_val : pd.DataFrame
        検証データ特徴量
    y_val : np.array
        検証データターゲット
    
    Returns
    -------
    results : dict
        最適化結果
    """
    print("\n" + "=" * 80)
    print("STEP 2: Agentic Ensemble Weight Optimization")
    print("=" * 80)
    
    # AgenticTuner を初期化
    # 検証データをさらに分割（重み最適化用 vs 最終テスト用）
    n_val = len(X_val)
    split_idx = int(n_val * 0.5)
    
    X_weight_opt = X_val.iloc[:split_idx]
    y_weight_opt = y_val[:split_idx]
    X_final_test = X_val.iloc[split_idx:]
    y_final_test = y_val[split_idx:]
    
    tuner = AgenticTuner(
        models_dict=models_dict,
        X_train=X_weight_opt,  # DataFrameとして渡す
        y_train=y_weight_opt,
        X_val=X_final_test,  # DataFrameとして渡す
        y_val=y_final_test,
        optimization_metric='r2',
        max_iterations=50,
        patience=10,
        random_state=42
    )
    
    # 最適化実行
    results = tuner.run()
    
    # 結果保存
    output_path = Path('outputs/results/agentic_tuning_v0.5.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tuner.save_results(str(output_path))
    
    return results, X_final_test, y_final_test


def evaluate_on_full_test_set(models_dict, best_weights, X_test, y_test):
    """
    最終テストセットで評価
    
    Parameters
    ----------
    models_dict : dict
        訓練済みモデルの辞書
    best_weights : np.array
        最適な重み
    X_test : pd.DataFrame
        テストデータ特徴量
    y_test : np.array
        テストデータターゲット
    
    Returns
    -------
    results : dict
        評価結果
    """
    print("\n" + "=" * 80)
    print("STEP 3: Final Evaluation on Test Set")
    print("=" * 80)
    
    results = {}
    
    # 個別モデルの評価
    print("\nIndividual Model Performance:")
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        results[name] = metrics
        print(f"  {name:10s}: R²={metrics['r2']:.6f}, RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
    
    # アンサンブルの評価
    print("\nOptimized Ensemble Performance:")
    ensemble_pred = create_diverse_ensemble_predictions(
        models_dict, best_weights, X_test
    )
    ensemble_metrics = calculate_metrics(y_test, ensemble_pred)
    results['ENSEMBLE'] = ensemble_metrics
    print(f"  ENSEMBLE:   R²={ensemble_metrics['r2']:.6f}, RMSE={ensemble_metrics['rmse']:.6f}, MAE={ensemble_metrics['mae']:.6f}")
    
    # 均等重みアンサンブルも比較
    print("\nUniform Ensemble Performance (for comparison):")
    uniform_weights = np.ones(len(models_dict)) / len(models_dict)
    uniform_pred = create_diverse_ensemble_predictions(
        models_dict, uniform_weights, X_test
    )
    uniform_metrics = calculate_metrics(y_test, uniform_pred)
    results['UNIFORM_ENSEMBLE'] = uniform_metrics
    print(f"  UNIFORM:    R²={uniform_metrics['r2']:.6f}, RMSE={uniform_metrics['rmse']:.6f}, MAE={uniform_metrics['mae']:.6f}")
    
    # 改善度を計算
    improvement = ensemble_metrics['r2'] - uniform_metrics['r2']
    print(f"\nImprovement from Uniform Weights: {improvement:+.6f} R²")
    
    # ベストモデルとの比較
    best_individual = max(results.items(), key=lambda x: x[1]['r2'] if x[0] not in ['ENSEMBLE', 'UNIFORM_ENSEMBLE'] else -np.inf)
    print(f"\nBest Individual Model: {best_individual[0]} (R²={best_individual[1]['r2']:.6f})")
    improvement_vs_best = ensemble_metrics['r2'] - best_individual[1]['r2']
    print(f"Improvement from Best Individual: {improvement_vs_best:+.6f} R²")
    
    return results


def save_comparison_results(results, best_weights, model_names):
    """
    結果をCSVファイルに保存
    
    Parameters
    ----------
    results : dict
        評価結果
    best_weights : np.array
        最適な重み
    model_names : list
        モデル名のリスト
    """
    # 結果をDataFrameに変換
    df = pd.DataFrame(results).T
    df = df[['rmse', 'mae', 'r2']]  # 列の順序を整理
    df = df.round(6)
    
    # 重みを追加
    weights_df = pd.DataFrame({
        'model': model_names + ['ENSEMBLE', 'UNIFORM_ENSEMBLE'],
        'weight': list(best_weights) + [np.nan, np.nan]
    })
    weights_df = weights_df.set_index('model')
    
    df = df.join(weights_df)
    
    # 保存
    output_path = Path('outputs/results/all_models_comparison_v0.5.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    
    print(f"\nResults saved to: {output_path}")
    print("\nFinal Results Table:")
    print(df.to_string())


def main():
    """メイン実行関数"""
    print("\n" + "=" * 80)
    print("HRAM v0.5 - Agentic Ensemble Weight Tuning")
    print("=" * 80)
    print("\nDiverse Ensemble: GAM + FAM + GBAM + TabNAM")
    print("Optimization: Agentic AI with Planner-Executor-Evaluator Pattern")
    print()
    
    # データ読み込み
    print("Loading and preprocessing data...")
    data_path = "data/260311v1_inspection_base_and_hazard_estimation.csv"
    data = load_and_preprocess_data(data_path, add_base_interactions=True)
    
    X = data['X']
    y = data['y']
    feature_names = data['feature_names']
    
    print(f"Data loaded: {len(X)} samples, {len(feature_names)} features")
    print(f"Target variable range: [{y.min():.3f}, {y.max():.3f}]")
    
    # 訓練/検証分割（80/20）
    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    
    indices = np.random.RandomState(42).permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_val = X.iloc[val_indices]
    y_val = y[val_indices]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # STEP 1: ベースモデルの訓練
    models_dict = train_base_models(X_train, y_train, X_val, y_val)
    
    # STEP 2: Agentic Tuner でアンサンブル重みを最適化
    tuning_results, X_test, y_test = optimize_ensemble_weights(models_dict, X_val, y_val)
    
    best_weights = tuning_results['best_weights']
    model_names = tuning_results['model_names']
    
    print("\n" + "=" * 80)
    print("Optimized Weights Summary:")
    print("=" * 80)
    for name, weight in zip(model_names, best_weights):
        print(f"  {name:10s}: {weight:.4f}")
    
    # STEP 3: 最終評価
    results = evaluate_on_full_test_set(models_dict, best_weights, X_test, y_test)
    
    # 結果保存
    save_comparison_results(results, best_weights, model_names)
    
    print("\n" + "=" * 80)
    print("v0.5 Execution Completed!")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Diverse Ensemble successfully optimized with Agentic AI")
    print("2. Planner-Executor-Evaluator pattern demonstrated")
    print("3. Weight optimization results saved")
    print("\nNext Steps:")
    print("- Review optimization history in outputs/results/agentic_tuning_v0.5.json")
    print("- Analyze model weight distribution")
    print("- Document lessons in Lesson_5th_train.md")


if __name__ == "__main__":
    main()
