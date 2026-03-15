"""
材料劣化分析 v0.6 - 複合劣化メカニズムの相互作用分析

目的:
1. Diverse Ensemble: GAM + FAM + GBAM + TabNAM の異種アンサンブル
2. 複合劣化メカニズム（Corrosion + Fatigue など）の相互作用を分析
3. アルゴリズムはv0.5と完全に同一

対象データ:
- Dataset of material degradation event within process industry.xlsx
- 3,772件の材料劣化事故データ（1966-2023）

問題意識:
- 事故は単一要因ではなく、複数の劣化メカニズムが同時に作用して発生
- 複合劣化（multi-mechanism degradation）の本質を理解する

実行方法:
    python run_all_models_v0.6.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

# v0.6の前処理を使用
from data_preprocessing_material_v06 import load_and_preprocess_data_v06

# v0.5と同じモデルとチューナーを使用（アルゴリズムは完全に同一）
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
    print("STEP 1: Training Base Models (v0.6 - Material Degradation)")
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
    output_path = Path('outputs/results/agentic_tuning_v0.6_material.json')
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


def save_results_summary(all_results, best_weights, output_path='outputs/results/all_models_comparison_v0.6_material.csv'):
    """
    結果のサマリーを保存
    
    Parameters
    ----------
    all_results : dict
        全モデルの評価結果
    best_weights : np.array
        最適な重み
    output_path : str
        出力ファイルパス
    """
    print("\n" + "=" * 80)
    print("Saving Results Summary")
    print("=" * 80)
    
    # DataFrameに変換
    results_df = pd.DataFrame(all_results).T
    results_df.index.name = 'Model'
    results_df = results_df.reset_index()
    
    # 重みを追加
    weights_dict = {name: weight for name, weight in zip(['GAM', 'FAM', 'GBAM', 'TabNAM'], best_weights)}
    results_df['ensemble_weight'] = results_df['Model'].map(weights_dict)
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nResults saved to: {output_path}")
    print("\n--- SUMMARY TABLE ---")
    print(results_df.to_string(index=False))


def main():
    """
    メイン実行関数
    """
    print("\n" + "=" * 80)
    print("MATERIAL DEGRADATION ANALYSIS v0.6")
    print("Multi-Mechanism Degradation with Diverse GAMs")
    print("=" * 80)
    
    # データの読み込みと前処理（v0.6）
    print("\n--- Loading and Preprocessing Data ---")
    splits, preprocessor = load_and_preprocess_data_v06(
        data_path='data/Dataset of material degradation event within process industry.xlsx',
        target_type='severity_score',
        target_transform='yeo-johnson',
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )
    
    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_val = splits['y_val']
    y_test = splits['y_test']
    
    print(f"\n--- Data Summary ---")
    print(f"Training set:   {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set:       {X_test.shape}")
    print(f"\nFeatures: {splits['feature_names']}")
    
    # ベースモデルの訓練
    models_dict = train_base_models(X_train, y_train, X_val, y_val)
    
    # アンサンブル重みの最適化
    opt_results, X_final_test, y_final_test = optimize_ensemble_weights(
        models_dict, X_val, y_val
    )
    
    best_weights = opt_results['best_weights']
    print(f"\nOptimized Ensemble Weights:")
    for name, weight in zip(['GAM', 'FAM', 'GBAM', 'TabNAM'], best_weights):
        print(f"  {name:10s}: {weight:.4f}")
    
    # 最終テストセットで評価
    all_results = evaluate_on_full_test_set(
        models_dict, best_weights, X_test, y_test
    )
    
    # 結果の保存
    save_results_summary(all_results, best_weights)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\n次のステップ:")
    print("1. 結果の可視化: python visualize_shap_material_v0.6.py")
    print("2. 複合劣化メカニズムの相互作用分析")
    print("3. 論文の第2ケーススタディとして執筆")


if __name__ == '__main__':
    main()
