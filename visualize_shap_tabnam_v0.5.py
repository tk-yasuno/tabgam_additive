"""
TabNAM SHAP可視化スクリプト (v0.5)
TabNAMモデルを使用してSHAP値を計算し、解釈可能な可視化を生成

実行方法:
    python visualize_shap_tabnam_v0.5.py
    
出力:
    outputs/figures/tabnam_shap_summary.png
    outputs/figures/tabnam_shap_importance_bar.png
    outputs/figures/tabnam_shap_dependence/*.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# プロジェクトのルートに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import load_and_preprocess_data
from src.models.tabnam import TabNAMModel
from src.evaluation import calculate_metrics


def train_tabnam_model(X_train, y_train, X_val, y_val):
    """TabNAMモデルを訓練"""
    print("=" * 80)
    print("Training TabNAM Model")
    print("=" * 80)
    
    tabnam = TabNAMModel(
        n_d=8,
        n_a=8,
        n_steps=3,
        lambda_sparse=1e-3,
        seed=42,
        verbose=1
    )
    
    # max_epochs, patience, batch_sizeをfit前に設定
    tabnam.max_epochs = 50
    tabnam.patience = 10
    tabnam.batch_size = 256
    
    print("\nTraining TabNAM...")
    tabnam.fit(X_train, y_train)
    
    # 検証
    y_val_pred = tabnam.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)
    print(f"\nValidation Performance:")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    return tabnam


def compute_shap_values(model, X_sample, X_background, feature_names):
    """
    TabNAMのSHAP値を計算
    
    Parameters
    ----------
    model : TabNAMModel
        訓練済みTabNAMモデル
    X_sample : pd.DataFrame or np.array
        SHAP値を計算するサンプル
    X_background : pd.DataFrame or np.array
        背景データ（KernelExplainerで使用）
    feature_names : list
        特徴量名
    
    Returns
    -------
    shap_values : np.array
        SHAP値
    """
    print("\n" + "=" * 80)
    print("Computing SHAP Values for TabNAM")
    print("=" * 80)
    
    # DataFrameをnumpy配列に変換
    if isinstance(X_sample, pd.DataFrame):
        X_sample_array = X_sample.values
    else:
        X_sample_array = X_sample
        
    if isinstance(X_background, pd.DataFrame):
        X_background_array = X_background.values
    else:
        X_background_array = X_background
    
    # TabNetの予測関数をラップ
    def predict_fn(X):
        """予測関数（SHAP用）"""
        if isinstance(X, np.ndarray):
            # numpy配列の場合、DataFrameに変換
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
        return model.predict(X_df)
    
    # KernelExplainerを使用（深層学習モデル用）
    print(f"\nInitializing SHAP KernelExplainer...")
    print(f"  Background samples: {len(X_background_array)}")
    print(f"  Evaluation samples: {len(X_sample_array)}")
    print(f"  Note: This may take several minutes...")
    
    # 背景データをサンプリング（計算時間短縮）
    if len(X_background_array) > 100:
        np.random.seed(42)
        background_indices = np.random.choice(len(X_background_array), 100, replace=False)
        X_background_sampled = X_background_array[background_indices]
    else:
        X_background_sampled = X_background_array
    
    # SHAP値計算用のサンプルも制限
    if len(X_sample_array) > 1000:
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_sample_array), 1000, replace=False)
        X_sample_for_shap = X_sample_array[sample_indices]
    else:
        X_sample_for_shap = X_sample_array
    
    explainer = shap.KernelExplainer(predict_fn, X_background_sampled)
    
    print(f"\nComputing SHAP values...")
    shap_values = explainer.shap_values(X_sample_for_shap)
    
    print(f"SHAP computation completed!")
    print(f"  Shape: {shap_values.shape}")
    
    return shap_values, X_sample_for_shap


def visualize_shap_summary(X, shap_values, feature_names, output_path):
    """SHAP Summary Plot（全体的な特徴量重要度）"""
    print("\n[1/4] Generating SHAP Summary Plot...")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=20)
    plt.title('TabNAM: SHAP Summary (Top 20 Features)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def visualize_shap_bar(shap_values, feature_names, output_path):
    """SHAP Bar Plot（平均絶対SHAP値）"""
    print("\n[2/4] Generating SHAP Bar Plot...")
    
    # 平均絶対SHAP値を計算
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # データフレーム作成
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)
    
    # Top 20のみ表示
    shap_importance_top20 = shap_importance.tail(20)
    
    # プロット
    plt.figure(figsize=(10, 10))
    plt.barh(shap_importance_top20['feature'], shap_importance_top20['importance'], color='steelblue')
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.title('TabNAM: Feature Importance (SHAP, Top 20)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"\nTop 20 Important Features (TabNAM):")
    for i, row in shap_importance.tail(20).iterrows():
        print(f"  {row['feature']:<40s}: {row['importance']:.6f}")


def visualize_shap_dependence(X, shap_values, feature_names, n_top_features, output_dir):
    """SHAP Dependence Plots（個別特徴量の影響）"""
    print(f"\n[3/4] Generating SHAP Dependence Plots (Top {n_top_features} features)...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Top N特徴量を取得
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-n_top_features:][::-1]
    
    for rank, feature_idx in enumerate(top_indices, 1):
        feature = feature_names[feature_idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, 
            shap_values, 
            X, 
            feature_names=feature_names,
            show=False
        )
        plt.title(f'TabNAM: SHAP Dependence Plot - {feature} (Rank {rank})', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        safe_feature_name = feature.replace('/', '_').replace(' ', '_').replace('×', 'x')
        output_path = output_dir / f"tabnam_shap_dependence_{rank:02d}_{safe_feature_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [{rank}/{n_top_features}] Saved: {output_path.name}")


def compare_base_and_interaction_features(shap_values, feature_names, output_path):
    """基本特徴量と交互作用特徴量のSHAP重要度を比較"""
    print("\n[4/4] Comparing Base vs Interaction Features...")
    
    # 平均絶対SHAP値を計算
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # データフレーム作成
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    })
    
    # 交互作用特徴量（_x_ または _by_ を含む）と基本特徴量を分類
    shap_df['is_interaction'] = shap_df['feature'].str.contains('_x_|_by_', regex=True)
    
    # 基本特徴量と交互作用特徴量の統計
    base_features = shap_df[~shap_df['is_interaction']]
    interaction_features = shap_df[shap_df['is_interaction']]
    
    base_sum = base_features['importance'].sum()
    interaction_sum = interaction_features['importance'].sum()
    base_mean = base_features['importance'].mean()
    interaction_mean = interaction_features['importance'].mean()
    
    print(f"\nBase Features ({len(base_features)}):")
    print(f"  Total SHAP: {base_sum:.4f}")
    print(f"  Mean SHAP: {base_mean:.6f}")
    
    print(f"\nInteraction Features ({len(interaction_features)}):")
    print(f"  Total SHAP: {interaction_sum:.4f}")
    print(f"  Mean SHAP: {interaction_mean:.6f}")
    
    if len(interaction_features) > 0:
        print(f"\nInteraction/Base Ratio: {interaction_sum/base_sum:.4f}")
    
    # 比較プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左: トップ10基本特徴量
    top_base = base_features.nlargest(10, 'importance')
    axes[0].barh(top_base['feature'], top_base['importance'], color='cornflowerblue')
    axes[0].set_xlabel('Mean |SHAP value|', fontsize=11)
    axes[0].set_title('Top 10 Base Features', fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # 右: トップ10交互作用特徴量
    if len(interaction_features) >= 10:
        top_interaction = interaction_features.nlargest(10, 'importance')
        axes[1].barh(top_interaction['feature'], top_interaction['importance'], color='coral')
        axes[1].set_xlabel('Mean |SHAP value|', fontsize=11)
        axes[1].set_title('Top 10 Interaction Features', fontsize=13, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Interaction Features\n(< 10 features)', 
                    ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('Top 10 Interaction Features', fontsize=13, fontweight='bold')
    
    plt.suptitle('TabNAM: Base vs Interaction Features (SHAP Importance)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return {
        'base_sum': base_sum,
        'interaction_sum': interaction_sum,
        'base_mean': base_mean,
        'interaction_mean': interaction_mean,
        'interaction_ratio': interaction_sum / base_sum if base_sum > 0 else 0
    }


def main():
    """メイン処理"""
    print("=" * 80)
    print("TabNAM SHAP Visualization (v0.5)")
    print("=" * 80)
    
    # データ読み込み
    print("\nLoading data...")
    data_path = "data/260311v1_inspection_base_and_hazard_estimation.csv"
    data = load_and_preprocess_data(
        data_path,
        add_base_interactions=True,  # v0.5では基本特徴量の交互作用を使用
        random_state=42
    )
    
    X = data['X']
    y = data['y']
    feature_names = data['feature_names']
    
    # 訓練/検証分割（80/20）
    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    
    indices = np.random.RandomState(42).permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y[test_indices]
    
    print(f"Data loaded: {len(X)} samples, {len(feature_names)} features")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # TabNAMモデルを訓練
    tabnam = train_tabnam_model(X_train, y_train, X_test, y_test)
    
    # SHAP値を計算
    shap_values, X_shap = compute_shap_values(
        model=tabnam,
        X_sample=X_test,
        X_background=X_train,
        feature_names=feature_names
    )
    
    # 出力ディレクトリ作成
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dependence_dir = output_dir / 'tabnam_shap_dependence'
    dependence_dir.mkdir(parents=True, exist_ok=True)
    
    # 可視化
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    # 1. SHAP Summary Plot
    visualize_shap_summary(
        X=X_shap,
        shap_values=shap_values,
        feature_names=feature_names,
        output_path=output_dir / 'tabnam_shap_summary_v0.5.png'
    )
    
    # 2. SHAP Bar Plot
    visualize_shap_bar(
        shap_values=shap_values,
        feature_names=feature_names,
        output_path=output_dir / 'tabnam_shap_importance_bar_v0.5.png'
    )
    
    # 3. SHAP Dependence Plots (Top 10)
    visualize_shap_dependence(
        X=X_shap,
        shap_values=shap_values,
        feature_names=feature_names,
        n_top_features=10,
        output_dir=dependence_dir
    )
    
    # 4. Base vs Interaction Features
    stats = compare_base_and_interaction_features(
        shap_values=shap_values,
        feature_names=feature_names,
        output_path=output_dir / 'tabnam_base_vs_interaction_v0.5.png'
    )
    
    print("\n" + "=" * 80)
    print("TabNAM SHAP Visualization Completed!")
    print("=" * 80)
    print(f"\nKey Statistics:")
    print(f"  Interaction/Base SHAP Ratio: {stats['interaction_ratio']:.4f}")
    print(f"  Base Features Mean SHAP: {stats['base_mean']:.6f}")
    print(f"  Interaction Features Mean SHAP: {stats['interaction_mean']:.6f}")
    
    print(f"\nOutput Files:")
    print(f"  - outputs/figures/tabnam_shap_summary_v0.5.png")
    print(f"  - outputs/figures/tabnam_shap_importance_bar_v0.5.png")
    print(f"  - outputs/figures/tabnam_base_vs_interaction_v0.5.png")
    print(f"  - outputs/figures/tabnam_shap_dependence/*.png (10 files)")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
