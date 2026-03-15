"""
材料劣化分析 v0.6 - SHAP可視化・相互作用分析

目的:
- 複合劣化メカニズム（Corrosion + Fatigue など）の相互作用を可視化
- GAMとTabNAMのSHAP値を計算
- 特徴量重要度と依存性プロットを生成

実行方法:
    python visualize_shap_material_v0.6.py
    
出力:
    outputs/figures/material_v0.6/
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing_material_v06 import load_and_preprocess_data_v06
from src.models.glm_gam import GLMGAMModel
from src.models.tabnam import TabNAMModel
from src.evaluation import calculate_metrics


def train_models(X_train, y_train, X_val, y_val):
    """GAMとTabNAMモデルを訓練"""
    print("=" * 80)
    print("Training Models for SHAP Analysis")
    print("=" * 80)
    
    models = {}
    
    # GAM
    print("\n[1/2] Training GAM...")
    gam = GLMGAMModel(n_splines=10, spline_order=3, lam=0.6, max_iter=100)
    gam.fit(X_train, y_train)
    y_val_pred = gam.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)
    print(f"  Validation R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.6f}")
    models['GAM'] = gam
    
    # TabNAM
    print("\n[2/2] Training TabNAM...")
    tabnam = TabNAMModel(n_d=8, n_a=8, n_steps=3, lambda_sparse=1e-3, seed=42, verbose=0)
    tabnam.max_epochs = 50
    tabnam.patience = 10
    tabnam.batch_size = 256
    tabnam.fit(X_train, y_train)
    y_val_pred = tabnam.predict(X_val)
    metrics = calculate_metrics(y_val, y_val_pred)
    print(f"  Validation R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.6f}")
    models['TabNAM'] = tabnam
    
    return models


def compute_shap_values_gam(model, X_sample, feature_names):
    """GAMのSHAP値を計算（高速）"""
    print("\n" + "=" * 80)
    print("Computing SHAP Values for GAM")
    print("=" * 80)
    
    # LinearExplainerを使用（GAMは線形なので高速）
    def predict_fn(X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
        return model.predict(X_df)
    
    if isinstance(X_sample, pd.DataFrame):
        X_array = X_sample.values
    else:
        X_array = X_sample
    
    # TreeExplainerの代わりにKernelExplainerを使用
    background = shap.sample(X_array, 100)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    # サンプル数を制限
    if len(X_array) > 500:
        sample_indices = np.random.choice(len(X_array), 500, replace=False)
        X_for_shap = X_array[sample_indices]
    else:
        X_for_shap = X_array
    
    print(f"Computing SHAP values for {len(X_for_shap)} samples...")
    shap_values = explainer.shap_values(X_for_shap)
    
    print(f"SHAP computation completed! Shape: {shap_values.shape}")
    return shap_values, X_for_shap


def visualize_feature_importance(shap_values, X, feature_names, model_name, output_dir):
    """特徴量重要度の可視化"""
    print(f"\n[{model_name}] Generating Feature Importance Plot...")
    
    # SHAP値の絶対値の平均を計算
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_importance
    }).sort_values('Importance', ascending=False)
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'])
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title(f'{model_name}: Feature Importance (Material Degradation v0.6)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    
    output_path = output_dir / f'{model_name.lower()}_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    return importance_df


def visualize_shap_summary(shap_values, X, feature_names, model_name, output_dir):
    """SHAP Summary Plot"""
    print(f"\n[{model_name}] Generating SHAP Summary Plot...")
    
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f'{model_name}: SHAP Summary (Material Degradation v0.6)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = output_dir / f'{model_name.lower()}_shap_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_mechanism_interactions(shap_values, X, feature_names, model_name, output_dir):
    """劣化メカニズムの相互作用を可視化"""
    print(f"\n[{model_name}] Analyzing Mechanism Interactions...")
    
    # メカニズム特徴量のインデックスを取得
    mechanism_features = [f for f in feature_names if f.startswith('mechanism_')]
    mechanism_indices = [feature_names.index(f) for f in mechanism_features]
    
    if len(mechanism_indices) == 0:
        print("  No mechanism features found")
        return
    
    # 各メカニズムのSHAP値
    mechanism_shap = shap_values[:, mechanism_indices]
    
    # メカニズム名を短縮
    mechanism_labels = [f.replace('mechanism_', '').replace('_', ' ').title() 
                        for f in mechanism_features]
    
    # 1. メカニズムごとのSHAP値分布
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(mechanism_labels))
    parts = ax.violinplot(mechanism_shap, positions=positions, widths=0.7, 
                           showmeans=True, showmedians=True)
    
    # 色付け
    for pc in parts['bodies']:
        pc.set_facecolor('#8B0000')
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(mechanism_labels, rotation=45, ha='right')
    ax.set_ylabel('SHAP Value', fontsize=12)
    ax.set_title(f'{model_name}: Degradation Mechanism SHAP Distribution', 
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f'{model_name.lower()}_mechanism_shap_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    # 2. メカニズム間の相関（SHAP値ベース）
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = np.corrcoef(mechanism_shap.T)
    im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(len(mechanism_labels)))
    ax.set_yticks(np.arange(len(mechanism_labels)))
    ax.set_xticklabels(mechanism_labels, rotation=45, ha='right')
    ax.set_yticklabels(mechanism_labels)
    
    # 値を表示
    for i in range(len(mechanism_labels)):
        for j in range(len(mechanism_labels)):
            text = ax.text(j, i, f'{correlation[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title(f'{model_name}: Mechanism Interaction Correlation (SHAP-based)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'{model_name.lower()}_mechanism_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_multi_vs_single_mechanism(preprocessor, shap_values, X, feature_names, 
                                        model_name, output_dir):
    """複合劣化 vs 単一劣化の比較"""
    print(f"\n[{model_name}] Comparing Multi-Mechanism vs Single-Mechanism...")
    
    # is_multi_mechanismのインデックス
    try:
        multi_idx = feature_names.index('is_multi_mechanism')
    except ValueError:
        print("  is_multi_mechanism feature not found")
        return
    
    # 元データから複合劣化フラグを取得
    is_multi = X[:, multi_idx]
    
    # 複合劣化と単一劣化のSHAP値を比較
    multi_mask = is_multi == 1
    single_mask = is_multi == 0
    
    if multi_mask.sum() == 0:
        print(f"  No multi-mechanism samples found")
        return
    
    shap_multi = shap_values[multi_mask]
    shap_single = shap_values[single_mask]
    
    # 各特徴量の平均SHAP値を比較
    mean_shap_multi = np.abs(shap_multi).mean(axis=0)
    mean_shap_single = np.abs(shap_single).mean(axis=0)
    
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Multi-Mechanism': mean_shap_multi,
        'Single-Mechanism': mean_shap_single,
        'Difference': mean_shap_multi - mean_shap_single
    }).sort_values('Difference', ascending=False)
    
    # プロット
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison_df['Multi-Mechanism'], width, 
                   label='Multi-Mechanism', color='#D62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, comparison_df['Single-Mechanism'], width,
                   label='Single-Mechanism', color='#1F77B4', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
    ax.set_ylabel('Mean |SHAP value|', fontsize=12)
    ax.set_title(f'{model_name}: Feature Importance Comparison\n'
                 f'Multi-Mechanism ({multi_mask.sum()}) vs Single-Mechanism ({single_mask.sum()})',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f'{model_name.lower()}_multi_vs_single.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    # 差が大きい特徴量を表示
    print(f"\n  Top features with difference (Multi - Single):")
    top_diff = comparison_df.head(10)
    for _, row in top_diff.iterrows():
        print(f"    {row['Feature']:40s}: {row['Difference']:+.4f}")


def main():
    """メイン実行関数"""
    print("\n" + "=" * 80)
    print("MATERIAL DEGRADATION v0.6 - SHAP ANALYSIS")
    print("Multi-Mechanism Degradation Interaction Analysis")
    print("=" * 80)
    
    # 出力ディレクトリ
    output_dir = Path('outputs/figures/material_v0.6')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    print("\n--- Loading Data ---")
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
    feature_names = splits['feature_names']
    
    # モデル訓練
    models = train_models(X_train, y_train, X_val, y_val)
    
    # GAMのSHAP分析
    print("\n" + "=" * 80)
    print("GAM SHAP ANALYSIS")
    print("=" * 80)
    
    shap_values_gam, X_shap_gam = compute_shap_values_gam(
        models['GAM'], X_test, feature_names
    )
    
    importance_df_gam = visualize_feature_importance(
        shap_values_gam, X_shap_gam, feature_names, 'GAM', output_dir
    )
    
    visualize_shap_summary(shap_values_gam, X_shap_gam, feature_names, 'GAM', output_dir)
    
    visualize_mechanism_interactions(
        shap_values_gam, X_shap_gam, feature_names, 'GAM', output_dir
    )
    
    visualize_multi_vs_single_mechanism(
        preprocessor, shap_values_gam, X_shap_gam, feature_names, 'GAM', output_dir
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\n主要な発見:")
    print("1. 特徴量重要度ランキング")
    print(importance_df_gam.head(10).to_string(index=False))
    print("\n2. 複合劣化メカニズムの相互作用が可視化されました")
    print("3. 論文の図として使用できる高品質な画像を生成しました")


if __name__ == '__main__':
    import sys
    np.random.seed(42)
    main()
