"""
論文用の可視化スクリプト（v0.8 - Jet Engine RUL Prediction）

生成する図:
1. モデル性能比較図（R², RMSE, MAE）
2. 重み最適化プロセスの推移
3. 最適重みの円グラフ
4. モデル相関行列（ヒートマップ）
5. アンサンブル比較
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# スタイル設定
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)


def load_results():
    """v0.8の結果を読み込み"""
    # モデル比較結果
    comparison_path = Path('outputs/results/all_models_comparison_v0.8_cmapss.csv')
    df_comparison = pd.read_csv(comparison_path)
    df_comparison = df_comparison.set_index('Model')
    
    # 最適化履歴
    tuning_path = Path('outputs/results/agentic_tuning_v0.8_cmapss.json')
    with open(tuning_path, 'r', encoding='utf-8') as f:
        tuning_results = json.load(f)
    
    return df_comparison, tuning_results


def plot_model_comparison(df_comparison, save_path='outputs/figures/cmapss_v0.8/model_comparison_v0.8.png'):
    """
    Figure 1: モデル性能比較図
    """
    # 個別モデルのみ抽出（アンサンブルは別表示）
    models = ['GAM', 'FAM', 'GBAM', 'TabNAM']
    df_models = df_comparison.loc[models].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # カラーパレット
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # (a) R²
    ax = axes[0]
    bars = ax.bar(models, df_models['r2'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('R² (Coefficient of Determination)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.7)
    ax.set_title('(a) R² Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 値ラベル
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ベストモデルにマーク
    best_idx = df_models['r2'].argmax()
    ax.scatter(best_idx, df_models['r2'].iloc[best_idx] + 0.02, 
               marker='*', s=300, color='gold', edgecolor='black', linewidth=1.5, zorder=10)
    
    # (b) RMSE
    ax = axes[1]
    bars = ax.bar(models, df_models['rmse'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('RMSE (Root Mean Square Error)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 20)
    ax.set_title('(b) RMSE Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 値ラベル
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ベストモデル（最小値）にマーク
    best_idx = df_models['rmse'].argmin()
    ax.scatter(best_idx, df_models['rmse'].iloc[best_idx] - 0.5, 
               marker='*', s=300, color='gold', edgecolor='black', linewidth=1.5, zorder=10)
    
    # (c) MAE
    ax = axes[2]
    bars = ax.bar(models, df_models['mae'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 16)
    ax.set_title('(c) MAE Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 値ラベル
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ベストモデル（最小値）にマーク
    best_idx = df_models['mae'].argmin()
    ax.scatter(best_idx, df_models['mae'].iloc[best_idx] - 0.5, 
               marker='*', s=300, color='gold', edgecolor='black', linewidth=1.5, zorder=10)
    
    plt.suptitle('Jet Engine RUL Prediction (v0.8): Model Performance', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure 1 saved: {save_path}")
    plt.close()


def plot_optimization_process(tuning_results, save_path='outputs/figures/cmapss_v0.8/optimization_process_v0.8.png'):
    """
    Figure 2: 重み最適化プロセスの推移
    """
    history = tuning_results['search_history']
    
    # データ抽出
    iterations = [h['iteration'] for h in history]
    r2_scores = [h['metrics']['r2'] for h in history]
    methods = [h['method'] for h in history]
    is_best = [h['is_best'] for h in history]
    
    # GAM, FAM, GBAM, TabNAMの重み推移
    weights_gam = [h['weights_dict']['GAM'] for h in history]
    weights_fam = [h['weights_dict']['FAM'] for h in history]
    weights_gbam = [h['weights_dict']['GBAM'] for h in history]
    weights_tabnam = [h['weights_dict']['TabNAM'] for h in history]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # (a) R²の推移
    ax = axes[0]
    ax.plot(iterations, r2_scores, marker='o', linewidth=2, markersize=8, 
            color='#3498db', label='Validation R²')
    
    # ベスト更新ポイントを強調
    best_iterations = [i for i, b in zip(iterations, is_best) if b]
    best_r2 = [r for r, b in zip(r2_scores, is_best) if b]
    ax.scatter(best_iterations, best_r2, s=200, color='gold', 
               edgecolor='black', linewidth=2, zorder=10, label='Best Update', marker='*')
    
    # 手法ごとに色分けした背景
    method_colors = {
        'uniform': '#ecf0f1',
        'scipy_SLSQP': '#e8f8f5',
        'differential_evolution': '#fef9e7',
        'individual_best': '#fdecea',
        'random_perturbation': '#f4ecf7'
    }
    
    for i in range(len(iterations) - 1):
        method = methods[i]
        if method in method_colors:
            ax.axvspan(iterations[i], iterations[i+1], alpha=0.2, 
                      color=method_colors.get(method, 'white'))
    
    # 手法アノテーション（最初の4つのみ）
    for i in [0, 1, 2, 3]:
        if i < len(iterations):
            ax.annotate(methods[i].replace('_', '\n'), 
                       xy=(iterations[i], r2_scores[i]),
                       xytext=(iterations[i], r2_scores[i] + 0.005),
                       fontsize=8, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('(a) Optimization Process: R² Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # (b) 重みの推移
    ax = axes[1]
    ax.plot(iterations, weights_gam, marker='o', linewidth=2, markersize=6, 
            label='GAM', color='#3498db')
    ax.plot(iterations, weights_fam, marker='s', linewidth=2, markersize=6, 
            label='FAM', color='#e74c3c')
    ax.plot(iterations, weights_gbam, marker='^', linewidth=2, markersize=6, 
            label='GBAM', color='#2ecc71')
    ax.plot(iterations, weights_tabnam, marker='D', linewidth=2, markersize=6, 
            label='TabNAM', color='#f39c12')
    
    # ベストイテレーションを強調
    best_iter = next((i for i, b in enumerate(is_best) if b and i > 0), 1)
    ax.axvline(best_iter, color='red', linestyle='--', linewidth=2, alpha=0.5, 
               label=f'Best Solution (Iter {best_iter})')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Weight', fontsize=12, fontweight='bold')
    ax.set_title('(b) Weight Evolution across Iterations', fontsize=13, fontweight='bold')
    ax.legend(loc='right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle('Jet Engine RUL v0.8: Agentic Weight Optimization', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure 2 saved: {save_path}")
    plt.close()


def plot_optimal_weights(tuning_results, save_path='outputs/figures/cmapss_v0.8/optimal_weights_v0.8.png'):
    """
    Figure 3: 最適重みの円グラフ
    """
    best_weights = tuning_results['best_weights']
    model_names = tuning_results['model_names']
    
    # best_r2を取得（best_scoreまたはsearch_historyから）
    if 'best_score' in tuning_results:
        best_r2 = tuning_results['best_score']
    else:
        # search_historyから最良のR²を取得
        best_history = [h for h in tuning_results['search_history'] if h['is_best'] and h['iteration'] > 0]
        best_r2 = best_history[0]['metrics']['r2'] if best_history else 0.0
    
    # 非ゼロの重みのみ表示
    non_zero = [(name, weight) for name, weight in zip(model_names, best_weights) if weight > 0.001]
    labels = [name for name, _ in non_zero]
    weights = [weight for _, weight in non_zero]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # カラー設定（v0.8はGAMとGBAMのみ）
    colors = ['#3498db', '#2ecc71']  # GAM, GBAM
    explode = tuple([0.05] * len(labels))
    
    wedges, texts, autotexts = ax.pie(weights, explode=explode, labels=labels, 
                                        autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 14, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    
    # パーセント表示を大きく
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')
    
    ax.set_title(f'Optimal Ensemble Weights\nJet Engine RUL v0.8 (R² = {best_r2:.4f})', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 凡例
    ax.legend(wedges, [f'{label}: {weight:.4f}' for label, weight in zip(labels, weights)],
              title="Model Weights",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=12)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure 3 saved: {save_path}")
    plt.close()


def plot_ensemble_comparison(df_comparison, save_path='outputs/figures/cmapss_v0.8/ensemble_comparison_v0.8.png'):
    """
    Figure 4: アンサンブルと個別モデルの比較
    """
    # データ準備（R²で降順ソート）
    all_models = ['GBAM', 'ENSEMBLE', 'UNIFORM_ENSEMBLE', 'TabNAM', 'FAM', 'GAM']
    df_all = df_comparison.loc[all_models].copy()
    
    # カラー設定
    colors = {
        'GBAM': '#2ecc71',
        'ENSEMBLE': '#e74c3c',  # 赤で強調
        'UNIFORM_ENSEMBLE': '#95a5a6',
        'TabNAM': '#f39c12',
        'FAM': '#bdc3c7',
        'GAM': '#3498db'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 横棒グラフ
    y_pos = np.arange(len(all_models))
    bars = ax.barh(y_pos, df_all['r2'], 
                   color=[colors[model] for model in all_models],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # ENSEMBLE を強調
    bars[1].set_linewidth(3)
    bars[1].set_edgecolor('red')
    
    # GBAM（最高性能）も強調
    bars[0].set_linewidth(3)
    bars[0].set_edgecolor('darkgreen')
    
    # 値ラベル
    for i, (model, r2) in enumerate(zip(all_models, df_all['r2'])):
        ax.text(r2 + 0.01, i, f'{r2:.4f}', 
                va='center', ha='left', fontsize=11, fontweight='bold')
    
    # GBAM にスターマーク（最高性能）
    ax.scatter(df_all.loc['GBAM', 'r2'] + 0.04, 0, 
               marker='*', s=500, color='gold', edgecolor='black', linewidth=2, zorder=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_models, fontsize=12, fontweight='bold')
    ax.set_xlabel('R² (Coefficient of Determination)', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Individual Models vs Ensembles\nJet Engine RUL v0.8', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0.4, 0.7)
    
    # 改善度のアノテーション
    ensemble_r2 = df_all.loc['ENSEMBLE', 'r2']
    uniform_r2 = df_all.loc['UNIFORM_ENSEMBLE', 'r2']
    
    # 矢印で改善を示す
    ax.annotate('', xy=(ensemble_r2, 2.3), xytext=(uniform_r2, 2.3),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text((ensemble_r2 + uniform_r2) / 2, 2.5, 
            f'+{(ensemble_r2 - uniform_r2)*100:.2f}%',
            ha='center', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure 4 saved: {save_path}")
    plt.close()


def create_summary_table(df_comparison, tuning_results, save_path='outputs/figures/cmapss_v0.8/summary_table_v0.8.png'):
    """
    Figure 5: サマリーテーブル（画像として保存）
    """
    # データ準備
    models = ['GAM', 'FAM', 'GBAM', 'TabNAM', 'ENSEMBLE', 'UNIFORM_ENSEMBLE']
    
    table_data = []
    for model in models:
        row = df_comparison.loc[model]
        if model in ['GAM', 'FAM', 'GBAM', 'TabNAM']:
            weight = tuning_results['best_weights'][tuning_results['model_names'].index(model)]
            weight_str = f'{weight:.3f}'
        else:
            weight_str = '-'
        
        table_data.append([
            model,
            f"{row['rmse']:.2f}",
            f"{row['mae']:.2f}",
            f"{row['r2']:.4f}",
            weight_str
        ])
    
    # テーブル作成
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'RMSE', 'MAE', 'R²', 'Optimal Weight'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # ヘッダー行のスタイル
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    
    # GBAMの行を強調（最高性能）
    for i in range(5):
        cell = table[(3, i)]  # GBAM行
        cell.set_facecolor('#e8f8f5')
        cell.set_text_props(weight='bold')
    
    # ENSEMBLEの行を強調
    for i in range(5):
        cell = table[(5, i)]  # ENSEMBLE行
        cell.set_facecolor('#fdecea')
        cell.set_text_props(weight='bold')
    
    # ベストR²値を強調（GBAM）
    table[(3, 3)].set_facecolor('#2ecc71')
    table[(3, 3)].set_text_props(color='white', weight='bold')
    
    plt.title('Model Performance Summary\nJet Engine RUL Prediction v0.8', 
              fontsize=14, fontweight='bold', pad=20)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure 5 saved: {save_path}")
    plt.close()


def main():
    """メイン実行"""
    print("=" * 80)
    print("Generating Figures for Paper (v0.8 - Jet Engine RUL Prediction)")
    print("=" * 80)
    
    # 結果読み込み
    print("\nLoading results...")
    df_comparison, tuning_results = load_results()
    
    print(f"\nModels: {df_comparison.index.tolist()}")
    print(f"Best R²: {df_comparison['r2'].max():.4f}")
    
    # 図の生成
    print("\nGenerating figures...")
    
    print("\n[1/5] Model Comparison...")
    plot_model_comparison(df_comparison)
    
    print("\n[2/5] Optimization Process...")
    plot_optimization_process(tuning_results)
    
    print("\n[3/5] Optimal Weights...")
    plot_optimal_weights(tuning_results)
    
    print("\n[4/5] Ensemble Comparison...")
    plot_ensemble_comparison(df_comparison)
    
    print("\n[5/5] Summary Table...")
    create_summary_table(df_comparison, tuning_results)
    
    print("\n" + "=" * 80)
    print("All figures generated successfully!")
    print("=" * 80)
    print("\nOutput directory: outputs/figures/cmapss_v0.8/")
    print("\nFigures:")
    print("  1. model_comparison_v0.8.png - モデル性能比較")
    print("  2. optimization_process_v0.8.png - 最適化プロセス")
    print("  3. optimal_weights_v0.8.png - 最適重み")
    print("  4. ensemble_comparison_v0.8.png - アンサンブル比較")
    print("  5. summary_table_v0.8.png - サマリーテーブル")
    print("\n論文のFigureとして使用できます。")


if __name__ == "__main__":
    main()
