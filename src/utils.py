"""
共通ユーティリティ関数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False


def save_model(model: Any, model_name: str, output_dir: str = None):
    """
    モデルの保存
    
    Parameters
    ----------
    model : Any
        保存するモデルオブジェクト
    model_name : str
        モデル名
    output_dir : str, optional
        出力ディレクトリ
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "models"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.pkl"
    
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    return model_path


def load_model(model_name: str, output_dir: str = None):
    """
    モデルの読み込み
    
    Parameters
    ----------
    model_name : str
        モデル名
    output_dir : str, optional
        モデルディレクトリ
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "models"
    else:
        output_dir = Path(output_dir)
    
    model_path = output_dir / f"{model_name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def save_figure(fig, fig_name: str, output_dir: str = None, dpi: int = 300):
    """
    図の保存
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        保存する図
    fig_name : str
        図のファイル名（拡張子なし）
    output_dir : str, optional
        出力ディレクトリ
    dpi : int
        解像度
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "figures"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PNG形式で保存
    fig_path = output_dir / f"{fig_name}.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    
    return fig_path


def save_results(results: Dict, results_name: str, output_dir: str = None):
    """
    結果の保存
    
    Parameters
    ----------
    results : dict
        保存する結果（辞書形式）
    results_name : str
        結果ファイル名
    output_dir : str, optional
        出力ディレクトリ
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DataFrameに変換可能な場合はCSVで保存
    if isinstance(results, pd.DataFrame):
        results_path = output_dir / f"{results_name}.csv"
        results.to_csv(results_path, index=False, encoding='utf-8-sig')
    else:
        # それ以外はpickleworkで保存
        results_path = output_dir / f"{results_name}.pkl"
        joblib.dump(results, results_path)
    
    print(f"Results saved to: {results_path}")
    return results_path


def plot_feature_contribution(x_values: np.ndarray, 
                              contributions: np.ndarray,
                              feature_name: str,
                              title: str = None,
                              xlabel: str = None,
                              ylabel: str = "Contribution to h_i"):
    """
    特徴量の寄与曲線をプロット
    
    Parameters
    ----------
    x_values : np.ndarray
        特徴量の値
    contributions : np.ndarray
        寄与度（加法的効果）
    feature_name : str
        特徴量名
    title : str, optional
        タイトル
    xlabel : str, optional
        x軸ラベル
    ylabel : str
        y軸ラベル
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # ソート
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    y_sorted = contributions[sort_idx]
    
    # プロット
    ax.plot(x_sorted, y_sorted, linewidth=2, color='steelblue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(x_sorted, 0, y_sorted, alpha=0.3, color='steelblue')
    
    # ラベル
    if title is None:
        title = f"{feature_name}の加法的寄与"
    if xlabel is None:
        xlabel = feature_name
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_cv_summary(cv_results: List[Dict]) -> pd.DataFrame:
    """
    CVの結果をサマリDataFrameに変換
    
    Parameters
    ----------
    cv_results : list of dict
        各foldの結果
        
    Returns
    -------
    pd.DataFrame
        CVサマリ
    """
    df = pd.DataFrame(cv_results)
    
    # 統計量を追加
    summary = df.mean().to_frame(name='mean').T
    summary = pd.concat([
        summary,
        df.std().to_frame(name='std').T,
        df.min().to_frame(name='min').T,
        df.max().to_frame(name='max').T
    ])
    
    return df, summary


def print_cv_results(cv_results: List[Dict], model_name: str = "Model"):
    """
    CV結果の表示
    
    Parameters
    ----------
    cv_results : list of dict
        各foldの結果
    model_name : str
        モデル名
    """
    df, summary = create_cv_summary(cv_results)
    
    print("=" * 60)
    print(f"{model_name} - Cross Validation Results")
    print("=" * 60)
    print("\nFold-wise Results:")
    print(df.to_string(index=False))
    print("\nSummary Statistics:")
    print(summary.to_string())
    print("=" * 60)
    

def compare_models_plot(model_results: Dict[str, Dict],
                       metrics: List[str] = ['rmse', 'mae', 'r2']):
    """
    複数モデルの性能比較プロット
    
    Parameters
    ----------
    model_results : dict
        {model_name: {'rmse': value, 'mae': value, 'r2': value}}
    metrics : list of str
        比較する指標
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    model_names = list(model_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        values = [model_results[name].get(metric, 0) for name in model_names]
        
        bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_title(metric.upper(), fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def print_model_comparison(model_results: Dict[str, Dict]):
    """
    モデル比較結果の表形式表示
    
    Parameters
    ----------
    model_results : dict
        {model_name: {'rmse': value, 'mae': value, 'r2': value}}
    """
    df = pd.DataFrame(model_results).T
    
    print("=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)
    
    # ベストモデルを特定
    if 'rmse' in df.columns:
        best_rmse = df['rmse'].idxmin()
        print(f"\nBest RMSE: {best_rmse} ({df.loc[best_rmse, 'rmse']:.6f})")
    
    if 'r2' in df.columns:
        best_r2 = df['r2'].idxmax()
        print(f"Best R²: {best_r2} ({df.loc[best_r2, 'r2']:.6f})")
    
    return df


if __name__ == '__main__':
    print("Utility functions loaded successfully!")
