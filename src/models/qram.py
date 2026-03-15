"""
QRAM (Quantile Regression Additive Models)
分位回帰加法モデル

平均ではなく、上位分位（τ = 0.75, 0.9など）のハザードを推定
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class QRAMModel:
    """
    Quantile Regression Additive Model
    
    Q_τ(h_i | x_i, z_i) = β_0,τ + Σ f_j,τ(x_ij) + Σ g_k,τ(z_ik)
    
    まずは線形分位回帰をベースとする
    """
    
    def __init__(self,
                 quantiles: List[float] = [0.5, 0.75, 0.9],
                 max_iter: int = 1000,
                 p_tol: float = 1e-6,
                 verbose: bool = False):
        """
        Parameters
        ----------
        quantiles : list of float
            推定する分位点（0 < q < 1）
        max_iter : int
            最適化の最大イテレーション数
        p_tol : float
            収束判定の許容誤差
        verbose : bool
            詳細出力
        """
        self.quantiles = quantiles
        self.max_iter = max_iter
        self.p_tol = p_tol
        self.verbose = verbose
        
        self.models = {}  # {quantile: model}
        self.feature_names = None
        
    def fit(self, X, y):
        """
        モデル学習
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            ターゲット
        """
        # DataFrameに変換
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        
        self.feature_names = list(X.columns)
        
        # 切片項を追加
        X_with_const = sm.add_constant(X)
        
        # 各分位点でモデルを学習
        for q in self.quantiles:
            if self.verbose:
                print(f"Fitting quantile regression for τ={q}...")
            
            model = QuantReg(y, X_with_const)
            result = model.fit(
                q=q,
                max_iter=self.max_iter,
                p_tol=self.p_tol
            )
            
            self.models[q] = result
            
            if self.verbose:
                print(f"  Converged: {result.converged}")
                print(f"  Pseudo R²: {result.prsquared:.4f}")
        
        return self
    
    def predict(self, X, quantile: float = None):
        """
        予測
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        quantile : float, optional
            予測する分位点（Noneの場合は中央値0.5）
            
        Returns
        -------
        np.ndarray or dict
            quantile指定時: 予測値（np.ndarray）
            quantile未指定: {quantile: 予測値} の辞書
        """
        if not self.models:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # DataFrameに変換
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 切片項を追加
        X_with_const = sm.add_constant(X)
        
        if quantile is not None:
            # 指定された分位点で予測
            if quantile not in self.models:
                raise ValueError(f"Quantile {quantile} not fitted. "
                               f"Available: {list(self.models.keys())}")
            return self.models[quantile].predict(X_with_const)
        else:
            # 全分位点で予測
            predictions = {}
            for q, model in self.models.items():
                predictions[q] = model.predict(X_with_const)
            return predictions
    
    def get_coefficients(self, quantile: float) -> pd.DataFrame:
        """
        指定分位点の係数を取得
        
        Parameters
        ----------
        quantile : float
            分位点
            
        Returns
        -------
        pd.DataFrame
            係数のDataFrame
        """
        if quantile not in self.models:
            raise ValueError(f"Quantile {quantile} not fitted.")
        
        result = self.models[quantile]
        
        df = pd.DataFrame({
            'feature': ['const'] + self.feature_names,
            'coef': result.params.values,
            'std_err': result.bse.values,
            'pvalue': result.pvalues.values
        })
        
        return df
    
    def plot_quantile_predictions(self, X, y, n_samples: int = 1000):
        """
        分位予測の可視化
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            実測値
        n_samples : int
            プロットするサンプル数
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # サンプリング
        if len(y) > n_samples:
            indices = np.random.choice(len(y), n_samples, replace=False)
            X_sample = X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # 全分位点で予測
        predictions = self.predict(X_sample)
        
        # ソート（プロット用）
        sort_idx = np.argsort(y_sample)
        y_sorted = y_sample[sort_idx]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 実測値
        x_axis = np.arange(len(y_sorted))
        ax.scatter(x_axis, y_sorted, alpha=0.3, s=10, label='Actual', color='black')
        
        # 各分位点の予測
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(self.quantiles)))
        
        for i, (q, pred) in enumerate(sorted(predictions.items())):
            pred_sorted = pred[sort_idx]
            ax.plot(x_axis, pred_sorted, linewidth=2, 
                   label=f'τ={q}', color=colors[i])
        
        ax.set_title('Quantile Regression Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample (sorted by actual value)', fontsize=12)
        ax.set_ylabel('h_i', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_coefficient_paths(self, top_k: int = 10):
        """
        分位点ごとの係数の変化をプロット
        
        Parameters
        ----------
        top_k : int
            表示する上位K個の特徴量
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # 各分位点の係数を集約
        coef_data = {}
        for q in sorted(self.quantiles):
            result = self.models[q]
            coef_data[q] = result.params.values[1:]  # 切片を除く
        
        df_coefs = pd.DataFrame(coef_data, index=self.feature_names)
        
        # 平均絶対係数が大きい順に選択
        mean_abs_coefs = df_coefs.abs().mean(axis=1).sort_values(ascending=False)
        top_features = mean_abs_coefs.head(top_k).index
        
        df_top = df_coefs.loc[top_features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for feat in df_top.index:
            ax.plot(df_top.columns, df_top.loc[feat], marker='o', 
                   linewidth=2, label=feat)
        
        ax.set_title('Coefficient Paths Across Quantiles', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Quantile (τ)', fontsize=12)
        ax.set_ylabel('Coefficient', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compute_quantile_loss(self, X, y, quantile: float) -> float:
        """
        分位損失（Pinball Loss）の計算
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            実測値
        quantile : float
            分位点
            
        Returns
        -------
        float
            分位損失
        """
        y_pred = self.predict(X, quantile)
        errors = y - y_pred
        loss = np.where(errors >= 0,
                       quantile * errors,
                       (quantile - 1) * errors)
        return np.mean(loss)
    
    def compute_quantile_coverage(self, X, y, quantile: float) -> float:
        """
        分位カバレッジの計算
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            実測値
        quantile : float
            分位点
            
        Returns
        -------
        float
            カバレッジ率（理想は quantile に近い値）
        """
        y_pred = self.predict(X, quantile)
        coverage = np.mean(y <= y_pred)
        return coverage
    
    def evaluate_quantiles(self, X, y) -> pd.DataFrame:
        """
        全分位点での評価
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            実測値
            
        Returns
        -------
        pd.DataFrame
            評価結果
        """
        results = []
        
        for q in self.quantiles:
            loss = self.compute_quantile_loss(X, y, q)
            coverage = self.compute_quantile_coverage(X, y, q)
            
            results.append({
                'quantile': q,
                'pinball_loss': loss,
                'coverage': coverage,
                'coverage_error': abs(coverage - q)
            })
        
        return pd.DataFrame(results)
    
    def print_summary(self):
        """モデルサマリの表示"""
        print("=" * 60)
        print("QRAM (Quantile Regression Additive Model) Summary")
        print("=" * 60)
        print(f"Quantiles: {self.quantiles}")
        print(f"N features: {len(self.feature_names)}")
        
        for q in self.quantiles:
            result = self.models[q]
            print(f"\nQuantile τ={q}:")
            print(f"  Converged: {result.converged}")
            print(f"  Pseudo R²: {result.prsquared:.4f}")
            print(f"  N iterations: {result.n_iter}")
        
        print("=" * 60)


if __name__ == '__main__':
    # テスト
    print("QRAM Model implementation loaded successfully!")
    
    # ダミーデータでテスト（分散が不均一）
    np.random.seed(42)
    n_samples = 2000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    
    # 条件付き分散が不均一なターゲット
    noise_scale = 0.5 + 0.5 * np.abs(X[:, 0])  # X[0]に応じて分散が変化
    y = (2 * X[:, 0] + 1.5 * X[:, 1] + 
         np.random.randn(n_samples) * noise_scale)
    
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    
    # モデル学習
    model = QRAMModel(quantiles=[0.5, 0.75, 0.9], verbose=True)
    model.fit(X_df, y)
    
    # 評価
    eval_results = model.evaluate_quantiles(X_df, y)
    print("\nEvaluation Results:")
    print(eval_results)
    
    # サマリ表示
    model.print_summary()
