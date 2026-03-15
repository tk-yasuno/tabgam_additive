"""
GLM-GAM (Generalized Additive Model) with Splines
スプライン付き一般化加法モデル

pyGAMを使用した実装
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f, l
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GLMGAMModel:
    """
    GLM-GAM Model for Hazard Prediction
    
    h_i = β_0 + Σ f_j(x_ij) + Σ g_k(z_ik)
    
    - f_j, g_k: B-spline or P-spline
    - Link function: identity (Gaussian) or log (Poisson)
    """
    
    def __init__(self,
                 continuous_features: List[str] = None,
                 categorical_features: List[str] = None,
                 n_splines: int = 10,
                 spline_order: int = 3,
                 lam: float = None,
                 fit_intercept: bool = True,
                 max_iter: int = 100,
                 verbose: bool = False):
        """
        Parameters
        ----------
        continuous_features : list of str
            連続変数の特徴量名
        categorical_features : list of str
            カテゴリカル変数の特徴量名
        n_splines : int
            スプライン基底数
        spline_order : int
            スプライン次数
        lam : float, optional
            正則化パラメータ（Noneの場合はCV自動選択）
        fit_intercept : bool
            切片項を含めるか
        max_iter : int
            最大イテレーション数
        verbose : bool
            詳細出力
        """
        self.continuous_features = continuous_features or []
        self.categorical_features = categorical_features or []
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.model = None
        self.feature_names = None
        self.feature_idx_map = {}
        
    def _build_gam_formula(self, X: pd.DataFrame):
        """
        GAMの式を構築
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量DataFrame
        """
        terms = []
        
        # 連続変数にはスプライン
        for feat in self.continuous_features:
            if feat in X.columns:
                idx = list(X.columns).index(feat)
                self.feature_idx_map[feat] = idx
                terms.append(s(idx, n_splines=self.n_splines, spline_order=self.spline_order))
        
        # カテゴリカル変数には因子項
        for feat in self.categorical_features:
            if feat in X.columns:
                idx = list(X.columns).index(feat)
                self.feature_idx_map[feat] = idx
                terms.append(f(idx))
        
        # Properly combine terms for pygam
        if not terms:
            return None
        elif len(terms) == 1:
            return terms[0]
        else:
            # Start with first term and add rest
            formula = terms[0]
            for term in terms[1:]:
                formula = formula + term
            return formula
    
    def fit(self, X, y, lam=None):
        """
        モデル学習
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            ターゲット
        lam : float, optional
            正則化パラメータ（Noneの場合はCV）
        """
        # DataFrameに変換
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        self.feature_names = list(X.columns)
        
        # 特徴量が指定されていない場合は自動設定
        if not self.continuous_features and not self.categorical_features:
            # 全て連続変数として扱う
            self.continuous_features = self.feature_names
        
        # GAM式を構築
        formula = self._build_gam_formula(X)
        
        if formula is None:
            raise ValueError("No valid features found for GAM")
        
        # モデル作成
        if lam is not None:
            self.lam = lam
        
        if self.lam is not None:
            # 指定されたlamで学習
            self.model = LinearGAM(
                formula,
                lam=self.lam,  # 初期化時に指定
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                verbose=self.verbose
            )
            self.model.fit(X.values, y)  # lamは既に設定済み
        else:
            # Grid searchでlamを自動選択
            self.model = LinearGAM(
                formula,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                verbose=self.verbose
            )
            # lamをgrid searchで最適化
            lam_grid = np.logspace(-3, 3, 11)
            self.model.gridsearch(X.values, y, lam=lam_grid, progress=self.verbose)
            self.lam = self.model.lam
        
        if self.verbose:
            print(f"Model fitted with lam={self.lam}")
            print(f"AIC: {self.model.statistics_['AIC']:.2f}")
            print(f"AICc: {self.model.statistics_['AICc']:.2f}")
        
        return self
    
    def predict(self, X):
        """
        予測
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
            
        Returns
        -------
        np.ndarray
            予測値
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def get_feature_contributions(self, X, feature_name: str):
        """
        特定特徴量の加法的寄与を取得
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        feature_name : str
            特徴量名
            
        Returns
        -------
        tuple
            (x_values, contributions)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        if feature_name not in self.feature_idx_map:
            raise ValueError(f"Feature '{feature_name}' not found in model")
        
        feat_idx = self.feature_idx_map[feature_name]
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # 部分依存を計算（手動でグリッド生成）
        # X_array の平均値を使い、特定の特徴量だけを変動させる
        n_grid = 200
        X_mean = X_array.mean(axis=0)
        
        # グリッド生成: 特定特徴量の範囲で均等に分割
        x_min = X_array[:, feat_idx].min()
        x_max = X_array[:, feat_idx].max()
        x_values = np.linspace(x_min, x_max, n_grid)
        
        # 部分依存グリッド作成（他の特徴量は平均値）
        XX = np.tile(X_mean, (n_grid, 1))
        XX[:, feat_idx] = x_values
        
        contributions = self.model.partial_dependence(term=feat_idx, X=XX)
        
        return x_values, contributions
    
    def plot_feature_contributions(self, X, feature_names: List[str] = None,
                                   figsize: Tuple[int, int] = None):
        """
        特徴量の寄与曲線をプロット
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        feature_names : list of str, optional
            プロットする特徴量名（Noneの場合は全て）
        figsize : tuple, optional
            図のサイズ
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        if feature_names is None:
            feature_names = list(self.feature_idx_map.keys())
        
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (6 * n_cols, 5 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        for i, feat_name in enumerate(feature_names):
            ax = axes[i]
            
            try:
                x_values, contributions = self.get_feature_contributions(X, feat_name)
                
                ax.plot(x_values, contributions, linewidth=2, color='steelblue')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.fill_between(x_values, 0, contributions, alpha=0.3, color='steelblue')
                
                ax.set_title(f'{feat_name}の寄与', fontsize=12, fontweight='bold')
                ax.set_xlabel(feat_name, fontsize=10)
                ax.set_ylabel('Contribution to h_i', fontsize=10)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feat_name, fontsize=12)
        
        # 余分な軸を非表示
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self) -> Dict:
        """
        モデルのサマリ情報を取得
        
        Returns
        -------
        dict
            サマリ情報
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        summary = {
            'n_features': len(self.feature_idx_map),
            'continuous_features': self.continuous_features,
            'categorical_features': self.categorical_features,
            'n_splines': self.n_splines,
            'spline_order': self.spline_order,
            'lam': self.lam,
            'statistics': self.model.statistics_
        }
        
        return summary
    
    def print_summary(self):
        """モデルサマリの表示"""
        summary = self.get_model_summary()
        
        print("=" * 60)
        print("GLM-GAM Model Summary")
        print("=" * 60)
        print(f"Number of features: {summary['n_features']}")
        print(f"Continuous features: {len(summary['continuous_features'])}")
        print(f"Categorical features: {len(summary['categorical_features'])}")
        print(f"N splines: {summary['n_splines']}")
        print(f"Spline order: {summary['spline_order']}")
        print(f"Lambda: {summary['lam']:.6f}")
        print("\nModel Statistics:")
        for key, val in summary['statistics'].items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}")
        print("=" * 60)


if __name__ == '__main__':
    # テスト
    print("GLM-GAM Model implementation loaded successfully!")
    
    # ダミーデータでテスト
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'age': np.random.uniform(0, 50, n_samples),
        'length': np.random.uniform(10, 200, n_samples),
        'width': np.random.uniform(5, 15, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # 非線形な関係を持つターゲット
    y = (0.1 * X['age']**2 + 0.05 * X['length'] + 
         np.random.randn(n_samples) * 5)
    
    # モデル学習
    model = GLMGAMModel(
        continuous_features=['age', 'length', 'width'],
        n_splines=10,
        verbose=True
    )
    model.fit(X, y)
    
    # 予測
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    print(f"\nTest RMSE: {rmse:.4f}")
    
    # サマリ表示
    model.print_summary()
