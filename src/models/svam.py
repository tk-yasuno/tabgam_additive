"""
SVAM (Support Vector Additive Models)
線形SVMを使用した加法的モデル

- 各特徴量ごとにLinear SVRを学習
- 線形SVMの係数から加法的寄与を計算
- スパース性とロバスト性を持つ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class SVAM:
    """
    Support Vector Additive Model using Linear SVR
    
    h_i ≈ φ_0 + Σ φ_j(x_ij)
    各特徴量ごとに独立したLinear SVRを学習し、それらを合成
    """
    
    def __init__(self,
                 epsilon: float = 0.1,
                 C: float = 1.0,
                 loss: str = 'epsilon_insensitive',
                 max_iter: int = 2000,
                 tol: float = 1e-4,
                 random_state: int = 42,
                 verbose: int = 0):
        """
        Parameters
        ----------
        epsilon : float
            Epsilon in the epsilon-SVR model (イプシロン管)
        C : float
            正則化パラメータ（大きいほど誤差を許容しない）
        loss : str
            損失関数: 'epsilon_insensitive' or 'squared_epsilon_insensitive'
        max_iter : int
            最大反復回数
        tol : float
            収束判定の閾値
        random_state : int
            乱数シード
        verbose : int
            詳細出力レベル
        """
        self.epsilon = epsilon
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        self.feature_models = []
        self.feature_scalers = []
        self.feature_names = None
        self.intercept_ = 0.0
        self.n_features_ = None
        
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
        # DataFrameの場合は特徴量名を保存
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
            self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        
        self.n_features_ = X_array.shape[1]
        
        # 各特徴量ごとにSVRを学習
        self.feature_models = []
        self.feature_scalers = []
        
        # ターゲットの平均をinterceptとして使用
        self.intercept_ = np.mean(y)
        
        # 残差を計算
        residual = y - self.intercept_
        
        if self.verbose > 0:
            print(f"Training {self.n_features_} individual SVR models...")
        
        for j in range(self.n_features_):
            # 特徴量を正規化
            scaler = StandardScaler()
            X_j = X_array[:, j].reshape(-1, 1)
            X_j_scaled = scaler.fit_transform(X_j)
            
            # Linear SVR学習
            model = LinearSVR(
                epsilon=self.epsilon,
                C=self.C,
                loss=self.loss,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state + j,
                verbose=0 if self.verbose == 0 else 1
            )
            
            # 各特徴量は残差の一部を予測
            # 単純化のため、各特徴量は全体の残差を予測しようとする
            model.fit(X_j_scaled, residual / self.n_features_)
            
            self.feature_models.append(model)
            self.feature_scalers.append(scaler)
            
            if self.verbose > 0 and (j + 1) % 10 == 0:
                print(f"  Trained {j + 1}/{self.n_features_} models")
        
        if self.verbose > 0:
            print("Training completed!")
        
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
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # intercept（ベース予測値）から開始
        y_pred = np.full(X_array.shape[0], self.intercept_)
        
        # 各特徴量の寄与を加算
        for j in range(self.n_features_):
            X_j = X_array[:, j].reshape(-1, 1)
            X_j_scaled = self.feature_scalers[j].transform(X_j)
            y_pred += self.feature_models[j].predict(X_j_scaled)
        
        return y_pred
    
    def get_feature_contributions(self, X):
        """
        各特徴量の寄与を取得
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
            
        Returns
        -------
        pd.DataFrame
            各特徴量の寄与（サンプル × 特徴量）
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        contributions = np.zeros((X_array.shape[0], self.n_features_))
        
        for j in range(self.n_features_):
            X_j = X_array[:, j].reshape(-1, 1)
            X_j_scaled = self.feature_scalers[j].transform(X_j)
            contributions[:, j] = self.feature_models[j].predict(X_j_scaled)
        
        return pd.DataFrame(contributions, columns=self.feature_names)
    
    def plot_shape_functions(self, X, feature_indices: List[int] = None, 
                            n_points: int = 100, save_path: str = None):
        """
        個別の形状関数をプロット
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量（範囲決定のため）
        feature_indices : list of int, optional
            プロットする特徴量のインデックス（Noneの場合は全て）
        n_points : int
            プロットする点数
        save_path : str, optional
            保存パス
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if feature_indices is None:
            feature_indices = range(min(self.n_features_, 9))  # 最大9個
        
        n_plots = len(feature_indices)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, j in enumerate(feature_indices):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # 特徴量の範囲
            x_min, x_max = X_array[:, j].min(), X_array[:, j].max()
            x_range = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
            
            # 形状関数
            x_scaled = self.feature_scalers[j].transform(x_range)
            y_shape = self.feature_models[j].predict(x_scaled)
            
            ax.plot(x_range, y_shape, 'b-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel(self.feature_names[j], fontsize=10)
            ax.set_ylabel('Contribution', fontsize=10)
            ax.set_title(f'{self.feature_names[j]}', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # 空のサブプロットを非表示
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Shape functions plot saved to {save_path}")
        
        plt.show()


class SVAMModel:
    """
    SVAMモデルのラッパークラス（他のモデルとのインターフェース統一）
    """
    def __init__(self, epsilon: float = 0.1, C: float = 1.0,
                 max_iter: int = 2000, random_state: int = 42, verbose: int = 0):
        self.model = SVAM(
            epsilon=epsilon,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
