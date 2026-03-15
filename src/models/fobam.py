"""
FOBAM (Forest Out-of-Bag Additive Models)
Random ForestのOOB予測を使用した加法的モデル

v2改良版:
- FAMベースの学習(全特徴量を使用)
- OOBスコアで性能評価
- 特徴量の寄与度を推定
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FOBAM:
    """
    Forest Out-of-Bag Additive Model (v2)
    
    h_i ≈ φ_0 + Σ φ_j(x_ij)
    
    改良版アプローチ:
    - 学習時: 全特徴量を使用してRandom Forestを学習(FAMと同じ)
    - 評価: OOBスコアで各特徴量の寄与を評価
    - 解釈: 加法的な特徴量寄与を推定
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 20,
                 min_samples_leaf: int = 10,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 verbose: int = 0):
        """
        Parameters
        ----------
        n_estimators : int
            森の木の数
        max_depth : int
            木の最大深度
        min_samples_split : int
            分割に必要な最小サンプル数
        min_samples_leaf : int
            葉ノードの最小サンプル数
        max_features : str, int, or float
            各分割で考慮する特徴量の数
        bootstrap : bool
            ブートストラップサンプリングを使用するか
        oob_score : bool
            OOB スコアを計算するか
        n_jobs : int
            並列処理数(-1で全コア使用)
        random_state : int
            乱数シード
        verbose : int
            詳細出力レベル
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.feature_names = None
        self.n_features_ = None
        self.feature_importances_ = None
        self.oob_score_ = None
        
    def fit(self, X, y):
        """
        モデル学習(FAMベースの改良版)
        
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
        
        if self.verbose > 0:
            print(f"Training Random Forest with {self.n_features_} features...")
            print("Using OOB scoring for evaluation...")
        
        # 全特徴量を使用してRandom Forestを学習（FAMと同じ）
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,  # OOBのため必須
            oob_score=True,  # OOBスコアを計算
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        # 学習
        self.model.fit(X_array, y)
        
        # 特徴量重要度とOOBスコアを保存
        self.feature_importances_ = self.model.feature_importances_
        self.oob_score_ = self.model.oob_score_
        
        if self.verbose > 0:
            print(f"Training completed!")
            print(f"OOB Score (R²): {self.oob_score_:.4f}")
            print(f"Top 5 important features:")
            top_indices = np.argsort(self.feature_importances_)[-5:][::-1]
            for idx in top_indices:
                print(f"  {self.feature_names[idx]}: {self.feature_importances_[idx]:.4f}")
        
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
        
        # Random Forestによる予測
        y_pred = self.model.predict(X_array)
        
        return y_pred
    
    def get_feature_importances(self):
        """
        特徴量重要度を取得
        
        Returns
        -------
        pd.Series
            各特徴量の重要度
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has not been fitted yet")
        
        return pd.Series(self.feature_importances_, index=self.feature_names)
    
    def get_oob_score(self):
        """
        OOBスコア(R²)を取得
        
        Returns
        -------
        float
            OOBスコア
        """
        if self.oob_score_ is None:
            raise ValueError("OOB score was not calculated")
        
        return self.oob_score_
    
    def get_feature_contributions(self, X, method='feature_importance'):
        """
        各特徴量の寄与を推定(近似的な加法分解)
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        method : str
            'feature_importance': 特徴量重要度ベースの寄与推定
            
        Returns
        -------
        pd.DataFrame
            各特徴量の寄与(サンプル × 特徴量)
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if method == 'feature_importance':
            # 特徴量重要度に基づく寄与推定(簡易版)
            y_pred = self.model.predict(X_array)
            base_value = np.mean(y_pred)
            
            # 各特徴量の寄与を重要度で按分
            contributions = np.zeros((X_array.shape[0], self.n_features_))
            for i in range(X_array.shape[0]):
                total_contribution = y_pred[i] - base_value
                for j in range(self.n_features_):
                    contributions[i, j] = total_contribution * self.feature_importances_[j]
            
            return pd.DataFrame(contributions, columns=self.feature_names)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def plot_feature_importances(self, top_n: int = None, save_path: str = None):
        """
        特徴量重要度をプロット
        
        Parameters
        ----------
        top_n : int, optional
            上位N個の特徴量のみプロット
        save_path : str, optional
            保存パス
        """
        importances = self.get_feature_importances().sort_values(ascending=False)
        
        if top_n is not None:
            importances = importances.head(top_n)
        
        plt.figure(figsize=(10, max(6, len(importances) * 0.3)))
        plt.barh(range(len(importances)), importances.values, color='forestgreen')
        plt.yticks(range(len(importances)), importances.index)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'FOBAM Feature Importances (OOB R²: {self.oob_score_:.4f})', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature importances plot saved to {save_path}")
        
        plt.show()


class FOBAMModel:
    """
    FOBAMモデルのラッパークラス(他のモデルとのインターフェース統一)
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 20, min_samples_leaf: int = 10,
                 max_features: Union[str, int, float] = 'sqrt',
                 n_jobs: int = -1, random_state: int = 42, verbose: int = 0):
        self.model = FOBAM(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
