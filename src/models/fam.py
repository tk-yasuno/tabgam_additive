"""
FAM (Forest Additive Models)
ランダムフォレストベースの加法的モデル

変数重要度 + 部分依存プロット（PDP）で加法的寄与を解釈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FAMModel:
    """
    Forest Additive Model using RandomForest
    
    厳密な加法モデルではないが、以下で加法的解釈を行う：
    - 変数重要度（Gini or Permutation）
    - 部分依存プロット（PDP）
    """
    
    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: int = 20,
                 min_samples_leaf: int = 50,
                 min_samples_split: int = 100,
                 max_features: str = 'sqrt',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 verbose: int = 0):
        """
        Parameters
        ----------
        n_estimators : int
            決定木の数
        max_depth : int
            木の最大深さ
        min_samples_leaf : int
            葉ノードの最小サンプル数
        min_samples_split : int
            分割に必要な最小サンプル数
        max_features : str or int or float
            各分割で考慮する特徴量数
        n_jobs : int
            並列処理数（-1で全コア使用）
        random_state : int
            乱数シード
        verbose : int
            詳細出力レベル
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.feature_names = None
        self.feature_importances_gini = None
        self.feature_importances_perm = None
        
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
        
        # モデル作成・学習
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        self.model.fit(X_array, y)
        
        # Gini importanceを取得
        self.feature_importances_gini = self.model.feature_importances_
        
        if self.verbose > 0:
            print(f"Model fitted with {self.n_estimators} trees")
            print(f"Top 5 important features (Gini):")
            self._print_top_features(self.feature_importances_gini, top_k=5)
        
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
    
    def compute_permutation_importance(self, X, y, n_repeats: int = 10):
        """
        Permutation importanceの計算
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            ターゲット
        n_repeats : int
            permutationの繰り返し回数
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        perm_result = permutation_importance(
            self.model, 
            X_array, 
            y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.feature_importances_perm = perm_result.importances_mean
        
        if self.verbose > 0:
            print(f"Permutation importance computed with {n_repeats} repeats")
            print(f"Top 5 important features (Permutation):")
            self._print_top_features(self.feature_importances_perm, top_k=5)
        
        return self.feature_importances_perm
    
    def _print_top_features(self, importances, top_k: int = 10):
        """重要特徴量の表示"""
        indices = np.argsort(importances)[::-1][:top_k]
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {self.feature_names[idx]}: {importances[idx]:.6f}")
    
    def get_feature_importances(self, importance_type: str = 'gini') -> pd.DataFrame:
        """
        特徴量重要度の取得
        
        Parameters
        ----------
        importance_type : str
            'gini' or 'permutation'
            
        Returns
        -------
        pd.DataFrame
            特徴量名と重要度のDataFrame
        """
        if importance_type == 'gini':
            importances = self.feature_importances_gini
        elif importance_type == 'permutation':
            if self.feature_importances_perm is None:
                raise ValueError("Permutation importance not computed yet. "
                               "Call compute_permutation_importance() first.")
            importances = self.feature_importances_perm
        else:
            raise ValueError(f"Unknown importance_type: {importance_type}")
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_feature_importances(self, importance_type: str = 'gini',
                                 top_k: int = 20):
        """
        特徴量重要度のプロット
        
        Parameters
        ----------
        importance_type : str
            'gini' or 'permutation'
        top_k : int
            表示する上位K個
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        df_importance = self.get_feature_importances(importance_type)
        df_top = df_importance.head(top_k)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        
        ax.barh(df_top['feature'][::-1], df_top['importance'][::-1], 
               color='steelblue', alpha=0.7)
        
        title = f"Feature Importance ({importance_type.capitalize()})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def compute_partial_dependence(self, X, features: List[int] = None):
        """
        部分依存の計算
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        features : list of int, optional
            計算する特徴量のインデックス（Noneで全て）
            
        Returns
        -------
        dict
            {feature_idx: (grid_values, pd_values)}
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if features is None:
            features = list(range(X_array.shape[1]))
        
        pd_results = {}
        
        for feat_idx in features:
            pd_result = partial_dependence(
                self.model, X_array, features=[feat_idx],
                grid_resolution=100
            )
            
            pd_values = pd_result['average'][0]
            grid_values = pd_result['grid_values'][0]
            
            pd_results[feat_idx] = (grid_values, pd_values)
        
        return pd_results
    
    def plot_partial_dependence(self, X, feature_indices: List[int] = None,
                               feature_names: List[str] = None):
        """
        部分依存プロット
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        feature_indices : list of int, optional
            プロットする特徴量のインデックス
        feature_names : list of str, optional
            特徴量名（指定しない場合は self.feature_names を使用）
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        # 部分依存を計算
        pd_results = self.compute_partial_dependence(X, feature_indices)
        
        n_features = len(pd_results)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(6 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()
        
        for i, (feat_idx, (grid_vals, pd_vals)) in enumerate(pd_results.items()):
            ax = axes[i]
            
            ax.plot(grid_vals, pd_vals, linewidth=2, color='steelblue')
            ax.fill_between(grid_vals, pd_vals.min(), pd_vals, 
                           alpha=0.3, color='steelblue')
            
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature {feat_idx}'
            ax.set_title(f'{feat_name}の部分依存', fontsize=12, fontweight='bold')
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel('Partial Dependence', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # 余分な軸を非表示
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def get_top_features(self, top_k: int = 10, 
                        importance_type: str = 'gini') -> List[str]:
        """
        重要度上位K個の特徴量名を取得
        
        Parameters
        ----------
        top_k : int
            上位K個
        importance_type : str
            'gini' or 'permutation'
            
        Returns
        -------
        list of str
            特徴量名リスト
        """
        df_importance = self.get_feature_importances(importance_type)
        return df_importance['feature'].head(top_k).tolist()
    
    def print_summary(self):
        """モデルサマリの表示"""
        print("=" * 60)
        print("FAM (Forest Additive Model) Summary")
        print("=" * 60)
        print(f"N estimators: {self.n_estimators}")
        print(f"Max depth: {self.max_depth}")
        print(f"Min samples leaf: {self.min_samples_leaf}")
        print(f"N features: {len(self.feature_names)}")
        print("\nTop 10 Important Features (Gini):")
        self._print_top_features(self.feature_importances_gini, top_k=10)
        
        if self.feature_importances_perm is not None:
            print("\nTop 10 Important Features (Permutation):")
            self._print_top_features(self.feature_importances_perm, top_k=10)
        
        print("=" * 60)


if __name__ == '__main__':
    # テスト
    print("FAM Model implementation loaded successfully!")
    
    # ダミーデータでテスト
    np.random.seed(42)
    n_samples = 5000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # 一部の特徴量だけが重要
    y = (2 * X[:, 0]**2 + 1.5 * X[:, 1] + 0.5 * X[:, 2] + 
         np.random.randn(n_samples) * 0.5)
    
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    
    # モデル学習
    model = FAMModel(n_estimators=100, verbose=1)
    model.fit(X_df, y)
    
    # 予測
    y_pred = model.predict(X_df)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    print(f"\nTest RMSE: {rmse:.4f}")
    
    # サマリ表示
    model.print_summary()
