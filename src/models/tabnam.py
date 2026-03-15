"""
TabNAM (TabNet Additive Models)
TabNetの注意機構を使用した加法的モデル

- TabNetの注意係数から特徴量の重要度を計算
- スパース性と解釈可能性を両立
- Attention機構による特徴選択
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# TabNetのインポート（利用可能な場合）
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    warnings.warn("pytorch_tabnet is not installed. TabNAM will use a simplified implementation.")


class TabNAM:
    """
    TabNet Additive Model
    
    h_i ≈ φ_0 + Σ φ_j(x_ij)
    TabNetの注意機構を活用した加法的モデル
    """
    
    def __init__(self,
                 n_d: int = 8,
                 n_a: int = 8,
                 n_steps: int = 3,
                 gamma: float = 1.3,
                 n_independent: int = 2,
                 n_shared: int = 2,
                 lambda_sparse: float = 1e-3,
                 optimizer_fn = None,
                 optimizer_params: dict = None,
                 mask_type: str = 'sparsemax',
                 scheduler_params: dict = None,
                 scheduler_fn = None,
                 epsilon: float = 1e-15,
                 momentum: float = 0.02,
                 seed: int = 42,
                 verbose: int = 0,
                 device_name: str = 'cpu'):
        """
        Parameters
        ----------
        n_d : int
            決定予測層の幅
        n_a : int
            注意埋め込み層の幅
        n_steps : int
            意思決定ステップ数
        gamma : float
            スパース性の度合いを制御する係数
        n_independent : int
            各GLU blockの独立したGLUレイヤー数
        n_shared : int
            各GLU blockの共有GLUレイヤー数
        lambda_sparse : float
            スパース正則化の強度
        optimizer_fn : torch.optim
            オプティマイザー
        optimizer_params : dict
            オプティマイザーのパラメータ
        mask_type : str
            'sparsemax' or 'entmax'
        scheduler_params : dict
            スケジューラーのパラメータ
        scheduler_fn : torch.optim.lr_scheduler
            学習率スケジューラー
        epsilon : float
            数値安定性のための小さな値
        momentum : float
            バッチ正規化のモメンタム
        seed : int
            乱数シード
        verbose : int
            詳細出力レベル
        device_name : str
            'cpu' or 'cuda' or 'mps'
        """
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params or {'lr': 2e-2}
        self.mask_type = mask_type
        self.scheduler_params = scheduler_params
        self.scheduler_fn = scheduler_fn
        self.epsilon = epsilon
        self.momentum = momentum
        self.seed = seed
        self.verbose = verbose
        self.device_name = device_name
        
        self.model = None
        self.feature_names = None
        self.n_features_ = None
        self.feature_importances_ = None
        
        if not TABNET_AVAILABLE:
            # TabNetが利用できない場合の簡易実装
            from sklearn.ensemble import GradientBoostingRegressor
            self.fallback_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=seed
            )
            self.use_fallback = True
            if verbose > 0:
                print("Warning: Using GradientBoostingRegressor as fallback for TabNet")
        else:
            self.use_fallback = False
    
    def fit(self, X, y, eval_set=None, max_epochs=100, patience=15, 
            batch_size=1024, virtual_batch_size=128):
        """
        モデル学習
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            ターゲット
        eval_set : list of tuple, optional
            [(X_val, y_val)] 検証データ
        max_epochs : int
            最大エポック数
        patience : int
            Early stoppingの patience
        batch_size : int
            バッチサイズ
        virtual_batch_size : int
            仮想バッチサイズ
        """
        # DataFrameの場合は特徴量名を保存
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
            self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        
        self.n_features_ = X_array.shape[1]
        
        if self.use_fallback:
            # Fallback実装
            self.fallback_model.fit(X_array, y)
            self.feature_importances_ = self.fallback_model.feature_importances_
        else:
            # TabNet実装
            import torch
            if self.optimizer_fn is None:
                from torch.optim import Adam
                optimizer_fn = Adam
            else:
                optimizer_fn = self.optimizer_fn
            
            self.model = TabNetRegressor(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                n_independent=self.n_independent,
                n_shared=self.n_shared,
                lambda_sparse=self.lambda_sparse,
                optimizer_fn=optimizer_fn,
                optimizer_params=self.optimizer_params,
                mask_type=self.mask_type,
                scheduler_params=self.scheduler_params,
                scheduler_fn=self.scheduler_fn,
                epsilon=self.epsilon,
                momentum=self.momentum,
                seed=self.seed,
                verbose=self.verbose,
                device_name=self.device_name
            )
            
            # 学習
            self.model.fit(
                X_train=X_array,
                y_train=y.reshape(-1, 1),
                eval_set=eval_set,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size
            )
            
            # 特徴量重要度を取得
            self.feature_importances_ = self.model.feature_importances_
        
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
        
        if self.use_fallback:
            return self.fallback_model.predict(X_array)
        else:
            predictions = self.model.predict(X_array)
            return predictions.flatten()
    
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
    
    def explain(self, X):
        """
        各サンプルの特徴量重要度を取得（TabNetのマスク値）
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
            
        Returns
        -------
        np.ndarray
            サンプル × 特徴量 の重要度マトリクス
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if self.use_fallback:
            # Fallback: 全サンプルで同じ重要度を返す
            return np.tile(self.feature_importances_, (X_array.shape[0], 1))
        else:
            # TabNetの説明を取得
            explain_matrix, masks = self.model.explain(X_array)
            return explain_matrix
    
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
        plt.barh(range(len(importances)), importances.values)
        plt.yticks(range(len(importances)), importances.index)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('TabNet Feature Importances', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature importances plot saved to {save_path}")
        
        plt.show()


class TabNAMModel:
    """
    TabNAMモデルのラッパークラス（他のモデルとのインターフェース統一）
    """
    def __init__(self, n_d: int = 8, n_a: int = 8, n_steps: int = 3,
                 lambda_sparse: float = 1e-3, seed: int = 42, 
                 verbose: int = 0, device_name: str = 'cpu'):
        self.model = TabNAM(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            lambda_sparse=lambda_sparse,
            seed=seed,
            verbose=verbose,
            device_name=device_name
        )
        self.max_epochs = 100
        self.patience = 15
        self.batch_size = 1024
    
    def fit(self, X, y):
        self.model.fit(
            X, y, 
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)
