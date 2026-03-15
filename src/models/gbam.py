"""
GBAM (Gradient Boosting Additive Models)
LightGBMを使用した高性能加法的モデル

- 通常回帰（平均ハザード）
- 分位回帰（高リスク帯域）
- SHAP値による加法的寄与の解釈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class GBAMModel:
    """
    Gradient Boosting Additive Model using LightGBM
    
    h_i ≈ φ_0 + Σ φ_j(x_ij)  (SHAP値による加法的分解)
    """
    
    def __init__(self,
                 objective: str = 'regression',
                 quantile_alpha: float = None,
                 num_leaves: int = 31,
                 learning_rate: float = 0.05,
                 n_estimators: int = 800,
                 feature_fraction: float = 0.8,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 1,
                 min_child_samples: int = 20,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 0.0,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 verbose: int = -1):
        """
        Parameters
        ----------
        objective : str
            'regression' or 'quantile'
        quantile_alpha : float, optional
            quantile objective使用時の分位点（0 < alpha < 1）
        num_leaves : int
            葉ノード数
        learning_rate : float
            学習率
        n_estimators : int
            木の数
        feature_fraction : float
            各木で使用する特徴量の割合
        bagging_fraction : float
            各木で使用するサンプルの割合
        bagging_freq : int
            baggingの頻度
        min_child_samples : int
            葉ノードの最小サンプル数
        reg_alpha : float
            L1正則化
        reg_lambda : float
            L2正則化
        n_jobs : int
            並列処理数（-1で全コア使用）
        random_state : int
            乱数シード
        verbose : int
            詳細出力レベル（-1で非表示）
        """
        self.objective = objective
        self.quantile_alpha = quantile_alpha
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.feature_names = None
        self.shap_explainer = None
        self.shap_values = None
        
    def _get_params(self) -> Dict:
        """パラメータ辞書の取得"""
        params = {
            'objective': self.objective,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'min_child_samples': self.min_child_samples,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }
        
        if self.objective == 'regression':
            params['metric'] = 'rmse'
        elif self.objective == 'quantile':
            if self.quantile_alpha is None:
                raise ValueError("quantile_alpha must be specified for quantile objective")
            params['alpha'] = self.quantile_alpha
            params['metric'] = 'quantile'
        
        return params
    
    def fit(self, X, y, eval_set: Tuple = None, 
            early_stopping_rounds: int = 50):
        """
        モデル学習
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        y : np.ndarray
            ターゲット
        eval_set : tuple, optional
            (X_val, y_val) の検証データ
        early_stopping_rounds : int
            Early stoppingのラウンド数
        """
        # DataFrameの場合は特徴量名を保存
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
            self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        
        # LightGBM Dataset作成
        train_data = lgb.Dataset(X_array, label=y, feature_name=self.feature_names)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            valid_data = lgb.Dataset(X_val, label=y_val, 
                                    feature_name=self.feature_names,
                                    reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # パラメータ取得
        params = self._get_params()
        
        # 学習
        callbacks = []
        if early_stopping_rounds > 0 and eval_set is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks if callbacks else None
        )
        
        if self.verbose >= 0:
            print(f"Best iteration: {self.model.best_iteration}")
            print(f"Best score: {self.model.best_score}")
        
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
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        特徴量重要度の取得
        
        Parameters
        ----------
        importance_type : str
            'gain' or 'split'
            
        Returns
        -------
        pd.DataFrame
            特徴量名と重要度のDataFrame
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_feature_importance(self, importance_type: str = 'gain',
                               top_k: int = 20):
        """
        特徴量重要度のプロット
        
        Parameters
        ----------
        importance_type : str
            'gain' or 'split'
        top_k : int
            表示する上位K個
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        df_importance = self.get_feature_importance(importance_type)
        df_top = df_importance.head(top_k)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        
        ax.barh(df_top['feature'][::-1], df_top['importance'][::-1],
               color='lightcoral', alpha=0.7)
        
        title = f"Feature Importance ({importance_type.capitalize()})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def compute_shap_values(self, X, check_additivity: bool = False):
        """
        SHAP値の計算
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        check_additivity : bool
            加法性のチェックを行うか
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # SHAP explainerの作成
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        # SHAP値の計算
        self.shap_values = self.shap_explainer.shap_values(
            X_array,
            check_additivity=check_additivity
        )
        
        return self.shap_values
    
    def plot_shap_summary(self, X, plot_type: str = 'dot', 
                         max_display: int = 20):
        """
        SHAP summary plotの作成
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        plot_type : str
            'dot' or 'bar'
        max_display : int
            表示する特徴量数
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        fig = plt.figure(figsize=(10, max(6, max_display * 0.3)))
        
        shap.summary_plot(
            self.shap_values,
            X_array,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        return fig
    
    def plot_shap_dependence(self, X, feature_name: str,
                            interaction_feature: str = None):
        """
        SHAP依存性プロット（特徴量の加法的寄与）
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        feature_name : str
            プロットする特徴量名
        interaction_feature : str, optional
            交互作用を表示する特徴量名
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names)
        
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found")
        
        fig = plt.figure(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            X_df,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.tight_layout()
        return fig
    
    def get_shap_feature_importance(self) -> pd.DataFrame:
        """
        SHAP値に基づく特徴量重要度
        
        Returns
        -------
        pd.DataFrame
            特徴量名と平均絶対SHAP値のDataFrame
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. "
                           "Call compute_shap_values() first.")
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_shap_waterfall(self, X, sample_idx: int = 0):
        """
        SHAP waterfall plot（1サンプルの予測説明）
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            特徴量
        sample_idx : int
            プロットするサンプルのインデックス
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        fig = plt.figure(figsize=(10, 8))
        
        shap_explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.shap_explainer.expected_value,
            data=X_array[sample_idx],
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(shap_explanation, show=False)
        
        plt.tight_layout()
        return fig
    
    def print_summary(self):
        """モデルサマリの表示"""
        print("=" * 60)
        print("GBAM (Gradient Boosting Additive Model) Summary")
        print("=" * 60)
        print(f"Objective: {self.objective}")
        if self.objective == 'quantile':
            print(f"Quantile alpha: {self.quantile_alpha}")
        print(f"N estimators: {self.n_estimators}")
        print(f"Best iteration: {self.model.best_iteration}")
        print(f"Num leaves: {self.num_leaves}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"N features: {len(self.feature_names)}")
        
        print("\nTop 10 Important Features (Gain):")
        df_importance = self.get_feature_importance('gain')
        for i, row in df_importance.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.2f}")
        
        print("=" * 60)


if __name__ == '__main__':
    # テスト
    print("GBAM Model implementation loaded successfully!")
    
    # ダミーデータでテスト
    np.random.seed(42)
    n_samples = 5000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # 一部の特徴量が重要
    y = (2 * X[:, 0]**2 + 1.5 * X[:, 1] + 0.5 * X[:, 2] +
         np.random.randn(n_samples) * 0.5)
    
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    
    # 訓練・検証分割
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_df[:split_idx], X_df[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # モデル学習（回帰）
    model = GBAMModel(objective='regression', n_estimators=100, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # 予測
    y_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    print(f"\nTest RMSE: {rmse:.4f}")
    
    # サマリ表示
    model.print_summary()
