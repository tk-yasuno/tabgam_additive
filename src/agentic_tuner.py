"""
Agentic Ensemble Weight Tuning System for v0.5

このモジュールは、以下の機能を提供します：
1. Diverse Ensemble: GAM + FAM + GBAM + TabNAM の異種アンサンブル
2. Agentic AI: モデル個別ハイパーパラメータ + アンサンブル重みの自動探索
3. Planner-Executor-Evaluator パターンによる反復最適化

設計思想:
- 各モデルを「十分に良い」状態まで個別チューニング
- その上に weighted averaging を構築
- Agentic AI で「モデルHP + 重み」を共同最適化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize, differential_evolution
import json
import time
from datetime import datetime


class AgenticTuner:
    """
    Agentic AI による Ensemble Weight Tuning システム
    
    機能:
    - モデルごとのハイパーパラメータ探索空間の定義
    - アンサンブル重みの最適化
    - Planner-Executor-Evaluator ループ
    - 探索履歴の記録と分析
    """
    
    def __init__(
        self,
        models_dict: Dict[str, Any],
        X_train,
        y_train: np.ndarray,
        X_val,
        y_val: np.ndarray,
        optimization_metric: str = 'r2',
        max_iterations: int = 50,
        patience: int = 10,
        random_state: int = 42
    ):
        """
        Parameters
        ----------
        models_dict : dict
            モデル名とモデルインスタンスの辞書
            例: {'GAM': glm_gam_model, 'FAM': fam_model, ...}
        X_train, y_train : DataFrame or ndarray
            学習データ（使用しない、参照のみ）
        X_val, y_val : DataFrame or ndarray
            検証データ（重み最適化用）
        optimization_metric : str
            最適化する指標 ('r2', 'rmse', 'mae')
        max_iterations : int
            最大イテレーション数
        patience : int
            改善が見られない場合の打ち切りイテレーション数
        random_state : int
            乱数シード
        """
        self.models_dict = models_dict
        self.model_names = list(models_dict.keys())
        self.n_models = len(self.model_names)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.optimization_metric = optimization_metric
        self.max_iterations = max_iterations
        self.patience = patience
        self.random_state = random_state
        
        # 探索履歴
        self.search_history = []
        
        # 最良の結果
        self.best_weights = None
        self.best_score = None
        self.best_predictions = None
        
        # 各モデルの予測値（キャッシュ）
        self.val_predictions = {}
        
        print(f"AgenticTuner initialized with {self.n_models} models: {self.model_names}")
    
    def get_model_predictions(self, model_name: str) -> np.ndarray:
        """
        指定モデルの検証データに対する予測値を取得（キャッシュ利用）
        
        Parameters
        ----------
        model_name : str
            モデル名
        
        Returns
        -------
        predictions : ndarray
            予測値
        """
        if model_name not in self.val_predictions:
            model = self.models_dict[model_name]
            self.val_predictions[model_name] = model.predict(self.X_val)
        
        return self.val_predictions[model_name]
    
    def compute_ensemble_prediction(self, weights: np.ndarray) -> np.ndarray:
        """
        アンサンブル予測を計算
        
        Parameters
        ----------
        weights : ndarray
            各モデルの重み（合計1に正規化される）
        
        Returns
        -------
        ensemble_pred : ndarray
            アンサンブル予測値
        """
        # 重みを正規化
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 加重平均
        ensemble_pred = np.zeros_like(self.y_val, dtype=float)
        for i, model_name in enumerate(self.model_names):
            pred = self.get_model_predictions(model_name)
            ensemble_pred += weights[i] * pred
        
        return ensemble_pred
    
    def evaluate_weights(self, weights: np.ndarray) -> float:
        """
        重みに対する評価指標を計算
        
        Parameters
        ----------
        weights : ndarray
            各モデルの重み
        
        Returns
        -------
        score : float
            評価スコア（最小化する場合は負値）
        """
        ensemble_pred = self.compute_ensemble_prediction(weights)
        
        if self.optimization_metric == 'r2':
            # R²は最大化したいので、負値を返す（minimize用）
            score = -r2_score(self.y_val, ensemble_pred)
        elif self.optimization_metric == 'rmse':
            score = np.sqrt(mean_squared_error(self.y_val, ensemble_pred))
        elif self.optimization_metric == 'mae':
            score = mean_absolute_error(self.y_val, ensemble_pred)
        else:
            raise ValueError(f"Unknown metric: {self.optimization_metric}")
        
        return score
    
    def optimize_weights_scipy(self, method: str = 'SLSQP') -> Tuple[np.ndarray, float]:
        """
        SciPyの最適化を使って重みを最適化
        
        Parameters
        ----------
        method : str
            最適化手法 ('SLSQP', 'trust-constr', 'L-BFGS-B')
        
        Returns
        -------
        best_weights : ndarray
            最適な重み
        best_score : float
            最良スコア
        """
        # 初期値: 均等重み
        x0 = np.ones(self.n_models) / self.n_models
        
        # 制約: 重みの合計が1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # 境界: 0 <= w_i <= 1
        bounds = [(0.0, 1.0) for _ in range(self.n_models)]
        
        # 最適化
        result = minimize(
            self.evaluate_weights,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        best_weights = result.x
        best_score = result.fun
        
        # スコアを元に戻す（R²の場合）
        if self.optimization_metric == 'r2':
            best_score = -best_score
        
        return best_weights, best_score
    
    def optimize_weights_differential_evolution(self) -> Tuple[np.ndarray, float]:
        """
        Differential Evolution を使って重みを最適化
        （大域的最適化、局所最適を避ける）
        
        Returns
        -------
        best_weights : ndarray
            最適な重み
        best_score : float
            最良スコア
        """
        # 境界: 0 <= w_i <= 1
        bounds = [(0.0, 1.0) for _ in range(self.n_models)]
        
        def objective_with_normalization(w):
            """重みを正規化してから評価"""
            w_norm = w / w.sum()
            return self.evaluate_weights(w_norm)
        
        # 最適化
        result = differential_evolution(
            objective_with_normalization,
            bounds,
            seed=self.random_state,
            maxiter=500,
            atol=1e-6,
            tol=1e-6,
            workers=1
        )
        
        best_weights = result.x
        best_weights = best_weights / best_weights.sum()  # 正規化
        best_score = result.fun
        
        # スコアを元に戻す（R²の場合）
        if self.optimization_metric == 'r2':
            best_score = -best_score
        
        return best_weights, best_score
    
    def plan_next_configuration(
        self,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Planner: 次の探索候補を計画
        
        現在は簡易版: 複数の最適化手法を試す
        将来的にLLMエージェントに置き換え可能
        
        Parameters
        ----------
        iteration : int
            現在のイテレーション番号
        
        Returns
        -------
        config : dict
            次の探索設定
        """
        if iteration == 0:
            # 初回: 均等重み
            return {
                'method': 'uniform',
                'weights': np.ones(self.n_models) / self.n_models,
                'reason': 'Baseline: uniform weights'
            }
        elif iteration == 1:
            # 2回目: SLSQP最適化
            return {
                'method': 'scipy_SLSQP',
                'reason': 'Local optimization with SLSQP'
            }
        elif iteration == 2:
            # 3回目: Differential Evolution（大域最適化）
            return {
                'method': 'differential_evolution',
                'reason': 'Global optimization with Differential Evolution'
            }
        elif iteration == 3:
            # 4回目: 個別モデルのベストを確認
            return {
                'method': 'individual_best',
                'reason': 'Test individual model performance'
            }
        else:
            # 5回目以降: ランダム探索 + 微調整
            return {
                'method': 'random_perturbation',
                'reason': f'Random perturbation around best weights (iter {iteration})'
            }
    
    def execute_configuration(
        self,
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        Executor: 設定に基づいて評価を実行
        
        Parameters
        ----------
        config : dict
            探索設定
        
        Returns
        -------
        weights : ndarray
            使用した重み
        score : float
            評価スコア
        metrics : dict
            詳細な評価指標
        """
        method = config['method']
        
        if method == 'uniform':
            weights = config['weights']
            score = self.evaluate_weights(weights)
            if self.optimization_metric == 'r2':
                score = -score  # 元に戻す
        
        elif method == 'scipy_SLSQP':
            weights, score = self.optimize_weights_scipy(method='SLSQP')
        
        elif method == 'differential_evolution':
            weights, score = self.optimize_weights_differential_evolution()
        
        elif method == 'individual_best':
            # 各モデル単体での性能を確認
            best_idx = 0
            best_individual_score = float('-inf') if self.optimization_metric == 'r2' else float('inf')
            
            for i in range(self.n_models):
                w = np.zeros(self.n_models)
                w[i] = 1.0
                s = self.evaluate_weights(w)
                if self.optimization_metric == 'r2':
                    s = -s
                
                if self.optimization_metric == 'r2':
                    if s > best_individual_score:
                        best_individual_score = s
                        best_idx = i
                else:
                    if s < best_individual_score:
                        best_individual_score = s
                        best_idx = i
            
            weights = np.zeros(self.n_models)
            weights[best_idx] = 1.0
            score = best_individual_score
        
        elif method == 'random_perturbation':
            # 現在のベスト重みに小さな摂動を加える
            if self.best_weights is not None:
                perturbation = np.random.randn(self.n_models) * 0.05
                weights = self.best_weights + perturbation
                weights = np.maximum(weights, 0)  # 非負制約
                weights = weights / weights.sum()  # 正規化
            else:
                weights = np.random.dirichlet(np.ones(self.n_models))
            
            score = self.evaluate_weights(weights)
            if self.optimization_metric == 'r2':
                score = -score
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 詳細な評価指標を計算
        ensemble_pred = self.compute_ensemble_prediction(weights)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_val, ensemble_pred)),
            'mae': mean_absolute_error(self.y_val, ensemble_pred),
            'r2': r2_score(self.y_val, ensemble_pred)
        }
        
        return weights, score, metrics
    
    def evaluate_iteration(
        self,
        iteration: int,
        weights: np.ndarray,
        score: float,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> bool:
        """
        Evaluator: イテレーションの結果を評価
        
        Parameters
        ----------
        iteration : int
            イテレーション番号
        weights : ndarray
            試した重み
        score : float
            評価スコア
        metrics : dict
            詳細評価指標
        config : dict
            使用した設定
        
        Returns
        -------
        is_best : bool
            最良結果を更新したか
        """
        is_best = False
        
        # 最良結果の更新判定
        if self.best_score is None:
            is_best = True
        elif self.optimization_metric == 'r2':
            if score > self.best_score:
                is_best = True
        else:
            if score < self.best_score:
                is_best = True
        
        # 最良結果を更新
        if is_best:
            self.best_score = score
            self.best_weights = weights.copy()
            self.best_predictions = self.compute_ensemble_prediction(weights)
        
        # 履歴に記録
        record = {
            'iteration': iteration,
            'method': config['method'],
            'reason': config.get('reason', ''),
            'weights': weights.tolist(),
            'weights_dict': {name: float(w) for name, w in zip(self.model_names, weights)},
            'score': float(score),
            'metrics': metrics,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        }
        self.search_history.append(record)
        
        # ログ出力
        print(f"\nIteration {iteration}: {config['method']}")
        print(f"  Reason: {config.get('reason', 'N/A')}")
        print(f"  Weights: {', '.join([f'{name}={w:.3f}' for name, w in zip(self.model_names, weights)])}")
        print(f"  R²={metrics['r2']:.6f}, RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
        if is_best:
            print(f"  ✅ NEW BEST! (R²={score:.6f})")
        
        return is_best
    
    def run(self) -> Dict[str, Any]:
        """
        Agentic Tuning のメインループを実行
        
        Returns
        -------
        results : dict
            最終結果
        """
        print("=" * 80)
        print("Agentic Ensemble Weight Tuning - Starting")
        print("=" * 80)
        print(f"Models: {', '.join(self.model_names)}")
        print(f"Optimization Metric: {self.optimization_metric}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Patience: {self.patience}")
        print()
        
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            # Planner: 次の設定を計画
            config = self.plan_next_configuration(iteration)
            
            # Executor: 実行
            weights, score, metrics = self.execute_configuration(config)
            
            # Evaluator: 評価
            is_best = self.evaluate_iteration(
                iteration, weights, score, metrics, config
            )
            
            # 改善カウントの更新
            if is_best:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 早期終了判定
            if no_improvement_count >= self.patience:
                print(f"\nEarly stopping: No improvement for {self.patience} iterations")
                break
        
        # 最終結果のサマリー
        print("\n" + "=" * 80)
        print("Agentic Tuning - Completed")
        print("=" * 80)
        print(f"Total Iterations: {len(self.search_history)}")
        print(f"\nBest Configuration:")
        print(f"  Optimization Metric ({self.optimization_metric}): {self.best_score:.6f}")
        print(f"  Best Weights:")
        for name, weight in zip(self.model_names, self.best_weights):
            print(f"    {name}: {weight:.4f}")
        
        # 詳細メトリクス
        best_pred = self.compute_ensemble_prediction(self.best_weights)
        final_metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_val, best_pred)),
            'mae': mean_absolute_error(self.y_val, best_pred),
            'r2': r2_score(self.y_val, best_pred)
        }
        print(f"\n  Final Metrics:")
        print(f"    RMSE: {final_metrics['rmse']:.6f}")
        print(f"    MAE: {final_metrics['mae']:.6f}")
        print(f"    R²: {final_metrics['r2']:.6f}")
        
        return {
            'best_weights': self.best_weights,
            'best_score': self.best_score,
            'best_predictions': self.best_predictions,
            'best_metrics': final_metrics,
            'search_history': self.search_history,
            'model_names': self.model_names
        }
    
    def save_results(self, output_path: str):
        """
        結果をJSONファイルに保存
        
        Parameters
        ----------
        output_path : str
            出力ファイルパス
        """
        results = {
            'model_names': self.model_names,
            'best_weights': self.best_weights.tolist(),
            'best_score': float(self.best_score),
            'optimization_metric': self.optimization_metric,
            'search_history': self.search_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")


def create_diverse_ensemble_predictions(
    models_dict: Dict[str, Any],
    weights: np.ndarray,
    X
) -> np.ndarray:
    """
    Diverse Ensemble の予測を作成
    
    Parameters
    ----------
    models_dict : dict
        モデル辞書
    weights : ndarray
        各モデルの重み
    X : DataFrame or ndarray
        入力データ
    
    Returns
    -------
    predictions : ndarray
        アンサンブル予測
    """
    model_names = list(models_dict.keys())
    weights = np.array(weights) / np.sum(weights)  # 正規化
    
    # X のサンプル数を取得（DataFrame or ndarray対応）
    n_samples = len(X)
    
    predictions = np.zeros(n_samples, dtype=float)
    for i, name in enumerate(model_names):
        model = models_dict[name]
        pred = model.predict(X)
        predictions += weights[i] * pred
    
    return predictions
