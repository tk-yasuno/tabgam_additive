"""
NAM (Neural Additive Models)
ニューラル加法モデル

PyTorchを使用した実装
h_i = Σ f_j^NN(x_ij) + Σ g_k^NN(z_ik)
各特徴量ごとに独立した小さなMLPを持つ
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureNN(nn.Module):
    """各特徴量用の小さなMLP"""
    
    def __init__(self, input_dim: int = 1, 
                 hidden_dims: List[int] = [32, 16],
                 dropout: float = 0.1):
        """
        Parameters
        ----------
        input_dim : int
            入力次元（通常は1）
        hidden_dims : list of int
            隠れ層のユニット数
        dropout : float
            ドロップアウト率
        """
        super(FeatureNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1)
        
        Returns
        -------
        torch.Tensor, shape (batch, 1)
        """
        return self.net(x)


class NAMModel(nn.Module):
    """Neural Additive Model"""
    
    def __init__(self, n_features: int,
                 hidden_dims: List[int] = [32, 16],
                 dropout: float = 0.1,
                 use_bias: bool = True):
        """
        Parameters
        ----------
        n_features : int
            特徴量の数
        hidden_dims : list of int
            各FeatureNNの隠れ層ユニット数
        dropout : float
            ドロップアウト率
        use_bias : bool
            バイアス項を使うか
        """
        super(NAMModel, self).__init__()
        
        self.n_features = n_features
        self.use_bias = use_bias
        
        # 各特徴量用のNN
        self.feature_nns = nn.ModuleList([
            FeatureNN(input_dim=1, hidden_dims=hidden_dims, dropout=dropout)
            for _ in range(n_features)
        ])
        
        # バイアス項
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, X):
        """
        Parameters
        ----------
        X : torch.Tensor, shape (batch, n_features)
        
        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        # 各特徴量の寄与を計算
        contributions = []
        for i, feature_nn in enumerate(self.feature_nns):
            x_i = X[:, i:i+1]  # (batch, 1)
            contrib = feature_nn(x_i)  # (batch, 1)
            contributions.append(contrib)
        
        # 加法的に合計
        output = torch.cat(contributions, dim=1).sum(dim=1)  # (batch,)
        
        if self.use_bias:
            output = output + self.bias
        
        return output
    
    def get_feature_contribution(self, X, feature_idx: int):
        """
        特定特徴量の寄与を取得
        
        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            特徴量
        feature_idx : int
            特徴量のインデックス
            
        Returns
        -------
        np.ndarray
            寄与度
        """
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        self.eval()
        with torch.no_grad():
            x_i = X[:, feature_idx:feature_idx+1]
            contrib = self.feature_nns[feature_idx](x_i)
        
        return contrib.squeeze().numpy()


class NAMTrainer:
    """NAMモデルの学習用クラス"""
    
    def __init__(self, n_features: int,
                 hidden_dims: List[int] = [16, 16],  # より小さいネットワーク
                 dropout: float = 0.3,                # より強い正則化
                 learning_rate: float = 1e-4,         # より小さい学習率
                 batch_size: int = 2048,              # より大きいバッチ
                 epochs: int = 50,
                 early_stopping_patience: int = 10,
                 device: str = None,
                 verbose: bool = True):
        """
        Parameters
        ----------
        n_features : int
            特徴量の数
        hidden_dims : list of int
            隠れ層ユニット数
        dropout : float
            ドロップアウト率
        learning_rate : float
            学習率
        batch_size : int
            バッチサイズ
        epochs : int
            エポック数
        early_stopping_patience : int
            Early stoppingの patience
        device : str, optional
            'cuda', 'cpu', or None (自動選択)
        verbose : bool
            詳細出力
        """
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        
        # デバイス設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if self.verbose:
            print(f"Using device: {self.device}")
        
        # モデル初期化
        self.model = NAMModel(
            n_features=n_features,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        self.best_model_state = None
        
    def _prepare_data(self, X, y):
        """データをTensorに変換"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        return X_tensor, y_tensor
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        モデル学習
        
        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame
            訓練データ特徴量
        y_train : np.ndarray or pd.Series
            訓練データターゲット
        X_val : np.ndarray or pd.DataFrame, optional
            検証データ特徴量
        y_val : np.ndarray or pd.Series, optional
            検証データターゲット
        """
        # データ準備
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
            X_val_tensor = X_val_tensor.to(self.device)
            y_val_tensor = y_val_tensor.to(self.device)
            use_validation = True
        else:
            use_validation = False
        
        # Early stopping用
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 学習ループ
        for epoch in range(self.epochs):
            # 訓練
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient Clipping（勾配爆発防止）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # 検証
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    self.history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}")
        
        # ベストモデルをロード
        if use_validation and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self
    
    def predict(self, X):
        """
        予測
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            特徴量
            
        Returns
        -------
        np.ndarray
            予測値
        """
        X_tensor, _ = self._prepare_data(X, np.zeros(len(X)))
        X_tensor = X_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def plot_training_history(self):
        """学習履歴のプロット"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            ax.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('NAM Training History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_contributions(self, X, feature_names: List[str] = None,
                                   n_samples: int = 1000):
        """
        特徴量の寄与曲線をプロット
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            特徴量
        feature_names : list of str, optional
            特徴量名
        n_samples : int
            プロット用のサンプル数
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(X_array.shape[1])]
        
        # サンプリング
        if len(X_array) > n_samples:
            indices = np.random.choice(len(X_array), n_samples, replace=False)
            X_sample = X_array[indices]
        else:
            X_sample = X_array
        
        n_features = X_sample.shape[1]
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()
        
        for i in range(n_features):
            ax = axes[i]
            
            # 各特徴量の値と寄与を取得
            x_values = X_sample[:, i]
            contributions = self.model.get_feature_contribution(X_sample, i)
            
            # ソート
            sort_idx = np.argsort(x_values)
            x_sorted = x_values[sort_idx]
            contrib_sorted = contributions[sort_idx]
            
            # プロット
            ax.scatter(x_sorted, contrib_sorted, alpha=0.3, s=10)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            feat_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'
            ax.set_title(f'{feat_name}の寄与', fontsize=12, fontweight='bold')
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel('Contribution', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # 余分な軸を非表示
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


if __name__ == '__main__':
    # テスト
    print("NAM Model implementation loaded successfully!")
    
    # ダミーデータでテスト
    np.random.seed(42)
    n_samples = 5000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0]**2 + np.sin(X[:, 1]) + 0.5 * X[:, 2] + 
         np.random.randn(n_samples) * 0.5)
    
    # 訓練・検証分割
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # モデル学習
    trainer = NAMTrainer(
        n_features=n_features,
        hidden_dims=[32, 16],
        epochs=30,
        verbose=True
    )
    trainer.fit(X_train, y_train, X_val, y_val)
    
    # 予測
    y_pred = trainer.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    print(f"\nTest RMSE: {rmse:.4f}")
