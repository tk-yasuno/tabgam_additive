"""
航空エンジン劣化データの前処理 - v0.7
CMAPSS FD004 Data Preprocessing for Multi-Mechanism Degradation Analysis

目的:
- 複合劣化（摩耗 × 熱疲労 × 振動 × 運転条件）の相互作用を分析
- v0.5のアルゴリズムと同じ構造を維持
- RUL（Remaining Useful Life）予測

データセット:
- FD004: 6 operational conditions × 2 fault modes (HPC + Fan degradation)
- Train: 249 engines, 61,249 samples
- Test: 248 engines, 41,214 samples
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class CMAPSSPreprocessor:
    """CMAPSS FD004データの前処理クラス - v0.7"""
    
    def __init__(self, data_dir='data/CMAPSSData', dataset='FD004'):
        """
        Parameters
        ----------
        data_dir : str
            データディレクトリのパス
        dataset : str
            データセット名（FD001, FD002, FD003, FD004）
        """
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.train_df = None
        self.test_df = None
        self.rul_df = None
        self.scaler = StandardScaler()
        
        # カラム名
        self.columns = ['unit', 'cycle'] + \
                      [f'op_setting_{i}' for i in range(1, 4)] + \
                      [f'sensor_{i}' for i in range(1, 22)]
        
        # 運転条件の特徴量
        self.operational_features = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        
        # センサー特徴量
        self.sensor_features = [f'sensor_{i}' for i in range(1, 22)]
        
        # 除去するセンサー（定数または低変動）
        self.sensors_to_drop = []
        
    def load_data(self):
        """データ読み込み"""
        print(f"Loading CMAPSS {self.dataset} data...")
        
        # 訓練データ
        self.train_df = pd.read_csv(
            self.data_dir / f'train_{self.dataset}.txt',
            sep=r'\s+',
            header=None,
            names=self.columns
        )
        
        # テストデータ
        self.test_df = pd.read_csv(
            self.data_dir / f'test_{self.dataset}.txt',
            sep=r'\s+',
            header=None,
            names=self.columns
        )
        
        # RUL
        self.rul_df = pd.read_csv(
            self.data_dir / f'RUL_{self.dataset}.txt',
            sep=r'\s+',
            header=None,
            names=['RUL']
        )
        
        print(f"  Train: {self.train_df.shape[0]:,} samples, {self.train_df['unit'].nunique()} engines")
        print(f"  Test:  {self.test_df.shape[0]:,} samples, {self.test_df['unit'].nunique()} engines")
        
        return self
    
    def remove_constant_sensors(self, threshold=0.001):
        """定数または低変動のセンサーを除去"""
        print("\nRemoving constant sensors...")
        
        sensors_to_drop = []
        for sensor in self.sensor_features:
            std = self.train_df[sensor].std()
            if std < threshold:
                sensors_to_drop.append(sensor)
                print(f"  Removing {sensor}: std={std:.6f}")
        
        if sensors_to_drop:
            self.train_df = self.train_df.drop(columns=sensors_to_drop)
            self.test_df = self.test_df.drop(columns=sensors_to_drop)
            self.sensor_features = [s for s in self.sensor_features if s not in sensors_to_drop]
            self.sensors_to_drop = sensors_to_drop
        else:
            print("  No constant sensors found")
        
        return self
    
    def create_rul_target(self, max_rul=125):
        """
        RUL（Remaining Useful Life）ターゲット変数の作成
        
        Parameters
        ----------
        max_rul : int
            最大RUL値（一般的に125サイクル）
            早期サイクルの影響を抑えるためキャップする
        """
        print(f"\nCreating RUL target (max_rul={max_rul})...")
        
        # 訓練データのRUL計算
        # 各エンジンの最大サイクル数を取得
        max_cycles = self.train_df.groupby('unit')['cycle'].max().to_dict()
        
        # RUL = max_cycle - current_cycle
        self.train_df['RUL'] = self.train_df.apply(
            lambda row: max_cycles[row['unit']] - row['cycle'],
            axis=1
        )
        
        # Piecewise Linear RUL（max_rulでキャップ）
        self.train_df['RUL_capped'] = self.train_df['RUL'].clip(upper=max_rul)
        
        print(f"  RUL range: [{self.train_df['RUL'].min()}, {self.train_df['RUL'].max()}]")
        print(f"  RUL_capped range: [{self.train_df['RUL_capped'].min()}, {self.train_df['RUL_capped'].max()}]")
        print(f"  Mean RUL: {self.train_df['RUL'].mean():.1f}")
        print(f"  Mean RUL_capped: {self.train_df['RUL_capped'].mean():.1f}")
        
        # テストデータのRUL追加
        # テストデータの最終サイクルに対するRULを設定
        test_final_cycles = self.test_df.groupby('unit')['cycle'].max().reset_index()
        test_final_cycles['RUL'] = self.rul_df['RUL'].values
        test_final_cycles['RUL_capped'] = test_final_cycles['RUL'].clip(upper=max_rul)
        
        # 各サイクルのRULを逆算
        self.test_df = self.test_df.merge(
            test_final_cycles[['unit', 'cycle', 'RUL']].rename(
                columns={'cycle': 'max_cycle', 'RUL': 'final_RUL'}
            ),
            left_on='unit',
            right_on='unit'
        )
        
        self.test_df['RUL'] = self.test_df['final_RUL'] + (
            self.test_df['max_cycle'] - self.test_df['cycle']
        )
        self.test_df['RUL_capped'] = self.test_df['RUL'].clip(upper=max_rul)
        
        # 不要な列を削除
        self.test_df = self.test_df.drop(columns=['max_cycle', 'final_RUL'])
        
        print(f"  Test RUL range: [{self.test_df['RUL'].min()}, {self.test_df['RUL'].max()}]")
        
        return self
    
    def create_rolling_features(self, window_sizes=[5, 10, 20]):
        """
        ローリング統計特徴量の作成
        
        Parameters
        ----------
        window_sizes : list
            ウィンドウサイズのリスト
        """
        print(f"\nCreating rolling features (windows={window_sizes})...")
        
        rolling_features = []
        
        for df in [self.train_df, self.test_df]:
            for window in window_sizes:
                for sensor in self.sensor_features[:5]:  # 主要センサーのみ
                    # ローリング平均
                    col_name = f'{sensor}_roll_mean_{window}'
                    df[col_name] = df.groupby('unit')[sensor].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    rolling_features.append(col_name)
                    
                    # ローリング標準偏差
                    col_name = f'{sensor}_roll_std_{window}'
                    df[col_name] = df.groupby('unit')[sensor].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    ).fillna(0)
                    rolling_features.append(col_name)
        
        # 重複を除去
        rolling_features = list(set(rolling_features))
        
        print(f"  Created {len(rolling_features)} rolling features")
        
        return self
    
    def normalize_features(self):
        """特徴量の正規化"""
        print("\nNormalizing features...")
        
        # 正規化する特徴量
        features_to_normalize = (
            self.operational_features + 
            self.sensor_features +
            [col for col in self.train_df.columns if 'roll_' in col]
        )
        
        # 存在する特徴量のみ選択
        features_to_normalize = [f for f in features_to_normalize if f in self.train_df.columns]
        
        # 訓練データでfitして両方に適用
        self.scaler.fit(self.train_df[features_to_normalize])
        
        self.train_df[features_to_normalize] = self.scaler.transform(
            self.train_df[features_to_normalize]
        )
        self.test_df[features_to_normalize] = self.scaler.transform(
            self.test_df[features_to_normalize]
        )
        
        print(f"  Normalized {len(features_to_normalize)} features")
        
        return self
    
    def get_feature_matrix(self, use_capped_rul=True):
        """
        特徴量行列の取得
        
        Parameters
        ----------
        use_capped_rul : bool
            キャップされたRULを使用するか
        
        Returns
        -------
        X_train, X_test : pd.DataFrame
            特徴量行列
        y_train, y_test : np.array
            ターゲット変数
        feature_names : list
            特徴量名のリスト
        """
        # 特徴量カラム
        feature_cols = (
            self.operational_features + 
            self.sensor_features +
            ['cycle'] +  # サイクル数も重要な特徴量
            [col for col in self.train_df.columns if 'roll_' in col]
        )
        
        # 存在するカラムのみ選択
        feature_cols = [col for col in feature_cols if col in self.train_df.columns]
        
        X_train = self.train_df[feature_cols].copy()
        X_test = self.test_df[feature_cols].copy()
        
        # ターゲット変数
        target_col = 'RUL_capped' if use_capped_rul else 'RUL'
        y_train = self.train_df[target_col].values
        y_test = self.test_df[target_col].values
        
        print(f"\nFeature matrix created:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target: {target_col}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        データを訓練/検証/テストに分割
        
        Note: CMAPSSの場合、engineベースでの分割が重要
        
        Parameters
        ----------
        test_size : float
            検証セットの割合（元の訓練データから分割）
        random_state : int
            乱数シード
        
        Returns
        -------
        splits : dict
            {'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'}
        """
        X_train_full, X_test, y_train_full, y_test, feature_names = self.get_feature_matrix()
        
        # エンジン単位で訓練/検証を分割
        unique_units = self.train_df['unit'].unique()
        np.random.seed(random_state)
        
        val_units = np.random.choice(
            unique_units, 
            size=int(len(unique_units) * test_size),
            replace=False
        )
        
        train_mask = ~self.train_df['unit'].isin(val_units)
        val_mask = self.train_df['unit'].isin(val_units)
        
        X_train = X_train_full[train_mask]
        X_val = X_train_full[val_mask]
        y_train = y_train_full[train_mask]
        y_val = y_train_full[val_mask]
        
        print(f"\nData split (engine-based):")
        print(f"  Train: {X_train.shape[0]:,} samples ({len(unique_units) - len(val_units)} engines)")
        print(f"  Val:   {X_val.shape[0]:,} samples ({len(val_units)} engines)")
        print(f"  Test:  {X_test.shape[0]:,} samples ({self.test_df['unit'].nunique()} engines)")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names
        }


def load_and_preprocess_data_v07(
    data_dir='data/CMAPSSData',
    dataset='FD004',
    max_rul=125,
    rolling_windows=[5, 10, 20],
    test_size=0.2,
    random_state=42
):
    """
    CMAPSS FD004データの読み込みと前処理 - v0.7
    
    Parameters
    ----------
    data_dir : str
        データディレクトリ
    dataset : str
        データセット名
    max_rul : int
        最大RUL値
    rolling_windows : list
        ローリング特徴量のウィンドウサイズ
    test_size : float
        検証セットの割合
    random_state : int
        乱数シード
    
    Returns
    -------
    splits : dict
        データ分割の辞書
    preprocessor : CMAPSSPreprocessor
        前処理器インスタンス
    """
    print("=" * 80)
    print("CMAPSS FD004 DATA PREPROCESSING - v0.7")
    print("Jet Engine Multi-Mechanism Degradation Analysis")
    print("=" * 80)
    
    preprocessor = CMAPSSPreprocessor(data_dir=data_dir, dataset=dataset)
    
    preprocessor.load_data()
    preprocessor.remove_constant_sensors()
    preprocessor.create_rul_target(max_rul=max_rul)
    preprocessor.create_rolling_features(window_sizes=rolling_windows)
    preprocessor.normalize_features()
    
    splits = preprocessor.split_data(
        test_size=test_size,
        random_state=random_state
    )
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)
    
    return splits, preprocessor


if __name__ == '__main__':
    # テスト実行
    splits, preprocessor = load_and_preprocess_data_v07()
    
    print("\n--- FEATURE SUMMARY ---")
    print(f"Operational features: {preprocessor.operational_features}")
    print(f"Sensor features: {len(preprocessor.sensor_features)} sensors")
    print(f"Total features: {len(splits['feature_names'])}")
    
    print("\n--- SAMPLE DATA ---")
    print(splits['X_train'].head())
    
    print("\n--- TARGET DISTRIBUTION ---")
    print(f"Train: mean={splits['y_train'].mean():.1f}, std={splits['y_train'].std():.1f}")
    print(f"Val:   mean={splits['y_val'].mean():.1f}, std={splits['y_val'].std():.1f}")
    print(f"Test:  mean={splits['y_test'].mean():.1f}, std={splits['y_test'].std():.1f}")
