"""
航空エンジン劣化データの前処理 - v0.8
CMAPSS FD002 Data Preprocessing for Multi-Mechanism Degradation Analysis

目的:
- 複合劣化（摩耗 × 熱疲労 × 振動 × 運転条件）の相互作用を分析
- v0.5のアルゴリズムと同じ構造を維持
- RUL（Remaining Useful Life）予測

データセット:
- FD002: 6 operational conditions × 1 fault mode (HPC degradation only)
- Train: 260 engines
- Test: 259 engines
- 比較: FD004は2故障モード、FD002は1故障モード

研究目的:
- 故障モードが1つの場合の複合劣化パターンを検証
- FD004（2故障モード）との比較分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class CMAPSSPreprocessor:
    """CMAPSS FD002データの前処理クラス - v0.8"""
    
    def __init__(self, data_dir='data/CMAPSSData', dataset='FD002'):
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
        
        # 除外するセンサー（定数センサー）
        self.constant_sensors = []
        
        # 特徴量リスト
        self.feature_names = None
        
    def load_data(self):
        """データの読み込み"""
        print(f"Loading CMAPSS {self.dataset} data...")
        
        # 訓練データ
        train_path = self.data_dir / f'train_{self.dataset}.txt'
        self.train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=self.columns)
        
        # テストデータ
        test_path = self.data_dir / f'test_{self.dataset}.txt'
        self.test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=self.columns)
        
        # RUL真値（テスト用）
        rul_path = self.data_dir / f'RUL_{self.dataset}.txt'
        self.rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['true_rul'])
        
        print(f"  Train: {len(self.train_df):,} samples, {self.train_df['unit'].nunique()} engines")
        print(f"  Test:  {len(self.test_df):,} samples, {self.test_df['unit'].nunique()} engines")
        
    def remove_constant_sensors(self):
        """定数センサーを除外"""
        print("\nRemoving constant sensors...")
        
        # 訓練データで標準偏差がほぼ0のセンサーを検出
        for sensor in self.sensor_features:
            if self.train_df[sensor].std() < 0.0001:
                self.constant_sensors.append(sensor)
        
        if self.constant_sensors:
            print(f"  Removed constant sensors: {self.constant_sensors}")
            self.sensor_features = [s for s in self.sensor_features if s not in self.constant_sensors]
        else:
            print("  No constant sensors found")
    
    def create_rul_target(self, max_rul=125):
        """
        RULターゲットを作成
        
        Parameters
        ----------
        max_rul : int
            最大RUL値（piecewise linear用）
        """
        print(f"\nCreating RUL target (max_rul={max_rul})...")
        
        # 訓練データ: 各エンジンの最大サイクル数からRULを計算
        self.train_df['max_cycle'] = self.train_df.groupby('unit')['cycle'].transform('max')
        self.train_df['RUL'] = self.train_df['max_cycle'] - self.train_df['cycle']
        
        # Piecewise linear RUL (RUL > max_rul の場合は max_rul に制限)
        self.train_df['RUL_capped'] = self.train_df['RUL'].clip(upper=max_rul)
        
        print(f"  RUL range: [{self.train_df['RUL'].min()}, {self.train_df['RUL'].max()}]")
        print(f"  RUL_capped range: [{self.train_df['RUL_capped'].min()}, {self.train_df['RUL_capped'].max()}]")
        print(f"  Mean RUL: {self.train_df['RUL'].mean():.1f}")
        print(f"  Mean RUL_capped: {self.train_df['RUL_capped'].mean():.1f}")
        
        # テストデータ: 各エンジンの最後のサイクルに真のRUL値を付与
        test_last_cycles = self.test_df.groupby('unit')['cycle'].max().reset_index()
        test_last_cycles.columns = ['unit', 'max_cycle']
        test_last_cycles['true_rul'] = self.rul_df['true_rul'].values
        
        # テストデータの各サイクルにRULを計算
        self.test_df = self.test_df.merge(test_last_cycles[['unit', 'max_cycle', 'true_rul']], on='unit')
        self.test_df['RUL'] = self.test_df['true_rul'] + (self.test_df['max_cycle'] - self.test_df['cycle'])
        self.test_df['RUL_capped'] = self.test_df['RUL'].clip(upper=max_rul)
        
        print(f"  Test RUL range: [{self.test_df['RUL'].min()}, {self.test_df['RUL'].max()}]")
        
    def create_rolling_features(self, window_sizes=[5, 10, 20]):
        """
        ローリング統計特徴量を作成
        
        Parameters
        ----------
        window_sizes : list
            ウィンドウサイズのリスト
        """
        print(f"\nCreating rolling features (windows={window_sizes})...")
        
        # 主要なセンサーのみ選択（温度、圧力、回転数系）
        key_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11']
        key_sensors = [s for s in key_sensors if s in self.sensor_features]
        
        # 特徴量名のリストを先に作成
        rolling_features = []
        for sensor in key_sensors:
            for window in window_sizes:
                rolling_features.append(f'{sensor}_roll_mean_{window}')
                rolling_features.append(f'{sensor}_roll_std_{window}')
        
        # 各DataFrameに対してローリング特徴量を作成
        for df in [self.train_df, self.test_df]:
            for sensor in key_sensors:
                for window in window_sizes:
                    # 移動平均
                    col_name_mean = f'{sensor}_roll_mean_{window}'
                    df[col_name_mean] = df.groupby('unit')[sensor].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    
                    # 移動標準偏差
                    col_name_std = f'{sensor}_roll_std_{window}'
                    df[col_name_std] = df.groupby('unit')[sensor].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
        
        # NaNを0で埋める
        self.train_df[rolling_features] = self.train_df[rolling_features].fillna(0)
        self.test_df[rolling_features] = self.test_df[rolling_features].fillna(0)
        
        print(f"  Created {len(rolling_features)} rolling features")
        
        return rolling_features
    
    def normalize_features(self):
        """特徴量の正規化"""
        print("\nNormalizing features...")
        
        # 正規化対象の特徴量
        features_to_normalize = self.operational_features + self.sensor_features
        
        # サイクル数も正規化
        self.train_df['cycle_norm'] = self.train_df['cycle'] / self.train_df['cycle'].max()
        self.test_df['cycle_norm'] = self.test_df['cycle'] / self.test_df['cycle'].max()
        
        # センサーと運転条件を正規化
        self.scaler.fit(self.train_df[features_to_normalize])
        self.train_df[features_to_normalize] = self.scaler.transform(self.train_df[features_to_normalize])
        self.test_df[features_to_normalize] = self.scaler.transform(self.test_df[features_to_normalize])
        
        print(f"  Normalized {len(features_to_normalize)} features")
    
    def get_feature_matrix(self, target_col='RUL_capped'):
        """
        特徴量マトリクスとターゲットを取得
        
        Parameters
        ----------
        target_col : str
            ターゲット列名
            
        Returns
        -------
        X_train, X_test : pd.DataFrame
            特徴量マトリクス
        y_train, y_test : np.array
            ターゲット値
        feature_names : list
            特徴量名のリスト
        """
        # 特徴量カラムを構築
        feature_cols = self.operational_features + self.sensor_features + ['cycle_norm']
        
        # ローリング特徴量を追加
        rolling_cols = [col for col in self.train_df.columns if 'roll_' in col]
        feature_cols.extend(rolling_cols)
        
        # 特徴量マトリクス
        X_train = self.train_df[feature_cols].copy()
        X_test = self.test_df[feature_cols].copy()
        
        # ターゲット
        y_train = self.train_df[target_col].values
        y_test = self.test_df[target_col].values
        
        print("\nFeature matrix created:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target: {target_col}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        訓練データを訓練セットと検証セットに分割（エンジン単位）
        
        Parameters
        ----------
        test_size : float
            検証セットの割合
        random_state : int
            乱数シード
            
        Returns
        -------
        splits : dict
            {'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'}
        """
        X_train_full, X_test, y_train_full, y_test, feature_names = self.get_feature_matrix()
        
        # エンジン単位で分割
        unique_units = self.train_df['unit'].unique()
        train_units, val_units = train_test_split(
            unique_units, test_size=test_size, random_state=random_state
        )
        
        # マスクを作成
        train_mask = self.train_df['unit'].isin(train_units).values
        val_mask = self.train_df['unit'].isin(val_units).values
        
        X_train = X_train_full[train_mask]
        X_val = X_train_full[val_mask]
        y_train = y_train_full[train_mask]
        y_val = y_train_full[val_mask]
        
        print("\nData split (engine-based):")
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


def load_and_preprocess_data_v08(
    data_dir='data/CMAPSSData',
    dataset='FD002',
    max_rul=125,
    rolling_windows=[5, 10, 20],
    test_size=0.2,
    random_state=42
):
    """
    CMAPSS FD002データの読み込みと前処理 - v0.8
    
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
    print("CMAPSS FD002 DATA PREPROCESSING - v0.8")
    print("Jet Engine Multi-Mechanism Degradation Analysis (1 Fault Mode)")
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
    splits, preprocessor = load_and_preprocess_data_v08()
    
    print("\n--- FEATURE SUMMARY ---")
    print(f"Operational features: {preprocessor.operational_features}")
    print(f"Sensor features: {len(preprocessor.sensor_features)} sensors")
    print(f"Total features: {len(splits['feature_names'])}")
    
    print("\n--- SAMPLE DATA ---")
    print(f"X_train shape: {splits['X_train'].shape}")
    print(f"y_train range: [{splits['y_train'].min():.1f}, {splits['y_train'].max():.1f}]")
    print(f"y_train mean: {splits['y_train'].mean():.1f}")
    
    print("\n--- RUL STATISTICS ---")
    print(f"Training set:")
    print(f"  Mean RUL: {splits['y_train'].mean():.2f}")
    print(f"  Std RUL: {splits['y_train'].std():.2f}")
    print(f"Validation set:")
    print(f"  Mean RUL: {splits['y_val'].mean():.2f}")
    print(f"  Std RUL: {splits['y_val'].std():.2f}")
    print(f"Test set:")
    print(f"  Mean RUL: {splits['y_test'].mean():.2f}")
    print(f"  Std RUL: {splits['y_test'].std():.2f}")
