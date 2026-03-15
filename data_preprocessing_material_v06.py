"""
材料劣化データの前処理 - v0.6
Material Degradation Data Preprocessing for Multi-Mechanism Analysis

目的:
- 複合劣化メカニズムの相互作用を分析するためのデータ前処理
- v0.5のHazardDataPreprocessorと同じ構造を維持
- アルゴリズムはv0.5と完全に同一
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
import warnings
warnings.filterwarnings('ignore')


class MaterialDegradationPreprocessor:
    """材料劣化データの前処理クラス - v0.6"""
    
    def __init__(self, data_path: str):
        """
        Parameters
        ----------
        data_path : str
            Excelファイルのパス
        """
        self.data_path = Path(data_path)
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_transformer = None
        
        # カテゴリカル特徴量
        self.categorical_features = [
            'macro_sector',
            'outcome',
            'final_scenario',
            'equipment_involved',
            'substance_hazard',
            'age_category',
            'continent'
        ]
        
        # 数値特徴量
        self.numerical_features = [
            'year',
            'decade',
        ]
        
        # 劣化メカニズム特徴量（One-hot + 相互作用）
        self.mechanism_features = []
        
    def load_data(self):
        """データ読み込み"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_excel(self.data_path, header=8)
        print(f"Data loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        return self
    
    def clean_column_names(self):
        """カラム名のクリーニング"""
        print("Cleaning column names...")
        
        column_mapping = {
            'No.': 'no',
            'Code': 'code',
            'Source': 'source',
            'Date': 'date',
            'Place': 'place',
            'Continent': 'continent',
            'Macro-Sector': 'macro_sector',
            'Outcome': 'outcome',
            'Final Scenario': 'final_scenario',
            'Cause': 'cause',
            'Equipment involved': 'equipment_involved',
            'Action taken': 'action_taken',
            'Substance hazard classification': 'substance_hazard',
            'Injured': 'injured',
            'Fatality': 'fatality',
            'Economical losses': 'economical_losses',
            'Environmental contamination': 'environmental_contamination',
            'Age': 'age_category'
        }
        
        self.df = self.df.rename(columns=column_mapping)
        print(f"Column names cleaned")
        return self
    
    def create_temporal_features(self):
        """時系列特徴量の作成"""
        print("Creating temporal features...")
        
        # Date から Year を抽出
        self.df['year'] = pd.to_datetime(self.df['date'], errors='coerce').dt.year
        
        # Decade（10年単位）
        self.df['decade'] = (self.df['year'] // 10) * 10
        
        # Year を正規化（1966-2023 → 0-1）
        self.df['year_normalized'] = (self.df['year'] - self.df['year'].min()) / (
            self.df['year'].max() - self.df['year'].min()
        )
        
        print(f"Temporal features created: year, decade, year_normalized")
        return self
    
    def create_mechanism_features(self):
        """
        劣化メカニズム特徴量の作成
        
        重要: 複合劣化（Corrosion + Fatigue など）を考慮
        - 各メカニズムをバイナリ特徴量化
        - 複合メカニズムは複数のフラグが立つ
        """
        print("Creating degradation mechanism features...")
        
        # 主要な劣化メカニズム
        mechanisms = [
            'Corrosion',
            'Fatigue',
            'Erosion',
            'Vibrations',
            'Hydrogen Embrittlement',
            'Unspecified material degradation'
        ]
        
        # 各メカニズムのバイナリ特徴量を作成
        for mechanism in mechanisms:
            col_name = f'mechanism_{mechanism.lower().replace(" ", "_")}'
            # "Corrosion + Fatigue" のような複合メカニズムも検出
            self.df[col_name] = self.df['cause'].str.contains(
                mechanism, case=False, na=False
            ).astype(int)
            self.mechanism_features.append(col_name)
        
        # 複合メカニズムフラグ（2つ以上のメカニズムが同時に発生）
        mechanism_cols = [col for col in self.df.columns if col.startswith('mechanism_')]
        self.df['is_multi_mechanism'] = (
            self.df[mechanism_cols].sum(axis=1) > 1
        ).astype(int)
        
        # メカニズムの数
        self.df['num_mechanisms'] = self.df[mechanism_cols].sum(axis=1)
        
        print(f"Mechanism features created: {len(self.mechanism_features)} mechanisms")
        print(f"  Multi-mechanism events: {self.df['is_multi_mechanism'].sum()} / {len(self.df)}")
        
        return self
    
    def create_target(self, target_type='severity_score'):
        """
        ターゲット変数の作成
        
        Parameters
        ----------
        target_type : str
            'severity_score': 事故重大度スコア（推奨）
            'binary_severe': 重大事故かどうか（2値分類）
        """
        print(f"Creating target variable: {target_type}")
        
        if target_type == 'severity_score':
            # 事故重大度スコアの作成
            self.df['severity_score'] = self.df.apply(
                self._calculate_severity_score, axis=1
            )
            self.df['h_i'] = self.df['severity_score']
        
        elif target_type == 'binary_severe':
            # 重大事故フラグ（Accident or 死傷者あり）
            self.df['h_i'] = (
                (self.df['outcome'] == 'Accident') |
                (self.df['injured'].isin(['MI', 'SI'])) |
                (self.df['fatality'].isin(['MF', 'SF']))
            ).astype(int)
        
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        print(f"Target statistics:")
        print(self.df['h_i'].describe())
        
        return self
    
    def _calculate_severity_score(self, row):
        """
        事故重大度スコアの計算
        
        Parameters
        ----------
        row : pd.Series
            データ行
        
        Returns
        -------
        score : float
            重大度スコア（0-5の範囲）
        """
        score = 0.0
        
        # 1. Outcome (30%)
        outcome_map = {
            'Near Miss': 1,
            'Incident': 2,
            'LOC': 3,
            'Accident': 4,
            'Unknown': 2  # 中間値
        }
        score += outcome_map.get(row['outcome'], 2) * 0.3
        
        # 2. Injured (20%)
        injured_map = {
            'NI': 0,     # No Injury
            'SI': 2,     # Single Injury
            'MI': 4,     # Multiple Injuries
            'Unknown': 0
        }
        score += injured_map.get(row['injured'], 0) * 0.2
        
        # 3. Fatality (30%)
        fatality_map = {
            'NF': 0,     # No Fatality
            'SF': 3,     # Single Fatality
            'MF': 5,     # Multiple Fatalities
            'Unknown': 0
        }
        score += fatality_map.get(row['fatality'], 0) * 0.3
        
        # 4. Economical losses (10%)
        econ_map = {
            'up to $100,000': 1,
            'between $100,000 and $1 million': 2,
            'between $1 million and $10 million': 3,
            'greater than $10 million': 4,
            'Unknown': 0
        }
        score += econ_map.get(row['economical_losses'], 0) * 0.1
        
        # 5. Environmental contamination (10%)
        env_map = {
            'ND': 0,     # No Damage
            'MMD': 1,    # Minor or Moderate Damages
            'SED': 2,    # Severe Environmental Damage
            'Unknown': 0
        }
        score += env_map.get(row['environmental_contamination'], 0) * 0.1
        
        return score
    
    def transform_target(self, method='yeo-johnson'):
        """
        ターゲット変数の正規化
        
        Parameters
        ----------
        method : str
            'yeo-johnson': Yeo-Johnson変換（推奨）
            'log1p': log(1+y)変換
            'none': 変換なし
        """
        print(f"Transforming target variable: {method}")
        
        if method == 'none':
            print("  No transformation applied")
            return self
        
        if method == 'yeo-johnson':
            self.target_transformer = PowerTransformer(method='yeo-johnson')
            self.df['h_i'] = self.target_transformer.fit_transform(
                self.df[['h_i']]
            ).ravel()
        
        elif method == 'log1p':
            self.df['h_i'] = np.log1p(self.df['h_i'])
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"  Transformed target statistics:")
        print(self.df['h_i'].describe())
        
        return self
    
    def encode_categorical_features(self):
        """カテゴリカル特徴量のエンコーディング"""
        print("Encoding categorical features...")
        
        for col in self.categorical_features:
            if col in self.df.columns:
                # Unknownを欠損値として扱う
                self.df[col] = self.df[col].replace('Unknown', np.nan)
                
                # Label Encoding
                le = LabelEncoder()
                mask = self.df[col].notna()
                self.df.loc[mask, col] = le.fit_transform(self.df.loc[mask, col])
                
                # 欠損値を-1で埋める
                self.df[col] = self.df[col].fillna(-1).astype(int)
                
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} categories")
        
        return self
    
    def normalize_numerical_features(self):
        """数値特徴量の正規化"""
        print("Normalizing numerical features...")
        
        numerical_cols = [col for col in self.numerical_features if col in self.df.columns]
        
        if len(numerical_cols) > 0:
            self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
            print(f"  Normalized {len(numerical_cols)} numerical features")
        
        return self
    
    def get_feature_matrix(self):
        """
        特徴量行列の取得
        
        Returns
        -------
        X : pd.DataFrame
            特徴量行列
        y : np.array
            ターゲット変数
        feature_names : list
            特徴量名のリスト
        """
        # 特徴量を選択
        feature_cols = (
            self.mechanism_features +
            ['is_multi_mechanism', 'num_mechanisms'] +
            self.numerical_features +
            self.categorical_features
        )
        
        # 存在する列のみ選択
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        X = self.df[feature_cols].copy()
        y = self.df['h_i'].values
        
        print(f"\nFeature matrix created:")
        print(f"  Shape: {X.shape}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target: {y.shape[0]} samples")
        
        return X, y, feature_cols
    
    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        データを訓練/検証/テストに分割
        
        Parameters
        ----------
        test_size : float
            テストセットの割合
        val_size : float
            検証セットの割合（訓練データから分割）
        random_state : int
            乱数シード
        
        Returns
        -------
        splits : dict
            {'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'}
        """
        X, y, feature_names = self.get_feature_matrix()
        
        # 訓練+検証 vs テスト
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 訓練 vs 検証
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"\nData split:")
        print(f"  Train: {X_train.shape[0]} samples ({100*(1-test_size-val_size):.1f}%)")
        print(f"  Val:   {X_val.shape[0]} samples ({100*val_size:.1f}%)")
        print(f"  Test:  {X_test.shape[0]} samples ({100*test_size:.1f}%)")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names
        }


def load_and_preprocess_data_v06(
    data_path='data/Dataset of material degradation event within process industry.xlsx',
    target_type='severity_score',
    target_transform='yeo-johnson',
    test_size=0.2,
    val_size=0.2,
    random_state=42
):
    """
    材料劣化データの読み込みと前処理 - v0.6
    
    Parameters
    ----------
    data_path : str
        データファイルのパス
    target_type : str
        'severity_score' or 'binary_severe'
    target_transform : str
        'yeo-johnson', 'log1p', or 'none'
    test_size : float
        テストセットの割合
    val_size : float
        検証セットの割合
    random_state : int
        乱数シード
    
    Returns
    -------
    splits : dict
        データ分割の辞書
    preprocessor : MaterialDegradationPreprocessor
        前処理器インスタンス
    """
    print("=" * 80)
    print("MATERIAL DEGRADATION DATA PREPROCESSING - v0.6")
    print("=" * 80)
    
    preprocessor = MaterialDegradationPreprocessor(data_path)
    
    preprocessor.load_data()
    preprocessor.clean_column_names()
    preprocessor.create_temporal_features()
    preprocessor.create_mechanism_features()
    preprocessor.create_target(target_type=target_type)
    preprocessor.transform_target(method=target_transform)
    preprocessor.encode_categorical_features()
    preprocessor.normalize_numerical_features()
    
    splits = preprocessor.split_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)
    
    return splits, preprocessor


if __name__ == '__main__':
    # テスト実行
    splits, preprocessor = load_and_preprocess_data_v06()
    
    print("\n--- FEATURE SUMMARY ---")
    print(f"Mechanism features: {preprocessor.mechanism_features}")
    print(f"Categorical features: {preprocessor.categorical_features}")
    print(f"Numerical features: {preprocessor.numerical_features}")
    
    print("\n--- SAMPLE DATA ---")
    print(splits['X_train'].head())
    print("\n--- TARGET DISTRIBUTION ---")
    print(f"Train: mean={splits['y_train'].mean():.3f}, std={splits['y_train'].std():.3f}")
    print(f"Val:   mean={splits['y_val'].mean():.3f}, std={splits['y_val'].std():.3f}")
    print(f"Test:  mean={splits['y_test'].mean():.3f}, std={splits['y_test'].std():.3f}")
