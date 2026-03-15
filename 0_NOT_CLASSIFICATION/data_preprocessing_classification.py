"""
データ前処理・特徴量エンジニアリング（分類タスク用）

福知山市橋梁データ向け：健全度（I〜IV）の分類モデル
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


class BridgeClassificationPreprocessor:
    """橋梁健全度分類データの前処理クラス"""
    
    def __init__(self, data_path: str):
        """
        Parameters
        ----------
        data_path : str
            CSVファイルのパス（福知山市橋梁データ）
        """
        self.data_path = Path(data_path)
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()  # 健全度のエンコーダー
        
        # 数値特徴量
        self.numeric_features = [
            '経年数', '橋長', '幅員', '支間数', '交通量'
        ]
        
        # カテゴリ特徴量
        self.categorical_features = [
            '構造形式', '材料', '管理者', '損傷状況', '損傷程度', 
            '補修履歴', '周辺環境'
        ]
        
        # 識別子（モデルには使わない）
        self.id_columns = ['橋梁ID', '橋梁名', '路線名']
        
        # ターゲット変数
        self.target_column = '健全度'
        
    def load_data(self):
        """データ読み込み"""
        print(f"Loading data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(self.data_path, encoding='shift-jis')
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.data_path, encoding='cp932')
        
        print(f"Data loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        print(f"Columns: {list(self.df.columns)}")
        return self
    
    def preprocess_features(self):
        """特徴量の前処理"""
        print("\n=== Feature Preprocessing ===")
        
        # 欠損値確認
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values")
        
        # 数値特徴量の処理
        for col in self.numeric_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                # 欠損値は中央値で補完
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  {col}: filled {self.df[col].isnull().sum()} missing values with median={median_val:.2f}")
        
        # カテゴリ特徴量の処理
        for col in self.categorical_features:
            if col in self.df.columns:
                # 欠損値は'不明'で補完
                self.df[col].fillna('不明', inplace=True)
                
                # Label Encoding
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} categories encoded")
        
        print("\nFeature preprocessing completed")
        return self
    
    def create_target(self):
        """
        ターゲット変数（健全度）の作成
        
        健全度 I, II, III, IV を 0, 1, 2, 3 にエンコード
        """
        print("\n=== Creating Target Variable ===")
        
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # 健全度の分布確認
        print(f"\nTarget distribution (before encoding):")
        print(self.df[self.target_column].value_counts().sort_index())
        
        # ローマ数字をアラビア数字に変換（もし必要なら）
        soundness_map = {
            'I': 0, 'II': 1, 'III': 2, 'IV': 3,
            '1': 0, '2': 1, '3': 2, '4': 3,
            1: 0, 2: 1, 3: 2, 4: 3
        }
        
        # Label Encoding
        self.df['target'] = self.df[self.target_column].map(soundness_map)
        
        # マッピングできなかった値の確認
        if self.df['target'].isnull().sum() > 0:
            print(f"\nWarning: {self.df['target'].isnull().sum()} values could not be mapped")
            print("Unmapped values:")
            print(self.df[self.df['target'].isnull()][self.target_column].unique())
            # 欠損値を削除
            self.df = self.df[self.df['target'].notnull()].copy()
        
        self.df['target'] = self.df['target'].astype(int)
        
        print(f"\nTarget distribution (after encoding):")
        print(self.df['target'].value_counts().sort_index())
        print(f"\nClass balance:")
        for i in range(4):
            count = (self.df['target'] == i).sum()
            pct = count / len(self.df) * 100
            print(f"  Class {i}: {count:3d} samples ({pct:5.1f}%)")
        
        return self
    
    def get_feature_matrix(self):
        """
        特徴量行列とターゲットを取得
        
        Returns
        -------
        X : pd.DataFrame
            特徴量行列
        y : np.array
            ターゲット変数（0, 1, 2, 3）
        feature_names : list
            特徴量名のリスト
        """
        # 特徴量列を選択
        feature_cols = []
        
        # 数値特徴量
        for col in self.numeric_features:
            if col in self.df.columns:
                feature_cols.append(col)
        
        # エンコードされたカテゴリ特徴量
        for col in self.categorical_features:
            encoded_col = f'{col}_encoded'
            if encoded_col in self.df.columns:
                feature_cols.append(encoded_col)
        
        X = self.df[feature_cols].copy()
        y = self.df['target'].values
        
        print(f"\n=== Feature Matrix ===")
        print(f"Shape: {X.shape}")
        print(f"Features: {feature_cols}")
        
        return X, y, feature_cols
    
    def scale_features(self, X):
        """
        特徴量のスケーリング
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量行列
            
        Returns
        -------
        X_scaled : pd.DataFrame
            スケールされた特徴量行列
        """
        print("\n=== Feature Scaling ===")
        X_scaled = X.copy()
        
        # 数値特徴量のみスケーリング
        numeric_cols = [col for col in self.numeric_features if col in X.columns]
        
        if len(numeric_cols) > 0:
            X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            print(f"Scaled {len(numeric_cols)} numeric features")
        
        return X_scaled


def load_and_preprocess_classification_data(data_path: str, scale_features: bool = True):
    """
    福知山市橋梁データの読み込みと前処理（分類タスク用）
    
    Parameters
    ----------
    data_path : str
        CSVファイルのパス
    scale_features : bool
        特徴量をスケーリングするか
        
    Returns
    -------
    X : pd.DataFrame
        特徴量行列
    y : np.array
        ターゲット変数（健全度: 0, 1, 2, 3）
    feature_names : list
        特徴量名のリスト
    preprocessor : BridgeClassificationPreprocessor
        前処理オブジェクト
    """
    preprocessor = BridgeClassificationPreprocessor(data_path)
    
    # データ読み込み
    preprocessor.load_data()
    
    # 特徴量前処理
    preprocessor.preprocess_features()
    
    # ターゲット変数作成
    preprocessor.create_target()
    
    # 特徴量行列取得
    X, y, feature_names = preprocessor.get_feature_matrix()
    
    # スケーリング
    if scale_features:
        X = preprocessor.scale_features(X)
    
    print("\n" + "="*60)
    print("Data preprocessing completed successfully!")
    print("="*60)
    
    return X, y, feature_names, preprocessor


def create_stratified_splits(X, y, n_splits=5, random_state=42):
    """
    層化K分割交差検証のインデックスを生成
    
    Parameters
    ----------
    X : pd.DataFrame
        特徴量行列
    y : np.array
        ターゲット変数
    n_splits : int
        分割数
    random_state : int
        乱数シード
        
    Returns
    -------
    splits : list of tuples
        (train_idx, val_idx)のリスト
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y))
    
    print(f"\n=== Stratified K-Fold Splits ===")
    print(f"Number of splits: {n_splits}")
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {i+1}:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val:   {len(val_idx)} samples")
        
        # クラス分布確認
        train_dist = pd.Series(y[train_idx]).value_counts().sort_index()
        val_dist = pd.Series(y[val_idx]).value_counts().sort_index()
        print(f"  Train class distribution: {dict(train_dist)}")
        print(f"  Val class distribution:   {dict(val_dist)}")
    
    return splits


if __name__ == '__main__':
    # テスト実行
    data_path = Path(__file__).parent.parent / 'data' / 'fukuchiyama_bridge_sample_v0.6.csv'
    
    print("Testing BridgeClassificationPreprocessor...")
    print("="*60)
    
    X, y, feature_names, preprocessor = load_and_preprocess_classification_data(
        str(data_path),
        scale_features=True
    )
    
    # 交差検証分割の作成
    splits = create_stratified_splits(X, y, n_splits=5, random_state=42)
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
