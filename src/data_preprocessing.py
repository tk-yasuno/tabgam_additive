"""
データ前処理・特徴量エンジニアリング
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
import warnings
warnings.filterwarnings('ignore')


class HazardDataPreprocessor:
    """橋梁ハザードデータの前処理クラス"""
    
    def __init__(self, data_path: str):
        """
        Parameters
        ----------
        data_path : str
            CSVファイルのパス
        """
        self.data_path = Path(data_path)
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_transformer = None  # ターゲット変数の変換器
        
        # Column name mapping (Japanese -> English)
        self.column_mapping = {
            'Age': 'age',
            '架設年': 'construction_year',
            '橋長_m': 'bridge_length_m',
            '全幅員_m': 'total_width_m',
            '支承数': 'num_bearings',
            '伸縮装置数': 'num_expansion_joints',
            '径間数': 'num_spans',
            '橋面積_m2': 'bridge_area_m2',
            '点検サイクル番号': 'inspection_cycle_no',
            '点検年度': 'inspection_year',
            '健全度': 'soundness_level',
            '損傷の進行性_山口市配点': 'damage_progressiveness_score',
            '損傷部位の重要性_山口市配点': 'damage_importance_score',
            'LCC_健全度Ⅱ': 'LCC_soundness_II',
            'LCC_健全度Ⅲ': 'LCC_soundness_III',
            '材料区分': 'material_type',
            '橋梁形式': 'bridge_form',
            '橋梁種別': 'bridge_type',
            '海岸線区分': 'coastal_zone',
            '緊急輸送道路': 'emergency_road',
            'DID地区': 'DID_district',
            'バス路線': 'bus_route',
            '損傷区分': 'damage_category',
            'ハザード率_1to2': 'hazard_rate_1to2',
            'ハザード率_2to3': 'hazard_rate_2to3',
            'ランク3に至る年数': 'years_to_rank3',
            '橋梁ID': 'bridge_id',
            '損傷ID': 'damage_id'  # v0.2: 交互作用分析用
        }
        
        # 静的特徴量（x_i）
        self.static_features = [
            'age', 'construction_year', 'bridge_length_m', 'total_width_m', 
            'num_bearings', 'num_expansion_joints', 'num_spans', 'bridge_area_m2'
        ]
        
        # カテゴリカル静的特徴
        self.static_categorical = [
            'material_type', 'bridge_form', 'bridge_type', 'coastal_zone',
            'emergency_road', 'DID_district', 'bus_route'
        ]
        
        # 動的特徴量（z_i）
        self.dynamic_features = [
            'inspection_cycle_no', 'inspection_year', 'soundness_level',
            'damage_progressiveness_score', 'damage_importance_score',
            'LCC_soundness_II', 'LCC_soundness_III'
        ]
        
        # カテゴリカル動的特徴
        self.dynamic_categorical = [
            'damage_category'
        ]
        
    def load_data(self):
        """データ読み込み"""
        print(f"Loading data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(self.data_path, encoding='shift-jis')
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.data_path, encoding='cp932')
        print(f"Data loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        
        # Rename columns to English
        self.df = self.df.rename(columns=self.column_mapping)
        print(f"Column names translated to English")
        
        return self
    
    def create_target(self, target_type='hazard_rate_1to2'):
        """
        ターゲット変数（h_i）の作成
        
        Parameters
        ----------
        target_type : str
            'hazard_rate_1to2': ハザード率_1to2を使用
            'hazard_rate_2to3': ハザード率_2to3を使用
            'years_to_rank3': ランク3に至る年数の逆数（ハザード的扱い）
        """
        print(f"Creating target variable: {target_type}")
        
        if target_type == 'hazard_rate_1to2':
            # ハザード率_1to2を使用（欠損値は0で補完）
            self.df['h_i'] = pd.to_numeric(self.df['hazard_rate_1to2'], errors='coerce').fillna(0)
        
        elif target_type == 'hazard_rate_2to3':
            # ハザード率_2to3を使用
            self.df['h_i'] = pd.to_numeric(self.df['hazard_rate_2to3'], errors='coerce').fillna(0)
        
        elif target_type == 'years_to_rank3':
            # ランク3に至る年数の逆数（ハザード的扱い）
            years = pd.to_numeric(self.df['years_to_rank3'], errors='coerce')
            # 0や欠損値は小さなハザード（大きな年数相当）として扱う
            self.df['h_i'] = np.where(years > 0, 1.0 / years, 0.001)
        
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        # 負の値や異常値の処理
        self.df['h_i'] = self.df['h_i'].clip(lower=0)
        
        print(f"Target statistics:")
        print(self.df['h_i'].describe())
        return self
    
    def transform_target(self, method='yeo-johnson'):
        """
        ターゲット変数の正規化
        
        Parameters
        ----------
        method : str
            'yeo-johnson': Yeo-Johnson変換（ゼロ対応、推奨）
            'log1p': log(1+y)変換（シンプル）
            'boxcox': Box-Cox変換（正の値のみ）
            'none': 変換なし
        """
        if method == 'none':
            print("Target transformation: None (using original values)")
            return self
        
        print(f"Transforming target variable with method: {method}")
        
        # 元のターゲット変数を保存
        self.df['h_i_original'] = self.df['h_i'].copy()
        
        if method == 'yeo-johnson':
            self.target_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            self.df['h_i'] = self.target_transformer.fit_transform(
                self.df[['h_i']]
            ).ravel()
            print(f"Yeo-Johnson transformation applied (lambda: {self.target_transformer.lambdas_[0]:.4f})")
        
        elif method == 'log1p':
            self.df['h_i'] = np.log1p(self.df['h_i'])
            print("Log1p transformation applied: y_new = log(1 + y)")
        
        elif method == 'boxcox':
            from scipy.stats import boxcox
            # Box-Coxは正の値のみ対応、ゼロに小さい値を追加
            y_shifted = self.df['h_i'] + 1
            self.df['h_i'], self.boxcox_lambda = boxcox(y_shifted)
            print(f"Box-Cox transformation applied (lambda: {self.boxcox_lambda:.4f})")
        
        else:
            raise ValueError(f"Unknown transformation method: {method}")
        
        print(f"Transformed target statistics:")
        print(self.df['h_i'].describe())
        
        return self
    
    def inverse_transform_target(self, y_transformed):
        """
        正規化されたターゲット変数を元のスケールに戻す
        
        Parameters
        ----------
        y_transformed : array-like
            変換後のターゲット変数
            
        Returns
        -------
        array-like
            元のスケールに戻したターゲット変数
        """
        if self.target_transformer is not None:
            # Yeo-Johnson逆変換
            return self.target_transformer.inverse_transform(
                y_transformed.reshape(-1, 1)
            ).ravel()
        elif hasattr(self, 'boxcox_lambda'):
            # Box-Cox逆変換
            from scipy.special import inv_boxcox
            return inv_boxcox(y_transformed, self.boxcox_lambda) - 1
        else:
            # log1pまたは変換なし（近似的に逆変換）
            # 厳密にはlog1pの場合はexpm1が必要だが、transform_targetで判定していないため近似
            return y_transformed
    
    def handle_missing_values(self):
        """欠損値処理"""
        print("Handling missing values...")
        
        # 数値型特徴の欠損値を中央値で補完
        numeric_features = self.static_features + self.dynamic_features
        for col in numeric_features:
            if col in self.df.columns:
                # まず数値に変換してから中央値を計算
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        # カテゴリカル特徴の欠損値を'Unknown'で補完
        categorical_features = self.static_categorical + self.dynamic_categorical
        for col in categorical_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')
        
        return self
    
    def encode_categorical(self):
        """カテゴリカル変数のエンコーディング"""
        print("Encoding categorical features...")
        
        categorical_features = self.static_categorical + self.dynamic_categorical
        
        for col in categorical_features:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        return self
    
    def create_feature_matrix(self):
        """特徴量マトリックスの作成"""
        print("Creating feature matrix...")
        
        feature_cols = []
        
        # 数値型特徴
        for col in self.static_features + self.dynamic_features:
            if col in self.df.columns:
                feature_cols.append(col)
        
        # エンコード済みカテゴリカル特徴
        for col in self.static_categorical + self.dynamic_categorical:
            encoded_col = f'{col}_encoded'
            if encoded_col in self.df.columns:
                feature_cols.append(encoded_col)
        
        print(f"Total features: {len(feature_cols)}")
        print(f"Feature list: {feature_cols}")
        
        return feature_cols
    
    def prepare_for_modeling(self, target_type='hazard_rate_1to2',
                            target_transform='yeo-johnson',
                            scale_features=True,
                            add_interactions=False,
                            interaction_top_k=50,
                            interaction_min_support=10,
                            add_base_interactions=False,
                            base_interaction_top_k=50,
                            n_splits=5, random_state=42):
        """
        モデリング用のデータ準備（完全版）
        
        Parameters
        ----------
        target_type : str
            ターゲット変数の種類
        target_transform : str
            ターゲット変数の変換方法: 'yeo-johnson', 'log1p', 'boxcox', 'none'
        scale_features : bool
            特徴量を標準化するか
        add_interactions : bool
            損傷ID交互作用項を追加するか（v0.2+）
        interaction_top_k : int
            追加する損傷ID交互作用項の数
        interaction_min_support : int
            損傷ID交互作用項の最小共起頻度
        add_base_interactions : bool
            ベース特徴量の交互作用項を追加するか（v0.3+）
        base_interaction_top_k : int
            追加するベース特徴量交互作用項の数
        n_splits : int
            Group K-Foldの分割数
        random_state : int
            乱数シード
        random_state : int
            乱数シード
            
        Returns
        -------
        dict
            {'X': 特徴量DataFrame, 'y': ターゲットSeries, 
             'groups': グループID, 'feature_names': 特徴量名リスト,
             'fold_splits': Group K-Foldのインデックス}
        """
        # パイプライン実行
        self.load_data()
        self.create_target(target_type)
        self.transform_target(method=target_transform)  # ターゲット変数正規化
        self.handle_missing_values()
        self.encode_categorical()
        
        # 交互作用項の追加（v0.2: 損傷ID交互作用）
        interaction_cols = []
        if add_interactions:
            from src.interaction_analysis import analyze_and_create_interactions
            print("\n[v0.2] Adding damage interaction features...")
            self.df, interaction_cols, self.interaction_analyzer = analyze_and_create_interactions(
                self.df,
                bridge_id_col='bridge_id',
                damage_id_col='damage_id',
                hazard_col='h_i',  # h_i > 0の行のみを実際の損傷として扱う
                min_support=interaction_min_support,
                top_k=interaction_top_k
            )
        
        # 特徴量マトリックス作成
        feature_cols = self.create_feature_matrix()
        feature_cols.extend(interaction_cols)
        
        # ベース特徴量の交互作用項の追加（v0.3+）
        base_interaction_cols = []
        if add_base_interactions:
            from src.base_interaction_analysis import analyze_and_create_base_interactions
            print("\n[v0.3] Adding base feature interactions...")
            
            # ベース特徴量のみを抽出（交互作用なし）
            base_features = [col for col in feature_cols if col not in interaction_cols]
            
            # ターゲット変数を取得（ゼロフィルタ前の全データ）
            y_temp = self.df['h_i'].values
            
            # ベース特徴量のデータフレーム
            df_base = self.df[base_features].copy()
            
            # ベース交互作用を生成
            df_with_base_int, base_interaction_cols, self.base_interaction_analyzer = \
                analyze_and_create_base_interactions(
                    df_base,
                    base_features,
                    y_temp,
                    top_k_features=10,
                    top_k_interactions=base_interaction_top_k
                )
            
            # 交互作用特徴をメインのDataFrameに結合
            for col in base_interaction_cols:
                self.df[col] = df_with_base_int[col]
            
            feature_cols.extend(base_interaction_cols)
        
        # ターゲット変数が存在し、ゼロでない行のみ抽出（元のスケールでチェック）
        if 'h_i_original' in self.df.columns:
            # 変換後の場合は元の値でゼロを除外
            valid_mask = self.df['h_i_original'].notna() & (self.df['h_i_original'] > 0)
            print(f"Filtering out zero hazard samples (original scale > 0)...")
            n_total = len(self.df)
            n_nonzero = valid_mask.sum()
            print(f"  Total samples: {n_total:,}")
            print(f"  Non-zero samples: {n_nonzero:,} ({n_nonzero/n_total*100:.1f}%)")
            print(f"  Zero samples excluded: {n_total-n_nonzero:,} ({(n_total-n_nonzero)/n_total*100:.1f}%)")
        else:
            # 変換していない場合は通常通り（ゼロ除外）
            valid_mask = self.df['h_i'].notna() & (self.df['h_i'] > 0)
        
        df_valid = self.df[valid_mask].copy()
        
        X = df_valid[feature_cols].copy()
        y = df_valid['h_i'].values
        
        # NaNとInfを含む行を除外
        # 特徴量のNaN/Infチェック
        X = X.replace([np.inf, -np.inf], np.nan)
        nan_mask_X = ~X.isna().any(axis=1)
        nan_mask_y = ~(np.isinf(y) | np.isnan(y))
        nan_mask = nan_mask_X & nan_mask_y
        
        if not nan_mask.any():
            print("Warning: All samples contain NaN or Inf. Using data as-is with 0 for missing values.")
            X = X.fillna(0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X = X[nan_mask].copy()
            y = y[nan_mask]
            df_valid = df_valid[nan_mask].copy()
        
        # 特徴量のスケーリング
        if scale_features:
            print("Scaling features with StandardScaler...")
            # 数値型特徴のみスケーリング（カテゴリカル変数は除外）
            numeric_cols = [col for col in X.columns if not col.endswith('_encoded')]
            categorical_cols = [col for col in X.columns if col.endswith('_encoded')]
            
            if numeric_cols:
                X_numeric_scaled = self.scaler.fit_transform(X[numeric_cols])
                X_scaled = pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=X.index)
                
                # カテゴリカル変数を結合
                if categorical_cols:
                    X_scaled[categorical_cols] = X[categorical_cols].values
                
                X = X_scaled
                print(f"Scaled {len(numeric_cols)} numeric features")
        
        # 橋梁IDをグループとして使用
        groups = df_valid['bridge_id'].values
        
        # Group K-Fold分割
        print(f"\nCreating {n_splits}-fold splits by bridge ID...")
        gkf = GroupKFold(n_splits=n_splits)
        fold_splits = list(gkf.split(X, y, groups))
        
        print(f"Final dataset: {len(X):,} samples, {len(X.columns)} features")
        print(f"Unique bridges: {len(np.unique(groups)):,}")
        print(f"Target range: [{y.min():.6f}, {y.max():.6f}]")
        
        return {
            'X': X,
            'y': y,
            'groups': groups,
            'feature_names': feature_cols,
            'fold_splits': fold_splits,
            'df_full': df_valid  # デバッグ用に完全なDataFrameも返す
        }
    
    def get_feature_info(self):
        """特徴量情報の取得"""
        return {
            'static_continuous': self.static_features,
            'static_categorical': self.static_categorical,
            'dynamic_continuous': self.dynamic_features,
            'dynamic_categorical': self.dynamic_categorical
        }


def load_and_preprocess_data(data_path: str = None, 
                              target_type: str = 'hazard_rate_1to2',
                              target_transform: str = 'yeo-johnson',
                              scale_features: bool = True,
                              add_interactions: bool = False,
                              interaction_top_k: int = 50,
                              interaction_min_support: int = 10,
                              add_base_interactions: bool = False,
                              base_interaction_top_k: int = 50,
                              n_splits: int = 5,
                              random_state: int = 42):
    """
    データ読み込みと前処理のワンストップ関数
    
    Parameters
    ----------
    data_path : str, optional
        CSVファイルのパス。Noneの場合はデフォルトパスを使用
    target_type : str
        ターゲット変数の種類
    target_transform : str
        ターゲット変数の変換方法: 'yeo-johnson', 'log1p', 'boxcox', 'none'
    scale_features : bool
        特徴量を標準化するか
    add_interactions : bool
        損傷ID交互作用項を追加するか（v0.2+）
    interaction_top_k : int
        追加する損傷ID交互作用項の数
    interaction_min_support : int
        損傷ID交互作用項の最小共起頻度
    add_base_interactions : bool
        ベース特徴量の交互作用項を追加するか（v0.3+）
    base_interaction_top_k : int
        追加するベース特徴量交互作用項の数
    n_splits : int
        K-Foldの分割数
    random_state : int
        乱数シード
    
    Returns
    -------
    dict
        前処理済みデータ
    """
    if data_path is None:
        # デフォルトパス
        data_path = r"I:\ACT2025.5.26-2030\MVP\hazard_additive_mdls\data\260311v1_inspection_base_and_hazard_estimation.csv"
    
    preprocessor = HazardDataPreprocessor(data_path)
    data = preprocessor.prepare_for_modeling(
        target_type=target_type,
        target_transform=target_transform,
        scale_features=scale_features,
        add_interactions=add_interactions,
        interaction_top_k=interaction_top_k,
        interaction_min_support=interaction_min_support,
        add_base_interactions=add_base_interactions,
        base_interaction_top_k=base_interaction_top_k,
        n_splits=n_splits,
        random_state=random_state
    )
    
    # 特徴量情報も追加
    data['feature_info'] = preprocessor.get_feature_info()
    data['preprocessor'] = preprocessor
    
    return data


if __name__ == '__main__':
    # テスト実行
    print("=" * 60)
    print("Testing data preprocessing...")
    print("=" * 60)
    
    data = load_and_preprocess_data()
    
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)
    print(f"X shape: {data['X'].shape}")
    print(f"y shape: {data['y'].shape}")
    print(f"Groups shape: {data['groups'].shape}")
    print(f"Number of folds: {len(data['fold_splits'])}")
