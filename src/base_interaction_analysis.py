"""
ベース特徴量交互作用分析モジュール（v0.3）
数値特徴とカテゴリ特徴の交互作用を生成
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class BaseFeatureInteractionAnalyzer:
    """ベース特徴量交互作用分析クラス"""
    
    def __init__(self, df: pd.DataFrame, feature_names: list):
        """
        Parameters
        ----------
        df : pd.DataFrame
            特徴量データ
        feature_names : list
            特徴量名のリスト
        """
        self.df = df
        self.feature_names = feature_names
        self.interaction_features = []
        self.numeric_features = []
        self.categorical_features = []
        
    def identify_feature_types(self):
        """
        数値特徴とカテゴリ特徴を識別
        """
        self.numeric_features = []
        self.categorical_features = []
        
        for col in self.feature_names:
            if 'encoded' in col or col in ['material_type_encoded', 
                                            'bridge_form_encoded',
                                            'bridge_type_encoded',
                                            'coastal_zone_encoded',
                                            'emergency_road_encoded',
                                            'DID_district_encoded',
                                            'bus_route_encoded',
                                            'damage_category_encoded']:
                self.categorical_features.append(col)
            else:
                self.numeric_features.append(col)
        
        print(f"\n[Base Feature Interaction Analysis]")
        print(f"Identified {len(self.numeric_features)} numeric features")
        print(f"Identified {len(self.categorical_features)} categorical features")
        
        return self.numeric_features, self.categorical_features
    
    def analyze_feature_importance(self, y, top_k=10):
        """
        特徴量の重要度を分析（相関ベース）
        
        Parameters
        ----------
        y : array-like
            ターゲット変数
        top_k : int
            選択する上位特徴数
            
        Returns
        -------
        list
            重要な数値特徴のリスト
        """
        print(f"\nAnalyzing feature importance...")
        
        # 数値特徴とターゲットの相関
        correlations = {}
        for feature in self.numeric_features:
            if feature in self.df.columns:
                corr = abs(np.corrcoef(self.df[feature], y)[0, 1])
                if not np.isnan(corr):
                    correlations[feature] = corr
        
        # 相関が高い順にソート
        sorted_features = sorted(correlations.items(), 
                                key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {min(top_k, len(sorted_features))} important numeric features:")
        for i, (feature, corr) in enumerate(sorted_features[:top_k], 1):
            print(f"  {i:2d}. {feature}: {corr:.4f}")
        
        top_features = [f for f, _ in sorted_features[:top_k]]
        return top_features
    
    def create_numeric_interactions(self, feature_pairs, interaction_type='multiply'):
        """
        数値特徴同士の交互作用を生成
        
        Parameters
        ----------
        feature_pairs : list of tuple
            特徴ペアのリスト
        interaction_type : str
            'multiply': 積, 'divide': 比率, 'add': 和, 'subtract': 差
            
        Returns
        -------
        pd.DataFrame
            交互作用特徴を追加したDataFrame
        """
        print(f"\nCreating {len(feature_pairs)} numeric interactions ({interaction_type})...")
        
        df_new = self.df.copy()
        created_features = []
        
        for feat1, feat2 in feature_pairs:
            if feat1 not in df_new.columns or feat2 not in df_new.columns:
                continue
            
            if interaction_type == 'multiply':
                feature_name = f'{feat1}_x_{feat2}'
                df_new[feature_name] = df_new[feat1] * df_new[feat2]
            elif interaction_type == 'divide':
                feature_name = f'{feat1}_div_{feat2}'
                # ゼロ除算を避ける
                df_new[feature_name] = df_new[feat1] / (df_new[feat2] + 1e-6)
            elif interaction_type == 'add':
                feature_name = f'{feat1}_plus_{feat2}'
                df_new[feature_name] = df_new[feat1] + df_new[feat2]
            elif interaction_type == 'subtract':
                feature_name = f'{feat1}_minus_{feat2}'
                df_new[feature_name] = df_new[feat1] - df_new[feat2]
            else:
                continue
            
            created_features.append(feature_name)
        
        print(f"  Created {len(created_features)} interaction features")
        
        # 表示（最初の5個）
        if created_features:
            print(f"  Sample features:")
            for i, feat in enumerate(created_features[:5], 1):
                mean_val = df_new[feat].mean()
                std_val = df_new[feat].std()
                print(f"    {i}. {feat}: mean={mean_val:.4f}, std={std_val:.4f}")
        
        return df_new, created_features
    
    def create_categorical_interactions(self, numeric_features, categorical_features):
        """
        数値特徴×カテゴリ特徴の交互作用を生成
        （カテゴリ値で重み付けした数値特徴）
        
        Parameters
        ----------
        numeric_features : list
            数値特徴のリスト
        categorical_features : list
            カテゴリ特徴のリスト
            
        Returns
        -------
        pd.DataFrame, list
            交互作用特徴を追加したDataFrame、特徴名リスト
        """
        print(f"\nCreating numeric×categorical interactions...")
        
        df_new = self.df.copy()
        created_features = []
        
        for num_feat in numeric_features[:5]:  # 上位5個の数値特徴のみ
            for cat_feat in categorical_features[:3]:  # 上位3個のカテゴリ特徴のみ
                if num_feat not in df_new.columns or cat_feat not in df_new.columns:
                    continue
                
                feature_name = f'{num_feat}_by_{cat_feat}'
                # カテゴリでグループ化した数値特徴の平均値で補正
                df_new[feature_name] = df_new[num_feat] * (df_new[cat_feat] + 1)
                created_features.append(feature_name)
        
        print(f"  Created {len(created_features)} interaction features")
        
        return df_new, created_features
    
    def select_top_interactions(self, all_interactions, y, top_k=50):
        """
        相関が高い交互作用項を選択
        
        Parameters
        ----------
        all_interactions : list
            交互作用特徴名のリスト
        y : array-like
            ターゲット変数
        top_k : int
            選択する交互作用項の数
            
        Returns
        -------
        list
            選択された交互作用特徴のリスト
        """
        print(f"\nSelecting top {top_k} interactions by correlation with target...")
        
        # 各交互作用とターゲットの相関を計算
        correlations = {}
        for feature in all_interactions:
            if feature in self.df.columns:
                corr = abs(np.corrcoef(self.df[feature], y)[0, 1])
                if not np.isnan(corr) and not np.isinf(corr):
                    correlations[feature] = corr
        
        # 相関が高い順にソート
        sorted_interactions = sorted(correlations.items(), 
                                    key=lambda x: x[1], reverse=True)
        
        # 上位k個を選択
        self.interaction_features = [f for f, _ in sorted_interactions[:top_k]]
        
        print(f"\nSelected top {len(self.interaction_features)} interaction features:")
        for i, (feature, corr) in enumerate(sorted_interactions[:10], 1):
            print(f"  {i:2d}. {feature}: {corr:.4f}")
        
        if len(sorted_interactions) > 10:
            print(f"  ... (and {len(self.interaction_features) - 10} more)")
        
        return self.interaction_features


def analyze_and_create_base_interactions(df: pd.DataFrame,
                                         feature_names: list,
                                         y: np.ndarray,
                                         top_k_features=10,
                                         top_k_interactions=50):
    """
    ベース特徴量の交互作用分析と生成のワンストップ関数
    
    Parameters
    ----------
    df : pd.DataFrame
        特徴量データ
    feature_names : list
        ベース特徴量名のリスト
    y : np.ndarray
        ターゲット変数
    top_k_features : int
        選択する重要特徴数
    top_k_interactions : int
        選択する交互作用項の数
        
    Returns
    -------
    tuple
        (df_with_interactions, interaction_features, analyzer)
    """
    analyzer = BaseFeatureInteractionAnalyzer(df, feature_names)
    
    # 特徴タイプを識別
    numeric_features, categorical_features = analyzer.identify_feature_types()
    
    # 重要な数値特徴を選択
    top_numeric_features = analyzer.analyze_feature_importance(y, top_k=top_k_features)
    
    # 数値×数値の交互作用（積）
    numeric_pairs = list(combinations(top_numeric_features, 2))
    print(f"\nGenerating {len(numeric_pairs)} numeric×numeric pairs...")
    
    df_new, multiply_features = analyzer.create_numeric_interactions(
        numeric_pairs, interaction_type='multiply'
    )
    analyzer.df = df_new
    
    # 数値×カテゴリの交互作用
    df_new, categorical_features_created = analyzer.create_categorical_interactions(
        top_numeric_features, categorical_features
    )
    analyzer.df = df_new
    
    # 全交互作用から上位k個を選択
    all_interactions = multiply_features + categorical_features_created
    print(f"\nTotal interactions generated: {len(all_interactions)}")
    
    selected_interactions = analyzer.select_top_interactions(
        all_interactions, y, top_k=top_k_interactions
    )
    
    # 選択された交互作用のみを保持
    final_features = feature_names + selected_interactions
    df_final = df_new[final_features]
    
    print(f"\nFinal feature count: {len(final_features)}")
    print(f"  Base features: {len(feature_names)}")
    print(f"  Interaction features: {len(selected_interactions)}")
    
    return df_final, selected_interactions, analyzer
