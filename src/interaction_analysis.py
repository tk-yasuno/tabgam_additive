"""
交互作用分析モジュール
損傷IDのペアによる複合劣化を分析
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DamageInteractionAnalyzer:
    """損傷ID交互作用分析クラス"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            橋梁検査データ（damage_id列を含む）
        """
        self.df = df
        self.cooccurrence_matrix = None
        self.interaction_features = []
        
    def analyze_cooccurrence(self, bridge_id_col='bridge_id', 
                            damage_id_col='damage_id',
                            hazard_col='h_i',
                            min_support=10):
        """
        橋梁ごとの損傷ID共起パターンを分析
        
        Parameters
        ----------
        bridge_id_col : str
            橋梁IDカラム名
        damage_id_col : str
            損傷IDカラム名
        hazard_col : str
            ハザード率カラム名（これが>0の行のみを対象とする）
        min_support : int
            最小共起頻度（これ以下は除外）
            
        Returns
        -------
        pd.DataFrame
            共起マトリックス（32x32）
        """
        print(f"\n[Damage Interaction Analysis]")
        print(f"Analyzing co-occurrence patterns of {damage_id_col}...")
        
        # h_i > 0の行のみを使用（実際に損傷が発生している行）
        df_actual_damages = self.df[self.df[hazard_col] > 0].copy()
        print(f"  Filtered to {len(df_actual_damages):,} rows with {hazard_col} > 0")
        print(f"  (Original: {len(self.df):,} rows)")
        
        # 橋梁ごとに損傷IDのセットを作成
        bridge_damages = (
            df_actual_damages.groupby(bridge_id_col)[damage_id_col]
            .apply(lambda x: set(x.dropna().astype(int)))
            .reset_index()
        )
        
        print(f"  Bridges with actual damages: {len(bridge_damages):,}")
        
        # 橋梁あたりの損傷数の統計
        damage_counts = bridge_damages[damage_id_col].apply(len)
        print(f"  Damages per bridge: mean={damage_counts.mean():.1f}, "
              f"median={damage_counts.median():.0f}, "
              f"min={damage_counts.min()}, max={damage_counts.max()}")
        
        # 共起カウント
        cooccurrence_counts = Counter()
        
        for damage_set in bridge_damages[damage_id_col]:
            if len(damage_set) >= 2:
                # 同一橋梁内で発生している損傷IDのペアを列挙
                for pair in combinations(sorted(damage_set), 2):
                    cooccurrence_counts[pair] += 1
        
        # 32x32の共起マトリックスを作成
        damage_ids = range(1, 33)
        matrix = pd.DataFrame(0, index=damage_ids, columns=damage_ids)
        
        for (id1, id2), count in cooccurrence_counts.items():
            if count >= min_support:
                matrix.loc[id1, id2] = count
                matrix.loc[id2, id1] = count  # 対称
        
        self.cooccurrence_matrix = matrix
        
        # 頻出ペアを表示
        print(f"\nTop 20 co-occurring damage pairs (min_support={min_support}):")
        top_pairs = sorted(cooccurrence_counts.items(), 
                          key=lambda x: x[1], reverse=True)[:20]
        
        for i, ((id1, id2), count) in enumerate(top_pairs, 1):
            print(f"  {i:2d}. Damage {id1:2d} × Damage {id2:2d}: {count:4d} bridges")
        
        return matrix
    
    def select_top_interactions(self, top_k=50):
        """
        重要な交互作用項（共起頻度が高いペア）を選択
        
        Parameters
        ----------
        top_k : int
            選択する交互作用項の数
            
        Returns
        -------
        list of tuple
            選択された損傷IDペアのリスト [(id1, id2), ...]
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("Run analyze_cooccurrence() first")
        
        # 上三角行列から上位k個を抽出
        interactions = []
        for i in range(1, 33):
            for j in range(i + 1, 33):
                count = self.cooccurrence_matrix.loc[i, j]
                if count > 0:
                    interactions.append(((i, j), count))
        
        # 頻度順にソート
        interactions_sorted = sorted(interactions, 
                                    key=lambda x: x[1], reverse=True)
        
        # 上位k個を選択
        self.interaction_features = [pair for pair, _ in interactions_sorted[:top_k]]
        
        print(f"\nSelected top {len(self.interaction_features)} interaction features:")
        for i, (id1, id2) in enumerate(self.interaction_features[:10], 1):
            count = self.cooccurrence_matrix.loc[id1, id2]
            print(f"  {i:2d}. damage_{id1}_x_damage_{id2}: {count:4d} bridges")
        
        if len(self.interaction_features) > 10:
            print(f"  ... (and {len(self.interaction_features) - 10} more)")
        
        return self.interaction_features
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                    bridge_id_col='bridge_id',
                                    damage_id_col='damage_id',
                                    hazard_col='h_i'):
        """
        交互作用特徴量を生成
        
        Parameters
        ----------
        df : pd.DataFrame
            元データ
        bridge_id_col : str
            橋梁IDカラム名
        damage_id_col : str
            損傷IDカラム名
        hazard_col : str
            ハザード率カラム名（これが>0の行のみを対象とする）
            
        Returns
        -------
        pd.DataFrame
            交互作用特徴量を追加したDataFrame
        """
        if not self.interaction_features:
            raise ValueError("Run select_top_interactions() first")
        
        print(f"\nCreating {len(self.interaction_features)} interaction features...")
        
        df_new = df.copy()
        
        # 橋梁ごとの実際の損傷IDセット（h_i > 0の行のみ）
        df_actual_damages = df[df[hazard_col] > 0]
        bridge_damages = (
            df_actual_damages.groupby(bridge_id_col)[damage_id_col]
            .apply(lambda x: set(x.dropna().astype(int)))
            .to_dict()
        )
        
        # 各交互作用特徴量を生成（バイナリ：両方の損傷が存在するか）
        for id1, id2 in self.interaction_features:
            feature_name = f'damage_{id1}_x_damage_{id2}'
            df_new[feature_name] = df_new[bridge_id_col].map(
                lambda bid: int(id1 in bridge_damages.get(bid, set()) and 
                               id2 in bridge_damages.get(bid, set()))
            )
        
        interaction_cols = [f'damage_{id1}_x_damage_{id2}' 
                           for id1, id2 in self.interaction_features]
        
        print(f"Interaction features created: {len(interaction_cols)} columns")
        print(f"Non-zero samples per feature:")
        for col in interaction_cols[:5]:
            nonzero_count = (df_new[col] == 1).sum()
            print(f"  {col}: {nonzero_count:5d} ({nonzero_count/len(df_new)*100:.2f}%)")
        
        return df_new, interaction_cols
    
    def visualize_cooccurrence(self, output_path=None, top_n=20):
        """
        共起マトリックスを可視化
        
        Parameters
        ----------
        output_path : str, optional
            保存先パス
        top_n : int
            表示する損傷IDの数（頻度上位）
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("Run analyze_cooccurrence() first")
        
        # 頻度の高い損傷IDを選択
        total_cooccurrence = self.cooccurrence_matrix.sum(axis=1)
        top_damages = total_cooccurrence.nlargest(top_n).index
        
        # サブマトリックス作成
        submatrix = self.cooccurrence_matrix.loc[top_damages, top_damages]
        
        # ヒートマップ
        plt.figure(figsize=(12, 10))
        sns.heatmap(submatrix, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Co-occurrence Count'})
        plt.title(f'Damage ID Co-occurrence Matrix (Top {top_n})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Damage ID', fontsize=12)
        plt.ylabel('Damage ID', fontsize=12)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.close()


def analyze_and_create_interactions(df: pd.DataFrame,
                                    bridge_id_col='bridge_id',
                                    damage_id_col='damage_id',
                                    hazard_col='h_i',
                                    min_support=10,
                                    top_k=50):
    """
    損傷ID交互作用分析と特徴量生成のワンストップ関数
    
    Parameters
    ----------
    df : pd.DataFrame
        元データ
    bridge_id_col : str
        橋梁IDカラム名
    damage_id_col : str
        損傷IDカラム名
    hazard_col : str
        ハザード率カラム名（これが>0の行のみを対象とする）
    min_support : int
        最小共起頻度
    top_k : int
        選択する交互作用項の数
        
    Returns
    -------
    tuple
        (df_with_interactions, interaction_features, analyzer)
    """
    analyzer = DamageInteractionAnalyzer(df)
    
    # 共起分析（h_i > 0の行のみを使用）
    analyzer.analyze_cooccurrence(
        bridge_id_col=bridge_id_col,
        damage_id_col=damage_id_col,
        hazard_col=hazard_col,
        min_support=min_support
    )
    
    # 上位交互作用項を選択
    analyzer.select_top_interactions(top_k=top_k)
    
    # 交互作用特徴量を生成
    df_new, interaction_cols = analyzer.create_interaction_features(
        df, 
        bridge_id_col=bridge_id_col, 
        damage_id_col=damage_id_col,
        hazard_col=hazard_col
    )
    
    # 可視化
    output_dir = Path('outputs/figures/interactions')
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer.visualize_cooccurrence(
        output_path=output_dir / 'damage_cooccurrence_matrix.png'
    )
    
    return df_new, interaction_cols, analyzer
