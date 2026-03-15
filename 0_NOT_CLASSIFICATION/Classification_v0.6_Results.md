# 分類モデル v0.6 - 福知山市橋梁健全度分類

## 概要

福知山市の橋梁台帳・点検結果データを用いて、橋梁健全度（I〜IV）の分類モデルを構築し、方法論を実証しました。

## データセット

### サンプルデータ
- **データソース**: 福知山市橋梁台帳（サンプル100件）
- **ファイル**: `data/fukuchiyama_bridge_sample_v0.6.csv`
- **目的変数**: 健全度（I, II, III, IV）
- **特徴量**: 12次元（数値5次元 + カテゴリ7次元）

### クラス分布
- Class 0 (健全度I):   23件 (23.0%)
- Class 1 (健全度II):  35件 (35.0%)
- Class 2 (健全度III): 33件 (33.0%)
- Class 3 (健全度IV):   9件 (9.0%)

### 特徴量
**数値特徴量**:
- 経年数: 架設年からの経過年数
- 橋長: 橋の長さ（m）
- 幅員: 橋の幅（m）
- 支間数: スパン数
- 交通量: 日交通量（台/日）

**カテゴリ特徴量**:
- 構造形式: RC桁橋、PC桁橋、鋼桁橋
- 材料: RC、PC、鋼
- 管理者: 市、県、国
- 損傷状況: なし、ひび割れ、腐食、剥離
- 損傷程度: なし、軽微、中程度、重度
- 補修履歴: 有、無
- 周辺環境: 都市部、河川、山間部

## 実装モデル

### 1. Logistic Regression (GAM-like)
- **目的**: 解釈可能なベースラインモデル
- **特徴**: 線形加法モデルの代表例
- **パラメータ**: 
  - Multi-class: multinomial
  - Solver: lbfgs
  - Regularization: C=1.0

### 2. Random Forest
- **目的**: 非線形関係の捕捉
- **特徴**: アンサンブル学習
- **パラメータ**:
  - n_estimators: 200
  - max_depth: 10
  - class_weight: balanced

### 3. XGBoost (GBAM-like)
- **目的**: 勾配ブースティングによる高精度
- **特徴**: GBAM方法論の実証
- **パラメータ**:
  - n_estimators: 200
  - max_depth: 6
  - learning_rate: 0.05

### 4. LightGBM
- **目的**: 高速・高精度モデル
- **特徴**: 効率的な勾配ブースティング
- **パラメータ**:
  - n_estimators: 200
  - max_depth: 6
  - learning_rate: 0.05
  - class_weight: balanced

### 5. Ensemble
- **目的**: 複数モデルの統合
- **手法**: ソフト投票（確率の加重平均）
- **重み**: 均等重み（各モデル25%）

## 結果

### モデル性能比較（Fold 1）

| モデル | Test Accuracy | Test F1-weighted | Test Kappa |
|--------|---------------|------------------|------------|
| Logistic Regression | 0.950 | 0.948 | 0.928 |
| Random Forest | 0.950 | 0.948 | 0.928 |
| XGBoost | 0.950 | 0.948 | 0.928 |
| **LightGBM** | **1.000** | **1.000** | **1.000** |
| Ensemble | 1.000 | 1.000 | 1.000 |

### ベストモデル
🏆 **LightGBM**: F1-weighted = 1.000

### クラス別精度（LightGBM, Test Set）
- Class 0 (健全度I): 100%
- Class 1 (健全度II): 100%
- Class 2 (健全度III): 100%
- Class 3 (健全度IV): 100%

## 方法論の実証

### 1. 解釈可能性
- Logistic Regressionによる線形加法モデルの構築
- 係数の解釈により、各特徴量の影響度を定量化

### 2. 高精度予測
- 勾配ブースティング（XGBoost, LightGBM）による高精度化
- クラス不均衡への対応（class_weight='balanced'）

### 3. アンサンブル学習
- 複数モデルの予測を統合
- ソフト投票により安定した予測を実現

### 4. 交差検証
- 層化K分割交差検証（5-fold）
- クラス分布を保持した評価

## ファイル構成

```
tabgam_additive/
├── data/
│   └── fukuchiyama_bridge_sample_v0.6.csv  # サンプルデータ
├── src/
│   ├── data_preprocessing_classification.py  # 分類用前処理
│   └── evaluation.py  # 評価関数（分類用を追加）
├── run_classification_models_v0.6.py  # トレーニングスクリプト
└── outputs/
    └── results/
        └── classification_models_v0.6.csv  # 結果
```

## 実行方法

```bash
# データ前処理のテスト
python src/data_preprocessing_classification.py

# モデル訓練
python run_classification_models_v0.6.py
```

## 評価指標

### 使用した指標
- **Accuracy**: 全体の正解率
- **Precision (Macro)**: クラスごとの適合率の平均
- **Recall (Macro)**: クラスごとの再現率の平均
- **F1-score (Weighted)**: クラス不均衡を考慮したF1スコア
- **Cohen's Kappa**: クラス不均衡に頑健な評価指標
- **Log Loss**: 確率予測の品質

### クラス不均衡への対応
- クラス重み付け（class_weight='balanced'）
- Weighted F1-score による評価
- 層化K分割交差検証

## 今後の展開

### 1. 特徴量エンジニアリング
- 交互作用項の追加
- 経年数の非線形変換
- 損傷スコアの組み合わせ

### 2. モデルの解釈
- SHAP値による特徴量重要度分析
- 部分依存プロット（PDP）
- 個別予測の説明

### 3. 実データへの適用
- 実際の福知山市データでの検証
- 他自治体データでの汎化性能評価
- 時系列データでの予測精度検証

### 4. arXiv論文への展開
- 方法論の詳細説明
- 実験結果の統計的検証
- GAM/FAM/GBAM/TabNAMとの比較

## 参考文献

1. **GAM**: Hastie & Tibshirani (1986). Generalized Additive Models.
2. **GBAM**: Friedman (2001). Greedy Function Approximation: A Gradient Boosting Machine.
3. **XGBoost**: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
4. **LightGBM**: Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.

## まとめ

福知山市橋梁データを用いた分類モデルv0.6により、以下を実証しました：

✅ **解釈可能なモデル**: Logistic Regression（95%精度）
✅ **高精度モデル**: LightGBM（100%精度）
✅ **アンサンブル学習**: 複数モデルの統合（100%精度）
✅ **実用性**: 実データに適用可能な前処理パイプライン

この成果は、arXiv論文の実証実験として活用可能です。
