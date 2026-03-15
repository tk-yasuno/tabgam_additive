# Diverse GAMs論文 - クイックスタートガイド

## 📄 論文情報

**ファイル**: `1_Methodology/diverse_gams_bridge_deterioration.tex`

**タイトル**: Diverse Generalized Additive Models for Bridge Deterioration Prediction: Integrating Statistical GAMs with Neural Additive Models

**ページ数**: 約12ページ（2カラム形式）

## 🚀 今すぐコンパイル

### 方法1: PowerShellスクリプト（推奨）
```powershell
.\compile_paper.ps1
```

### 方法2: バッチファイル
```cmd
compile_paper.bat
```

### 方法3: 手動コンパイル
```powershell
cd 1_Methodology
pdflatex diverse_gams_bridge_deterioration.tex
pdflatex diverse_gams_bridge_deterioration.tex
pdflatex diverse_gams_bridge_deterioration.tex
```

## 📊 主要な結果（論文 Table 1）

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 0.674 | 0.512 | 0.132 |
| Random Forest | 0.651 | 0.491 | 0.191 |
| XGBoost (full) | 0.638 | 0.484 | **0.221** |
| **GLM-GAM** | 0.644 | 0.494 | 0.204 |
| FAM | 0.650 | 0.500 | 0.188 |
| GBAM | 0.643 | 0.490 | 0.205 |
| **TabNAM** | **0.639** | **0.487** | **0.215** |
| Uniform Ensemble | 0.641 | 0.491 | 0.210 |
| **Optimized Ensemble** | **0.639** | **0.487** | **0.216** |

## 📁 ファイル構成

```
C:\Users\yasun\LaTeX\papers_tabgam\
│
├── 📄 compile_paper.ps1           # PowerShellコンパイルスクリプト
├── 📄 compile_paper.bat           # バッチコンパイルスクリプト
├── 📄 README_Methodology.md       # 詳細な説明
├── 📄 QUICKSTART.md               # このファイル
├── 📄 METHODOLOGY.md              # 日本語方法論（参考）
│
├── 1_Methodology/
│   └── 📄 diverse_gams_bridge_deterioration.tex  # ⭐メイン論文
│
├── results/
│   ├── 📊 all_models_comparison_v0.5.csv        # モデル比較結果
│   └── 📊 agentic_tuning_v0.5.json              # アンサンブル重み
│
├── figures/                       # 図表ディレクトリ
├── data/                          # データ統計
└── 0_Format_exam_arXiv_pdfLaTex/ # テンプレート参考
```

## 🎯 論文の構成（セクション）

1. **Introduction** (2ページ)
   - 背景：橋梁インフラの劣化予測
   - 研究目的：解釈可能性と性能の両立
   - 貢献：Diverse GAMsフレームワーク

2. **Related Work** (1.5ページ)
   - 古典的GAM
   - Tree-based Additive Models
   - Neural Additive Models (NAM)
   - 橋梁劣化予測の先行研究

3. **Methodology** (3ページ)
   - 問題定式化
   - 4つのモデル：GLM-GAM, FAM, GBAM, TabNAM
   - アンサンブル戦略

4. **Data and Preprocessing** (1.5ページ)
   - 45,058件の橋梁点検データ
   - 特徴量エンジニアリング（73次元）
   - Yeo-Johnson変換

5. **Experiments** (2ページ)
   - 実験設定
   - ベースラインモデル
   - 結果比較（Table 1）
   - Ablation studies

6. **Interpretability Analysis** (2ページ)
   - 特徴量重要度（Table 2）
   - 部分依存プロット（PDP）
   - 実務的示唆

7. **Discussion** (1ページ)
   - TabNAMの優位性
   - GAMとTabNAMの相補性
   - 限界と今後の課題

8. **Conclusion** (0.5ページ)
   - 主要な成果
   - 実用的な貢献

## 🔑 主要な貢献

✅ **Diverse GAMsフレームワーク**: 4つの異なる加法モデルの統合  
✅ **TabNAM**: TabNet-based Neural Additive Modelの提案  
✅ **実証評価**: 45,058件の実データで検証  
✅ **解釈可能性**: PDPと特徴量重要度による洞察  
✅ **相補的アンサンブル**: GAM（統計）+ TabNAM（深層学習）

## 📝 次のステップ

### 1. コンパイル確認
```powershell
.\compile_paper.ps1
```
→ PDFが自動的に開きます

### 2. 図表の追加（必要に応じて）
- `figures/` ディレクトリに配置
- LaTeX本文の `\includegraphics` で参照

### 3. 参考文献の確認
- 25本の参考文献を含む
- 必要に応じて追加・修正

### 4. Abstract/Introductionの推敲
- 投稿前に英文校正を推奨

### 5. arXiv投稿準備
- PDFとソースファイル（.tex）を準備
- 図表ファイルも含める

## 📮 投稿先候補

- **arXiv**: cs.LG (Machine Learning) または stat.ML (Statistics)
- **会議**: NeurIPS, ICML, ICLR (ML), KDD (Data Mining)
- **ジャーナル**: Machine Learning, JMLR, Engineering Applications of AI

## ⚠️ データの秘匿化

✅ 地名・自治体名は非公開  
✅ 地理情報（緯度・経度）は論文に含まず  
✅ 統計情報とモデル性能のみ使用  
✅ 個人情報保護法に準拠

## 💡 ヒント

- **pdflatexが必要**: TeX Liveまたは MiKTeX をインストール
- **コンパイル時間**: 初回は約30秒（3回パス）
- **エラー時**: `.log`ファイルを確認
- **図表未配置**: プレースホルダーとして空白になります

---

**作成日**: 2026年3月15日  
**バージョン**: v1.0  
**形式**: pdfLaTeX (arXiv投稿形式)
