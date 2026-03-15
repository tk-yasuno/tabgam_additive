# Diverse GAMs論文執筆スペース

## 論文ファイル

**メインファイル**: `1_Methodology/diverse_gams_bridge_deterioration.tex`

## コンパイル方法

```bash
cd 1_Methodology
pdflatex diverse_gams_bridge_deterioration.tex
bibtex diverse_gams_bridge_deterioration
pdflatex diverse_gams_bridge_deterioration.tex
pdflatex diverse_gams_bridge_deterioration.tex
```

または、Windows PowerShellから:

```powershell
cd C:\Users\yasun\LaTeX\papers_tabgam\1_Methodology
pdflatex diverse_gams_bridge_deterioration.tex
```

## ディレクトリ構成

```
papers_tabgam/
├── 1_Methodology/
│   └── diverse_gams_bridge_deterioration.tex  # メイン論文ファイル
├── results/
│   ├── all_models_comparison_v0.5.csv        # モデル比較結果
│   └── agentic_tuning_v0.5.json              # アンサンブル重み最適化結果
├── figures/                                   # 図表（必要に応じて追加）
├── data/                                      # データ統計情報
└── METHODOLOGY.md                             # 日本語方法論（参考資料）
```

## 論文概要

**タイトル**: Diverse Generalized Additive Models for Bridge Deterioration Prediction: Integrating Statistical GAMs with Neural Additive Models

**主要結果**:
- TabNAM: R²=0.215（個別モデル最高性能）
- GAM+TabNAM Ensemble: R²=0.216（最適化アンサンブル）
- GLM-GAM: R²=0.204（ベースライン）

**データ**: 45,058件の橋梁点検記録（地名秘匿化）

**貢献**:
1. 4つの解釈可能加法モデルの実装と比較
2. TabNet-based Neural Additive Model (TabNAM)の提案
3. 統計GAMと深層学習の相補的組み合わせの発見

## 図表の準備

論文に必要な図表（作成予定）:

- **Figure 1**: Model architecture comparison (GAM, FAM, GBAM, TabNAM)
- **Figure 2**: Feature importance comparison across models
- **Table 1**: Model performance comparison（既に本文に含まれる）
- **Table 2**: Top 10 features by permutation importance（既に本文に含まれる）

PDPやSHAP図は `I:\ACT2025.5.26-2030\MVP\tabgam_additive\outputs\figures\` から利用可能。

## 次のステップ

1. ✅ LaTeX論文ファイル作成完了
2. ✅ 結果データのコピー完了
3. ⬜ pdflatexでコンパイルして確認
4. ⬜ 図表の整備（必要に応じて）
5. ⬜ 参考文献の確認と追加
6. ⬜ Abstract/Introductionの推敲
7. ⬜ arXivへの投稿準備

## データの秘匿化

論文では以下のように秘匿化:
- 自治体名: "Japanese municipality (anonymized)"
- 地理情報: 非開示
- 使用データ: 統計情報とモデル性能のみ

個人情報保護法およびarXivのガイドラインに準拠。

## 連絡先

本研究に関する問い合わせは、論文投稿後に公開される連絡先へ。
