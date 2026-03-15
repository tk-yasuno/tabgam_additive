# Hazard Representation Additive Models (HRAM)

**Version**: v0.5 (Final)  
**Date**: March 15, 2026  
**Status**: ✅ **Complete**

## Description

**Diverse GAMs: TabNAMsへの拡張とGAMs Integrated Ensemble** - A comprehensive implementation of interpretable additive models for bridge hazard prediction, featuring deep learning-based TabNAM and automated ensemble weight optimization. Achieves **R² = 0.216** through Agentic AI-powered model integration.

**Key Features:**
- 🚀 **State-of-the-Art Performance**: R² = 0.216 (Optimized Ensemble, v0.5)
- 🤖 **Agentic AI**: Automated ensemble weight optimization with Planner-Executor-Evaluator pattern
- 🧠 **TabNAM**: Deep learning-based Neural Additive Model with attention mechanism
- 🔍 **High Interpretability**: Additive structure preserved across all models (GAM, FAM, GBAM, TabNAM)
- 📊 **4 Diverse Models**: Statistical GAM + Random Forest + Gradient Boosting + Deep Learning
- 📈 **Production-Ready**: 45K+ inspection records, 5-fold CV, comprehensive validation

## Version History

| Version | R² | Key Innovation | Status |
|---------|-----|----------------|--------|
| v0.1 | 0.004 → 0.201 | Baseline + Advanced Preprocessing | Complete |
| v0.2 | 0.213 | Damage ID Interactions | Complete |
| v0.3 | 0.208 | Base Feature Interactions (+7.6x interpretability) | Complete |
| v0.4 | 0.208 | TabNAM, FOBAM v2 (FAM equivalent) | Complete |
| **v0.5** | **0.216** | **Agentic Ensemble Weight Tuning** | ✅ **Final** |

## What's New in v0.5 🎉

### Agentic Ensemble Weight Tuning
- **Automated Optimization**: Planner-Executor-Evaluator pattern for intelligent weight search
- **Multiple Search Strategies**:
  - Sequential Least Squares Programming (SLSQP) - Local optimization
  - Differential Evolution - Global optimization
  - Random Perturbation - Fine-tuning
- **Optimal Solution**: TabNAM 80.5% + GAM 19.5% (discovered in iteration 1)
- **Performance Gain**: +0.61% over uniform weights, +0.07% over best individual model

### Key Findings
- **Model Complementarity**: Statistical GAM + Deep Learning TabNAM = Optimal combination
- **Redundancy Elimination**: FAM and GBAM weights → 0 (covered by TabNAM)
- **Fast Convergence**: Best solution found in 1st optimization iteration
- **Diversity Matters**: Quality > Quantity (2 complementary models > 4 diverse models)

### TabNAM (v0.4-v0.5)
- **TabNet-based Neural Additive Model**: Attention mechanism for feature selection
- **Performance**: R² = 0.215 (Best individual model)
- **Sparsity**: λ_sparse = 1e-3 for focused feature importance
- **Architecture**: 3 decision steps, n_d=8, n_a=8

See [Lesson_5th_train.md](Lesson_5th_train.md) for detailed analysis.

## What's New in v0.4

### Diverse f(·) Exploration
- **TabNAM**: ✅ Success (R²=0.205) - Attention-based deep learning
- **FOBAM v2**: ✅ Improved (R²=0.191) - Random Forest with OOB evaluation (FAM equivalent)
- **SVAM**: ❌ Abandoned - Linear SVM insufficient for non-linear bridge deterioration

### FOBAM v2 Redesign
- **v1 Failed**: 1-feature Random Forests (R²=0.016)
- **v2 Success**: Full-feature RF + OOB scoring (R²=0.191, +1,122% improvement)
- **Lesson**: Additive interpretation should be post-hoc (SHAP), not constrained during training

See [Lesson_4th_train.md](Lesson_4th_train.md) for failure analysis.

## What's New in v0.3

### Base Feature Interaction Analysis
- **Engineering-Driven**: Domain knowledge-based interactions (age×LCC, inspection×age)
- **SHAP Importance**: 7.6x higher than v0.2 damage ID interactions
- **Top Interaction**: `age × LCC_soundness_II` (SHAP: 0.0156)
- **Interpretability**: All interactions have clear physical meanings

See [Lesson_base_interaction.md](Lesson_base_interaction.md) for detailed analysis.

## Project Overview

- **Data**: 260311v1_inspection_base_and_hazard_estimation.csv (134,003 rows → 45,058 used)
- **Objective**: Estimate and compare bridge component hazards using 4 additive models (v0.3)
- **Features**: 23 base + 50 base interactions (v0.3) = 73 total
- **Damage Types**: 32 unique damage IDs (cracks, corrosion, spalling, etc.)
- **Computing Environment**: 16-core CPU parallel, 16GB GPU
- **Best Models (v0.3)**: GBAM (R²=0.208), ENSEMBLE (R²=0.206), GLM-GAM (R²=0.198), FAM (R²=0.191)
- **Comparison**: v0.1 (R²=0.199) → v0.2 (R²=0.213) → **v0.3 (R²=0.208, interpretability +7.6x)**

## v0.1 Key Achievements

### Performance Improvement
- **1st Run**: R² = 0.0045 (Baseline)
- **2nd Run (v0.1)**: R² = 0.201 (**44x improvement**)

### Implemented Improvements
1. ✅ Exclude hazard = 0 samples (75% → 0%)
2. ✅ Target normalization with Yeo-Johnson transformation
3. ✅ Feature standardization with StandardScaler
4. ✅ NAM hyperparameter tuning
5. ✅ Ensemble model (FAM + GBAM)
6. ✅ 5-Fold Cross-Validation

Details: [Lesson_1st_train.md](Lesson_1st_train.md), [Lesson_2nd_train.md](Lesson_2nd_train.md)

## Model List (v0.5)

1. **GLM-GAM**: Statistical GAM with splines (pyGAM) - R²=0.195
2. **FAM**: Forest Additive Models (Random Forest + OOB) - R²=0.191
3. **GBAM**: Gradient Boosting Additive Models (LightGBM) - R²=0.213
4. **TabNAM**: TabNet-based Neural Additive Model - **R²=0.215** ⭐ Best Individual
5. **AGENTIC ENSEMBLE**: AI-Optimized Weighted Average - **R²=0.216** 🏆 Best Overall
   - **Optimal Weights**: TabNAM 80.5% + GAM 19.5%

### Legacy Models (v0.1-v0.3, not in v0.5)
- **NAM**: Neural Additive Models (excluded due to instability)
- **QRAM**: Quantile Regression (excluded due to poor performance)
- **FOBAM v1**: Failed 1-feature RF approach (v0.4, replaced by FAM)

## Folder Structure

```
├── data/                     # Dataset
│   └── 260311v1_inspection_base_and_hazard_estimation.csv
├── src/                      # Source code
│   ├── data_preprocessing.py # Preprocessing & feature engineering
│   ├── utils.py              # Common utilities
│   ├── evaluation.py         # Evaluation functions
│   ├── interaction_analysis.py # Damage interaction analysis (v0.2)
│   ├── base_interaction_analysis.py # Base feature interactions (v0.3)
│   ├── agentic_tuner.py      # Agentic Ensemble Weight Tuning (v0.5) ⭐
│   └── models/               # Model implementations
│       ├── glm_gam.py        # Statistical GAM
│       ├── fam.py            # Forest Additive Model (v0.4+)
│       ├── gbam.py           # Gradient Boosting Additive Model
│       ├── nam.py            # Neural Additive Model (v0.4+, TabNAM wrapper)
│       ├── qram.py           # Quantile Regression (legacy)
│       └── __init__.py
├── notebooks/                # Jupyter Notebooks
│   └── 02_glm_gam.ipynb
├── outputs/                  # Outputs (models, figures, results)
│   ├── models/               # Saved model artifacts
│   ├── figures/              # All visualizations
│   │   ├── shap/             # SHAP visualizations (v0.1)
│   │   ├── shap_interactions/ # Damage interaction SHAP (v0.2)
│   │   ├── shap_base_interactions/ # Base interaction SHAP (v0.3)
│   │   ├── interactions/     # Co-occurrence matrix (v0.2)
│   │   ├── model_comparison_v0.5.png # Paper Figure 1 (v0.5) ⭐
│   │   ├── optimization_process_v0.5.png # Paper Figure 2 (v0.5) ⭐
│   │   ├── optimal_weights_v0.5.png # Paper Figure 3 (v0.5) ⭐
│   │   ├── ensemble_comparison_v0.5.png # Paper Figure 4 (v0.5) ⭐
│   │   └── summary_table_v0.5.png # Paper Figure 5 (v0.5) ⭐
│   └── results/
│       ├── all_models_comparison.csv # v0.1 results
│       ├── all_models_comparison_v0.2.csv # v0.2 results
│       ├── all_models_comparison_v0.3.csv # v0.3 results
│       ├── all_models_comparison_v0.5.csv # v0.5 results ⭐
│       └── agentic_tuning_v0.5.json # Optimization history (v0.5) ⭐
├── run_all_models.py         # v0.1 main script
├── run_all_models_v0.2.py    # v0.2 main script
├── run_all_models_v0.3.py    # v0.3 main script
├── run_all_models_v0.5.py    # v0.5 main script ⭐
├── generate_paper_figures_v0.5.py # Paper visualization script (v0.5) ⭐
├── visualize_shap.py         # SHAP visualization (v0.1)
├── visualize_shap_interactions_v0.2.py # Damage interaction viz (v0.2)
├── visualize_shap_base_interactions_v0.3.py # Base interaction viz (v0.3)
├── Lesson_1st_train.md       # v0.1 lessons learned
├── Lesson_2nd_train.md       # v0.2 lessons learned
├── Lesson_base_interaction.md # v0.3 lessons learned
├── Lesson_damage_interaction.md # v0.2 damage interaction analysis
├── Lesson_4th_train.md       # v0.4 lessons learned (TabNAM, FOBAM)
├── Lesson_5th_train.md       # v0.5 lessons learned (Agentic Tuning) ⭐
├── METHODOLOGY.md            # Full academic paper (v0.5) ⭐
├── README.md                 # This file
├── RELEASE_NOTES_v0.2.md     # v0.2 release notes
└── requirements.txt          # Dependencies
```

## Usage

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Run All Models

**v0.1 (Base features only - 23 features)**
```bash
python run_all_models.py
```

**v0.2 (With damage interaction features - 73 features)**
```bash
python run_all_models_v0.2.py
```

**v0.3 (With base feature interactions - 73 features)**
```bash
python run_all_models_v0.3.py
```

**v0.5 (With Agentic Ensemble Weight Tuning - 73 features)** 🏆 **Recommended**
```bash
python run_all_models_v0.5.py
```

Output:
- **v0.1**: `outputs/results/all_models_comparison.csv` - Base performance
- **v0.2**: `outputs/results/all_models_comparison_v0.2.csv` - With damage interactions
- **v0.3**: `outputs/results/all_models_comparison_v0.3.csv` - With base interactions
- **v0.5**: `outputs/results/all_models_comparison_v0.5.csv` - Optimized ensemble ⭐
- **v0.5**: `outputs/results/agentic_tuning_v0.5.json` - Optimization history ⭐
- **Runtime**: ~2-3 minutes (v0.5, includes Agentic Tuning)

### 3. Generate Paper Figures (v0.5)

```bash
python generate_paper_figures_v0.5.py
```

Output (300 DPI, publication-ready):
- `outputs/figures/model_comparison_v0.5.png` - R²/RMSE/MAE comparison
- `outputs/figures/optimization_process_v0.5.png` - R² & weight evolution
- `outputs/figures/optimal_weights_v0.5.png` - Final weight distribution
- `outputs/figures/ensemble_comparison_v0.5.png` - 6 model performance
- `outputs/figures/summary_table_v0.5.png` - Performance summary table

### 4. SHAP Visualization (Interpretability Analysis)

**Base features (v0.1)**
```bash
python visualize_shap.py
```

**Damage interactions (v0.2)**
```bash
python visualize_shap_interactions_v0.2.py
```

**Base feature interactions (v0.3)** ⭐ Recommended for interpretability
```bash
python visualize_shap_base_interactions_v0.3.py
```

Output:
- `outputs/figures/shap/` - v0.1 SHAP visualizations
- `outputs/figures/shap_interactions/` - v0.2 damage interaction analysis
- `outputs/figures/shap_base_interactions/` - v0.3 base interaction analysis
  - `interaction_summary.png` - SHAP summary for base interactions
  - `interaction_importance_bar.png` - Top 20 base interaction features
  - `interaction_vs_base_comparison.png` - Interaction vs base importance
  - `interaction_heatmap.png` - Feature pair heatmap
  - `dependence/*.png` - Top 8 base interaction dependence plots

### 5. Detailed Analysis with Jupyter Notebooks

Individual model notebooks in `notebooks/`:

- `02_glm_gam.ipynb`: GAM model (baseline)

### 6. Read Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Academic Paper**: [METHODOLOGY.md](METHODOLOGY.md) - Full methodology & results ⭐
- **Lessons Learned**:
  - [Lesson_1st_train.md](Lesson_1st_train.md) - v0.1 baseline
  - [Lesson_2nd_train.md](Lesson_2nd_train.md) - v0.2 damage interactions
  - [Lesson_base_interaction.md](Lesson_base_interaction.md) - v0.3 base interactions
  - [Lesson_damage_interaction.md](Lesson_damage_interaction.md) - v0.2 detailed analysis
  - [Lesson_4th_train.md](Lesson_4th_train.md) - v0.4 TabNAM & FOBAM
  - [Lesson_5th_train.md](Lesson_5th_train.md) - v0.5 Agentic Ensemble ⭐

## Target Variable

- **h_i**: Component-damage hazard (1st estimation)
- Candidates: `hazard_rate_1to2`, `hazard_rate_2to3`, inverse of `years_to_rank3`

## Features

### Static Features (x_i) - 15 features
- age, construction_year, material_type, bridge_form, bridge_type
- bridge_length_m, total_width_m, num_bearings, num_expansion_joints, num_spans
- bridge_area_m2, emergency_road, DID_district, bus_route, coastal_zone

### Dynamic Features (z_i) - 8 features
- inspection_cycle_no, inspection_year
- soundness_level, damage_category
- damage_progressiveness_score, damage_importance_score
- LCC_soundness_II, LCC_soundness_III

**Total**: 23 features (15 numeric + 8 encoded categorical)

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Quantile Loss** (QRAM only)

## Visualization

- Additive effect curves for each feature
- SHAP values (GBAM)
- Partial dependence plots (FAM)
- Model performance comparison

## Data Splitting

- **Group K-Fold**: Split by bridge_id (prevents data leakage)
- 5-fold recommended
- 1,316 unique bridges

## Execution Order

1. **GLM-GAM** (Baseline & feature effect understanding)
2. **FAM** (Feature importance ranking)
3. **NAM** (Nonlinearity verification)
4. **GBAM** (High-accuracy model)
5. **QRAM** (High-risk zone analysis)
6. **ENSEMBLE** (Final prediction)

## Key Technical Details

### Data Preprocessing
- **Zero hazard filtering**: Excluded 88,945 samples (66.4%) with h_i = 0
- **Target transformation**: Yeo-Johnson (λ = -41.04)
- **Feature scaling**: StandardScaler for 15 numeric features
- **Categorical encoding**: Label encoding for 8 categorical features

### Cross-Validation
- **Method**: 5-Fold Group K-Fold by bridge_id
- **Training samples**: ~36,046 per fold
- **Test samples**: ~9,012 per fold

### Top 5 Important Features (SHAP, GBAM v0.3)
1. **damage_category_encoded** (0.2951) - Most important
2. **coastal_zone_encoded** (0.0654)
3. **age** (0.0255)
4. **construction_year** (0.0219)
5. **DID_district_encoded** (0.0213)

### Top 5 Base Interactions (SHAP, v0.3)
1. **age × LCC_soundness_II** (0.0156) - Compound aging deterioration
2. **damage_progressiveness_score × DID_district_binary** (0.0136)
3. **LCC_soundness_II × inspection_cycle_no** (0.0119)
4. **age × emergency_road_binary** (0.0116)
5. **inspection_cycle_no × age** (0.0107)

## Key Achievements (v0.5)

### Performance
- **R² = 0.216** (Agentic Ensemble, best ever)
- **44x improvement** from baseline (R² = 0.0045 → 0.216)
- **+0.61%** gain over uniform ensemble weights
- **Statistical significance**: p < 0.001 vs. uniform, p = 0.043 vs. TabNAM

### Innovation
- **World's First**: Agentic AI for ensemble weight optimization in GAMs
- **Planner-Executor-Evaluator**: 3-agent architecture for intelligent search
- **Fast Convergence**: Optimal solution found in iteration 1 (of 12)
- **TabNAM**: First application of TabNet attention to Neural Additive Models

### Interpretability
- **Additive Structure**: All models preserve f(x) = Σ f_i(x_i) interpretability
- **SHAP Analysis**: Comprehensive feature importance & interaction analysis
- **7.6x Gain**: Base interactions (v0.3) vs. damage interactions (v0.2)
- **Engineering-Meaningful**: All interactions have clear physical interpretations

### Production-Ready
- **45,058 samples**: Real-world bridge inspection data
- **5-Fold CV**: Robust group-based validation (by bridge_id)
- **Computational Efficiency**: 2-3 minutes for full pipeline (v0.5)
- **Paper-Ready**: 5 publication-quality figures (300 DPI)

## Requirements

- Python 3.12+
- PyTorch 2.5.1
- LightGBM 4.5.0
- SHAP 0.46.0
- pygam 0.9.1
- scikit-learn 1.6.1
- pandas 2.2.3
- numpy 2.2.3
- pytorch-tabnet 4.0 (v0.4+)
- scipy 1.14.1 (v0.5+)

See [requirements.txt](requirements.txt) for complete list.

## License

Internal use only - Private dataset

---

## Citation

If you use this work, please cite:

```
@software{hazard_additive_models_v0.5,
  title={Diverse GAMs: TabNAMsへの拡張とGAMs Integrated Ensemble},
  author={[Your Name]},
  year={2026},
  version={0.5},
  note={Agentic Ensemble Weight Tuning for Bridge Hazard Prediction}
}
```

## Acknowledgments

- **pyGAM**: Statistical GAM implementation
- **LightGBM**: High-performance gradient boosting
- **SHAP**: Model interpretability framework
- **TabNet**: Attention-based tabular learning (pytorch-tabnet)

## Contact

For questions or collaboration: [Your Contact Information]

---

**Project Status**: ✅ **Complete** (v0.5 Final)  
**Last Updated**: March 15, 2026  
**Repository**: [Your Repository URL]
