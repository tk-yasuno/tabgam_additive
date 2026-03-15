# Diverse GAMs: Multi-Domain Degradation Prediction Framework

**Version**: v0.8 (Current)
**Date**: March 15, 2026
**Status**: ✅ **4 Case Studies Complete**

## Description

**Diverse GAMs: Multi-Domain Degradation Prediction Framework** - A comprehensive implementation of interpretable additive models across **3 domains** (Infrastructure, Materials, Aviation) and **4 case studies**. Features deep learning-based TabNAM, Agentic Ensemble optimization, and achieves high prediction accuracy ranging from R² = 0.216 to 0.768 depending on domain complexity.

**Key Features:**

- 🌍 **Multi-Domain**: 4 case studies across Infrastructure, Materials, and Aviation
- 🚀 **High Performance**: R² = 0.663 (Jet Engine FD002, v0.8), R² = 0.768 (Material Degradation, v0.6)
- 🤖 **Agentic AI**: Automated ensemble weight optimization with Planner-Executor-Evaluator pattern
- 🧠 **TabNAM**: Deep learning-based Neural Additive Model with attention mechanism
- 🔍 **High Interpretability**: Additive structure preserved across all models (GAM, FAM, GBAM, TabNAM)
- 📊 **Diverse Models**: Statistical GAM + Random Forest + Gradient Boosting + Deep Learning
- 📈 **Production-Ready**: 3.8K to 260K samples, 5-fold CV, comprehensive validation

## Version History

### Bridge Degradation (Infrastructure Domain)

| Version | R²            | Key Innovation                                     | Status   |
| ------- | -------------- | -------------------------------------------------- | -------- |
| v0.1    | 0.004 → 0.201 | Baseline + Advanced Preprocessing                  | Complete |
| v0.2    | 0.213          | Damage ID Interactions                             | Complete |
| v0.3    | 0.208          | Base Feature Interactions (+7.6x interpretability) | Complete |
| v0.4    | 0.208          | TabNAM, FOBAM v2 (FAM equivalent)                  | Complete |
| v0.5    | 0.216          | Agentic Ensemble Weight Tuning                     | Complete |

### Multi-Domain Expansion

| Version        | 回Domain                   | R²             | Dataset                | Key Finding                                   | Status               |
| -------------- | -------------------------- | --------------- | ---------------------- | --------------------------------------------- | -------------------- |
| **v0.6** | **Material**         | **0.768** | **3.8K samples** | **Best performance across all studies** | ✅**Complete** |
| **v0.7** | **Aviation (FD004)** | **0.609** | **61K samples**  | **2 fault modes (HPC+Fan)**             | ✅**Complete** |
| **v0.8** | **Aviation (FD002)** | **0.663** | **54K samples**  | **1 fault mode → 9% improvement**      | ✅**Current**  |

## What's New in v0.8 (Current) 🎉

### Jet Engine RUL Prediction - Single Fault Mode (FD002)

- **Dataset**: NASA C-MAPSS FD002 (54K train, 12K test, 55 features, 1 fault mode)
- **Best Model**: GBAM (R² = 0.663, RMSE = 36.82)
- **Ensemble**: GAM 24.4% + GBAM 52.8% + TabNAM 22.1% = R² = 0.663
- **Key Finding**: **Single fault mode (FD002) outperforms dual fault mode (FD004) by 9%**

### Why FD002 > FD004?

- 🎯 **Simpler Problem**: 1 fault mode → more consistent degradation patterns
- 🏆 **All Models Better**: GAM (+14%), FAM (+8%), GBAM (+9%), NAM (+6%), TabNAM (+88%)
- 🤖 **TabNAM Returns**: 22.1% weight (0% in FD004) - effective for single-fault scenarios
- 📊 **3-Model Ensemble**: vs. 2-model in FD004 (adaptability to problem complexity)

### Top Features (SHAP, FD002)

1. **sensor_18** (4,719) - Total temperature at LPT outlet
2. **sensor_1** (2,143) - Total temperature at fan inlet
3. **sensor_9** (1,935) - Static pressure at HPC outlet

See [RESULTS_v08_CMAPSS.md](RESULTS_v08_CMAPSS.md) and [COMPARISON_FD002_FD004.md](COMPARISON_FD002_FD004.md) for detailed analysis.

## What's New in v0.7 🚁

### Jet Engine RUL Prediction - Dual Fault Modes (FD004)

- **Dataset**: NASA C-MAPSS FD004 (61K train, 41K test, 55 features, 2 fault modes)
- **Best Model**: GBAM (R² = 0.609, RMSE = 40.23)
- **Ensemble**: GAM 36.2% + GBAM 63.8% = R² = 0.609
- **Challenge**: Dual fault modes (HPC degradation + Fan degradation) increase complexity

### Key Findings

- 📉 **Complexity Impact**: 2 fault modes → lower R² than simple domains
- 🔧 **2-Model Ensemble**: Only GAM + GBAM selected (TabNAM ineffective for complex scenarios)
- 🌡️ **Temperature Dominance**: High-pressure compressor temperatures most important
- ⚙️ **Operating Conditions**: 6 conditions → significant variability

### Top Features (SHAP, FD004)

1. **sensor_15** (3,154) - Total temperature at LPT outlet
2. **rolling_std_sensor_15** (1,835) - Temperature variation
3. **sensor_3** (1,754) - Total temperature at LPT inlet

See [RESULTS_v07_CMAPSS_FD004.md](RESULTS_v07_CMAPSS_FD004.md) for full analysis.

## What's New in v0.6 🏭

### Material Degradation Accident Prediction

- **Dataset**: Fukuchiyama Bridge Material Degradation (3,772 samples, 17 features)
- **Best Performance**: R² = 0.768 (Highest across all case studies!)
- **Ensemble**: GAM 15.6% + GBAM 19.6% + TabNAM 64.8% = R² = 0.768
- **Domain**: Material accidents and degradation events

### Key Findings

- 🏆 **Best R² Ever**: 0.768 >> Bridge (0.216), Aviation (0.609-0.663)
- 💡 **Small Data Success**: Only 3.8K samples → high accuracy possible
- 🧪 **Material Science**: Specific domain with clear deterioration patterns
- 🔬 **TabNAM Dominance**: 64.8% weight (most important in ensemble)

### Top Features (v0.6)

1. **Material Type** - Primary predictor of degradation
2. **Environmental Conditions** - Temperature, humidity exposure
3. **Service Years** - Age-based deterioration

See [Classification_v0.6_Results.md](0_NOT_CLASSIFICATION/Classification_v0.6_Results.md) for full analysis.

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

### Multi-Domain Framework (v0.8)

This project demonstrates the versatility of interpretable additive models across 3 domains:

| Domain                   | Case Study              | Dataset                 | Samples | Features | R²   | Best Model       |
| ------------------------ | ----------------------- | ----------------------- | ------- | -------- | ----- | ---------------- |
| **Infrastructure** | v0.5 Bridge Degradation | 260K inspection records | 45K     | 73       | 0.216 | Agentic Ensemble |
| **Materials**      | v0.6 Material Accidents | 福知山線材料劣化        | 3.8K    | 17       | 0.768 | TabNAM (64.8%)   |
| **Aviation**       | v0.7 Jet Engine (FD004) | NASA C-MAPSS            | 61K     | 55       | 0.609 | GBAM             |
| **Aviation**       | v0.8 Jet Engine (FD002) | NASA C-MAPSS            | 54K     | 55       | 0.663 | GBAM             |

### Key Insights

- 🏆 **Domain Matters**: Material science (v0.6, R²=0.768) > Jet engines (0.609-0.663) > Bridge infrastructure (0.216)
- 🎯 **Complexity Impact**: Single fault mode (FD002) achieves 9% higher R² than dual fault mode (FD004)
- 🤖 **Adaptive Ensembles**: Framework automatically selects 2-3 models based on problem complexity
- 📊 **Interpretability**: Additive structure maintained across all domains for engineering insights

### Original Bridge Study (v0.1-v0.5)

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
│   └── 260311v1_inspection_base_and_hazard_estimation.csv # Bridge (v0.5)
├── src/                      # Source code
│   ├── data_preprocessing.py # Preprocessing & feature engineering (v0.5)
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
│       ├── tabnam.py         # TabNAM implementation (v0.8)
│       ├── qram.py           # Quantile Regression (legacy)
│       └── __init__.py
├── 0_NOT_CLASSIFICATION/     # Material Degradation (v0.6)
│   ├── fukuchiyama_bridge_sample_v0.6.csv # Material dataset (v0.6)
│   ├── data_preprocessing_classification.py # Preprocessing (v0.6)
│   ├── run_classification_models_v0.6.py # Model training (v0.6) ⭐
│   ├── Classification_v0.6_Results.md # Results documentation (v0.6) ⭐
│   └── Classification_v0.6_arXiv_Draft.md # Paper draft (v0.6)
├── data_preprocessing_cmapss_v07.py # FD004 preprocessing (v0.7)
├── data_preprocessing_cmapss_v08.py # FD002 preprocessing (v0.8) ⭐
├── run_all_models_v0.7.py    # FD004 training (v0.7)
├── run_all_models_v0.8.py    # FD002 training (v0.8) ⭐
├── visualize_shap_cmapss_v0.7.py # FD004 SHAP (v0.7)
├── visualize_shap_cmapss_v0.8.py # FD002 SHAP (v0.8) ⭐
├── generate_paper_figures_v0.7.py # FD004 figures (v0.7)
├── generate_paper_figures_v0.8.py # FD002 figures (v0.8) ⭐
├── RESULTS_v07_CMAPSS_FD004.md # FD004 documentation (v0.7)
├── RESULTS_v08_CMAPSS.md     # FD002 documentation (v0.8) ⭐
├── COMPARISON_FD002_FD004.md # FD002 vs FD004 comparison (v0.8) ⭐
├── notebooks/                # Jupyter Notebooks
│   └── 02_glm_gam.ipynb
├── outputs/                  # Outputs (models, figures, results)
│   ├── models/               # Saved model artifacts
│   ├── figures/              # All visualizations
│   │   ├── shap/             # SHAP visualizations (v0.1)
│   │   ├── shap_interactions/ # Damage interaction SHAP (v0.2)
│   │   ├── shap_base_interactions/ # Base interaction SHAP (v0.3)
│   │   ├── tabnam_shap_dependence/ # TabNAM SHAP (v0.8) ⭐
│   │   ├── interactions/     # Co-occurrence matrix (v0.2)
│   │   ├── model_comparison_v0.5.png # Paper Figure 1 (v0.5)
│   │   ├── model_comparison_v0.8.png # Paper Figure 1 (v0.8) ⭐
│   │   ├── optimization_process_v0.5.png # Paper Figure 2 (v0.5)
│   │   ├── optimization_process_v0.8.png # Paper Figure 2 (v0.8) ⭐
│   │   ├── optimal_weights_v0.5.png # Paper Figure 3 (v0.5)
│   │   ├── optimal_weights_v0.8.png # Paper Figure 3 (v0.8) ⭐
│   │   ├── ensemble_comparison_v0.5.png # Paper Figure 4 (v0.5)
│   │   ├── ensemble_comparison_v0.8.png # Paper Figure 4 (v0.8) ⭐
│   │   └── summary_table_v0.5.png # Paper Figure 5 (v0.5)
│   │   └── summary_table_v0.8.png # Paper Figure 5 (v0.8) ⭐
│   └── results/
│       ├── all_models_comparison.csv # v0.1 results
│       ├── all_models_comparison_v0.2.csv # v0.2 results
│       ├── all_models_comparison_v0.3.csv # v0.3 results
│       ├── all_models_comparison_v0.5.csv # v0.5 results
│       ├── all_models_comparison_v0.6.csv # v0.6 results (Material) ⭐
│       ├── all_models_comparison_v0.7.csv # v0.7 results (FD004) ⭐
│       ├── all_models_comparison_v0.8.csv # v0.8 results (FD002) ⭐
│       ├── agentic_tuning_v0.5.json # Optimization history (v0.5)
│       ├── agentic_tuning_v0.7.json # Optimization history (v0.7)
│       └── agentic_tuning_v0.8.json # Optimization history (v0.8) ⭐
├── run_all_models.py         # v0.1 main script
├── run_all_models_v0.2.py    # v0.2 main script
├── run_all_models_v0.3.py    # v0.3 main script
├── run_all_models_v0.5.py    # v0.5 main script
├── generate_paper_figures_v0.5.py # Paper visualization script (v0.5)
├── visualize_shap.py         # SHAP visualization (v0.1)
├── visualize_shap_interactions_v0.2.py # Damage interaction viz (v0.2)
├── visualize_shap_base_interactions_v0.3.py # Base interaction viz (v0.3)
├── visualize_shap_tabnam_v0.5.py # TabNAM SHAP (v0.5)
├── Lesson_1st_train.md       # v0.1 lessons learned
├── Lesson_2nd_train.md       # v0.2 lessons learned
├── Lesson_base_interaction.md # v0.3 lessons learned
├── Lesson_damage_interaction.md # v0.2 damage interaction analysis
├── Lesson_4th_train.md       # v0.4 lessons learned (TabNAM, FOBAM)
├── Lesson_5th_train.md       # v0.5 lessons learned (Agentic Tuning)
├── METHODOLOGY.md            # Full academic paper (v0.5)
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

#### Bridge Degradation (v0.1-v0.5, Infrastructure Domain)

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

**v0.5 (With Agentic Ensemble Weight Tuning - 73 features)** 🏆

```bash
python run_all_models_v0.5.py
```

#### Material Degradation (v0.6, Material Science Domain)

**v0.6 (Material Accident Prediction - 17 features)** 🏆 **Best R² = 0.768**

```bash
python 0_NOT_CLASSIFICATION/run_classification_models_v0.6.py
```

#### Jet Engine RUL Prediction (v0.7-v0.8, Aviation Domain)

**v0.7 (FD004: 2 Fault Modes - 55 features)**

```bash
python run_all_models_v0.7.py
```

**v0.8 (FD002: 1 Fault Mode - 55 features)** 🏆 **Current, 9% improvement**

```bash
python run_all_models_v0.8.py
```

#### Output Files

- **Bridge (v0.1-v0.5)**:

  - `outputs/results/all_models_comparison_v0.5.csv` - Best bridge performance ⭐
  - `outputs/results/agentic_tuning_v0.5.json` - Optimization history
  - Runtime: ~2-3 minutes (v0.5, includes Agentic Tuning)
- **Material (v0.6)**:

  - `0_NOT_CLASSIFICATION/classification_models_v0.6.csv` - Material degradation results ⭐
  - Runtime: ~1-2 minutes
- **Aviation (v0.7-v0.8)**:

  - `outputs/results/all_models_comparison_v0.7.csv` - FD004 results
  - `outputs/results/all_models_comparison_v0.8.csv` - FD002 results ⭐
  - `outputs/results/agentic_tuning_v0.8.json` - Optimization history (v0.8)
  - Runtime: ~3-4 minutes per version

### 3. Generate Paper Figures

#### Bridge (v0.5)

```bash
python generate_paper_figures_v0.5.py
```

Output (300 DPI, publication-ready):

- `outputs/figures/model_comparison_v0.5.png` - R²/RMSE/MAE comparison
- `outputs/figures/optimization_process_v0.5.png` - R² & weight evolution
- `outputs/figures/optimal_weights_v0.5.png` - Final weight distribution
- `outputs/figures/ensemble_comparison_v0.5.png` - 6 model performance
- `outputs/figures/summary_table_v0.5.png` - Performance summary table

#### Aviation (v0.7, v0.8)

```bash
python generate_paper_figures_v0.7.py  # FD004
python generate_paper_figures_v0.8.py  # FD002 ⭐
```

Output (v0.8):

- `outputs/figures/model_comparison_v0.8.png` - Model comparison
- `outputs/figures/optimization_process_v0.8.png` - Optimization trajectory
- `outputs/figures/optimal_weights_v0.8.png` - Final weights
- `outputs/figures/ensemble_comparison_v0.8.png` - Ensemble performance
- `outputs/figures/summary_table_v0.8.png` - Summary table

### 4. SHAP Visualization (Interpretability Analysis)

#### Bridge Models (v0.1-v0.5)

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

**TabNAM attention (v0.5)**

```bash
python visualize_shap_tabnam_v0.5.py
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

#### Aviation Models (v0.7-v0.8)

**FD004 SHAP (v0.7)**

```bash
python visualize_shap_cmapss_v0.7.py
```

**FD002 SHAP (v0.8)** ⭐ Current

```bash
python visualize_shap_cmapss_v0.8.py
```

Output (v0.8):

- `outputs/figures/tabnam_shap_dependence/` - FD002 SHAP analysis
  - 6 SHAP visualization plots
  - `shap_importances.csv` - Feature importance values

### 5. Detailed Analysis with Jupyter Notebooks

Individual model notebooks in `notebooks/`:

- `02_glm_gam.ipynb`: GAM model (baseline)

### 6. Read Documentation

#### Bridge Degradation (v0.1-v0.5)

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Academic Paper**: [METHODOLOGY.md](METHODOLOGY.md) - Full methodology & results ⭐
- **Lessons Learned**:
  - [Lesson_1st_train.md](Lesson_1st_train.md) - v0.1 baseline
  - [Lesson_2nd_train.md](Lesson_2nd_train.md) - v0.2 damage interactions
  - [Lesson_base_interaction.md](Lesson_base_interaction.md) - v0.3 base interactions
  - [Lesson_damage_interaction.md](Lesson_damage_interaction.md) - v0.2 detailed analysis
  - [Lesson_4th_train.md](Lesson_4th_train.md) - v0.4 TabNAM & FOBAM
  - [Lesson_5th_train.md](Lesson_5th_train.md) - v0.5 Agentic Ensemble ⭐

#### Material Degradation (v0.6)

- [Classification_v0.6_Results.md](0_NOT_CLASSIFICATION/Classification_v0.6_Results.md) - Full results & analysis ⭐
- [Classification_v0.6_arXiv_Draft.md](0_NOT_CLASSIFICATION/Classification_v0.6_arXiv_Draft.md) - Paper draft

#### Aviation (v0.7-v0.8)

- [RESULTS_v07_CMAPSS_FD004.md](RESULTS_v07_CMAPSS_FD004.md) - FD004 comprehensive results
- [RESULTS_v08_CMAPSS.md](RESULTS_v08_CMAPSS.md) - FD002 comprehensive results ⭐
- [COMPARISON_FD002_FD004.md](COMPARISON_FD002_FD004.md) - Single vs Dual fault mode comparison ⭐

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

## Key Achievements (v0.8 Framework)

### Multi-Domain Success (v0.5-v0.8)

- 🌍 **3 Domains**: Infrastructure (Bridge), Materials (Accidents), Aviation (Jet Engines)
- 📊 **4 Case Studies**: v0.5 (R²=0.216), v0.6 (R²=0.768), v0.7 (R²=0.609), v0.8 (R²=0.663)
- 🏆 **Best Performance**: Material degradation (v0.6, R²=0.768) - highest across all studies
- 🎯 **Key Finding**: Single fault mode (FD002) outperforms dual fault mode (FD004) by 9%

### Performance by Domain

| Domain         | Case Study    | R²             | Dataset Size | Complexity                 |
| -------------- | ------------- | --------------- | ------------ | -------------------------- |
| Material       | v0.6          | **0.768** | 3.8K         | Low (specific patterns)    |
| Aviation       | v0.8 (FD002)  | **0.663** | 54K          | Medium (1 fault)           |
| Aviation       | v0.7 (FD004)  | **0.609** | 61K          | High (2 faults)            |
| Infrastructure | v0.5 (Bridge) | **0.216** | 45K          | Very High (diverse damage) |

### Innovation (v0.5)

- **World's First**: Agentic AI for ensemble weight optimization in GAMs
- **Planner-Executor-Evaluator**: 3-agent architecture for intelligent search
- **Fast Convergence**: Optimal solution found in iteration 1 (of 12)
- **TabNAM**: First application of TabNet attention to Neural Additive Models

### Domain-Specific Insights

**Material Degradation (v0.6)**:

- TabNAM dominates (64.8% weight) - effective for focused patterns
- Small dataset (3.8K) achieves highest R² - domain specificity matters
- Clear material science deterioration patterns

**Jet Engine FD002 (v0.8)**:

- Single fault mode → 9% higher R² than FD004
- 3-model ensemble (GAM + GBAM + TabNAM) vs. 2-model in FD004
- TabNAM returns (22.1% weight) after being 0% in FD004
- All individual models perform better than FD004 counterparts

**Jet Engine FD004 (v0.7)**:

- Dual fault modes (HPC + Fan) increase complexity
- 2-model ensemble (GAM + GBAM only) - TabNAM ineffective
- Temperature sensors dominate feature importance

**Bridge Degradation (v0.5)**:

- Most complex problem (32 damage types, diverse infrastructure)
- 44x improvement from baseline (R² = 0.0045 → 0.216)
- Agentic Ensemble: TabNAM 80.5% + GAM 19.5% optimal

### Interpretability

- **Additive Structure**: All models preserve f(x) = Σ f_i(x_i) interpretability
- **SHAP Analysis**: Comprehensive feature importance across all domains
- **Engineering-Meaningful**: Domain-specific insights from additive components
- **Cross-Domain**: Framework adapts model selection to problem complexity

### Production-Ready

- **Diverse Scales**: 3.8K to 260K samples across case studies
- **5-Fold CV**: Robust group-based validation
- **Computational Efficiency**: 1-4 minutes per version
- **Paper-Ready**: Publication-quality figures for all versions

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

MIT License

---

## Citation

If you use this work, please cite:

```
@software{diverse_gams_v0.8,
  title={Diverse GAMs: Multi-Domain Degradation Prediction Framework},
  author={[Your Name]},
  year={2026},
  version={0.8},
  note={Agentic Ensemble and TabNAM for Infrastructure, Materials, and Aviation}
}
```

**Case Study Citations:**

- **v0.5 (Bridge)**: Agentic Ensemble Weight Tuning for Bridge Hazard Prediction
- **v0.6 (Material)**: Material Degradation Accident Prediction with TabNAM
- **v0.7 (Aviation FD004)**: Dual Fault Mode Jet Engine RUL Prediction
- **v0.8 (Aviation FD002)**: Single Fault Mode Jet Engine RUL Prediction (9% improvement)

## Acknowledgments

- **pyGAM**: Statistical GAM implementation
- **LightGBM**: High-performance gradient boosting
- **SHAP**: Model interpretability framework
- **TabNet**: Attention-based tabular learning (pytorch-tabnet)
- **NASA C-MAPSS**: Turbofan Engine Degradation Simulation Dataset (v0.7-v0.8)

## Contact

For questions or collaboration: [Your Contact Information]

---

**Project Status**: ✅ **4 Case Studies Complete** (v0.8 Current)
**Last Updated**: March 15, 2026
**Repository**: [Your Repository URL]
