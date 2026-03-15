# Quick Start Guide

**Version**: v0.1

## Setup

### 1. Install Required Packages

```bash
cd I:\ACT2025.5.26-2030\MVP\hazard_additive_mdls
pip install -r requirements.txt
```

### 2. Verify Data

Ensure the data file `data/260311v1_inspection_base_and_hazard_estimation.csv` exists.

## Execution Methods

### Method 1: Run All Models (Recommended)

```bash
python run_all_models.py
```

This script executes the following in sequence:
1. Data preprocessing (zero exclusion, Yeo-Johnson transform, feature scaling)
2. GLM-GAM (baseline)
3. FAM (feature importance analysis) - **Best Model R²=0.201**
4. NAM (Neural Additive Model)
5. GBAM (high-accuracy model) - R²=0.199
6. QRAM (quantile regression)
7. ENSEMBLE (FAM + GBAM)

Estimated runtime: ~5-10 minutes (5-fold CV)

### Method 2: SHAP Visualization (Interpretability Analysis)

```bash
python visualize_shap.py
```

Computes SHAP values using the GBAM model and generates:
- **Summary Plot**: Overall feature importance
- **Bar Plot**: Mean absolute SHAP value ranking
- **Dependence Plots**: Individual feature effect curves (top 5 features)
- **Waterfall Plot**: Detailed explanation of a single prediction

Estimated runtime: ~2-5 minutes

### Method 3: Run Individual Models

#### Run GLM-GAM Only

```python
from src.data_preprocessing import load_and_preprocess_data
from src.models.glm_gam import GLMGAMModel
from src.evaluation import evaluate_regression_model

# Load data
data = load_and_preprocess_data()
X = data['X']
y = data['y']

# Train model
model = GLMGAMModel(
    continuous_features=['age', 'bridge_length_m', 'total_width_m'],
    n_splines=10
)
model.fit(X, y)

# Predict
y_pred = model.predict(X)
```

#### For detailed implementations, see Jupyter Notebooks

- `notebooks/02_glm_gam.ipynb` - Detailed GLM-GAM execution
- Other notebooks follow similar structure

## Output Files

After execution, results are saved in the following directories:

```
outputs/
├── models/              # Trained models (.pkl)
│   └── glm_gam_final.pkl
├── figures/             # Visualizations (.png)
│   ├── glm_gam_continuous_contributions.png
│   └── shap/           # SHAP visualizations
│       ├── shap_summary.png
│       ├── shap_importance_bar.png
│       ├── shap_waterfall_sample.png
│       └── dependence/ # Individual feature effects
└── results/             # Evaluation results (.csv, .pkl)
    ├── glm_gam_cv_results.csv
    └── all_models_comparison.csv
```

## Checking Results

### Model Comparison Results

```python
import pandas as pd

# Load comparison results for all models
df_comparison = pd.read_csv('outputs/results/all_models_comparison.csv', encoding='utf-8-sig')
print(df_comparison)
```

### Load Trained Model

```python
import joblib

# Load model
model = joblib.load('outputs/models/glm_gam_final.pkl')

# Predict
y_pred = model.predict(X_new)
```

## Model-Specific Features

### 1. GLM-GAM
- **Feature**: Smooth nonlinear relationships via spline functions
- **Use Case**: Baseline, intuitive understanding of feature effects
- **Output**: Contribution curves for each feature

### 2. FAM (Random Forest)
- **Feature**: Variable importance and partial dependence plots
- **Use Case**: Feature selection and importance ranking
- **Output**: Feature importance ranking, partial dependence plots

### 3. NAM (Neural)
- **Feature**: Neural network-based additive model
- **Use Case**: Learning complex nonlinear relationships
- **Output**: Learned contribution functions for each feature

### 4. GBAM (LightGBM)
- **Feature**: Fast, high-accuracy production model
- **Use Case**: Best accuracy predictions, SHAP-based interpretation
- **Output**: SHAP value analysis, feature importance

### 5. QRAM (Quantile Regression)
- **Feature**: Quantile-specific predictions
- **Use Case**: High-risk zone estimation (90th percentile, etc.)
- **Output**: Multi-quantile predictions, prediction intervals

### 6. ENSEMBLE
- **Feature**: Weighted average of FAM (0.5) + GBAM (0.5)
- **Use Case**: Robust predictions combining multiple models
- **Output**: Best overall R² = 0.209

## Customization

### Change Target Variable

```python
data = load_and_preprocess_data(
    target_type='hazard_rate_2to3'  # or 'years_to_rank3'
)
```

### Adjust Hyperparameters

Modify `__init__` parameters of each model class.

Example (GBAM):
```python
model = GBAMModel(
    n_estimators=1000,     # Increase number of trees
    learning_rate=0.03,    # Lower learning rate
    num_leaves=50          # Increase leaf nodes
)
```

## Troubleshooting

### Out of Memory Error

- Reduce batch size (NAM, GBAM)
- Sample the data

```python
# Reduce sample size
X_sample = X.sample(n=50000, random_state=42)
y_sample = y[X_sample.index]
```

### GPU Error (NAM)

```python
trainer = NAMTrainer(
    device='cpu'  # Explicitly specify CPU
)
```

### Convergence Issues (QRAM)

```python
model = QRAMModel(
    max_iter=2000,   # Increase iterations
    p_tol=1e-5       # Relax tolerance
)
```

## Recommended Execution Order

1. **GLM-GAM**: Create baseline, understand feature effects
2. **FAM**: Identify important features
3. **GBAM**: Build high-accuracy model (production-ready)
4. **NAM**: Verify nonlinearity
5. **QRAM**: High-risk analysis
6. **ENSEMBLE**: Final predictions with best performance

## Next Steps

- Improve feature engineering
- Hyperparameter tuning with Optuna/GridSearch
- Apply advanced ensemble methods
- Implement temporal validation

## Support

For issues, refer to the project's README.md or contact the development team.
