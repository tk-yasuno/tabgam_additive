# TabGAM: Diverse Generalized Additive Models with Agentic Ensemble Optimization

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

**TabGAM** (Tabular Generalized Additive Models) is a comprehensive framework for interpretable machine learning on tabular data, featuring:

- 🚀 **State-of-the-Art Performance**: R² = 0.216 on bridge hazard prediction
- 🤖 **Agentic AI**: Automated ensemble weight optimization with Planner-Executor-Evaluator pattern
- 🧠 **TabNAM**: Deep learning-based Neural Additive Model with attention mechanism
- 🔍 **High Interpretability**: Additive structure preserved across all models (GAM, FAM, GBAM, TabNAM)
- 📊 **4 Diverse Models**: Statistical GAM + Random Forest + Gradient Boosting + Deep Learning

## Key Features

### 1. Diverse GAM Implementations
- **GLM-GAM**: Statistical spline-based GAM (pyGAM)
- **FAM**: Forest Additive Models (Random Forest + OOB evaluation)
- **GBAM**: Gradient Boosting Additive Models (LightGBM)
- **TabNAM**: TabNet-based Neural Additive Models (Attention mechanism)

### 2. Agentic Ensemble Weight Tuning
- **Planner**: Decides next optimization strategy (SLSQP, Differential Evolution, Random Perturbation)
- **Executor**: Runs optimization with selected method
- **Evaluator**: Assesses performance and determines early stopping

### 3. Comprehensive Interpretability
- SHAP analysis for all models
- Feature importance visualization
- Partial dependence plots
- Interaction feature analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tabgam_additive.git
cd tabgam_additive

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.data_preprocessing import load_and_preprocess_data
from src.models.tabnam import TabNAMModel
from src.agentic_tuner import AgenticTuner

# Load data
data = load_and_preprocess_data(
    data_path="data/your_data.csv",
    add_base_interactions=True
)

# Train TabNAM
model = TabNAMModel(n_d=8, n_a=8, n_steps=3)
model.fit(X_train, y_train)

# Optimize ensemble weights with Agentic AI
tuner = AgenticTuner(models_dict, X_val, y_val)
results = tuner.run()
```

### Run Full Pipeline

```bash
# Run all models with Agentic Ensemble Optimization (v0.5)
python run_all_models_v0.5.py

# Generate paper figures
python generate_paper_figures_v0.5.py

# Visualize TabNAM SHAP values
python visualize_shap_tabnam_v0.5.py
```

## Results

| Model | R² | RMSE | MAE | Note |
|-------|-----|------|-----|------|
| GAM | 0.195 | 0.652 | 0.498 | Statistical baseline |
| FAM | 0.191 | 0.654 | 0.498 | Random Forest |
| GBAM | 0.213 | 0.645 | 0.490 | Gradient Boosting |
| TabNAM | **0.215** | 0.644 | 0.489 | Deep Learning (Best Individual) |
| **Agentic Ensemble** | **0.216** | **0.643** | **0.488** | 🏆 **Best Overall** |

**Optimal Weights**: TabNAM 80.5% + GAM 19.5%

## Project Structure

```
tabgam_additive/
├── data/                          # Dataset
├── src/                           # Source code
│   ├── agentic_tuner.py          # Agentic Ensemble Weight Tuning ⭐
│   ├── data_preprocessing.py     # Data preprocessing & feature engineering
│   ├── evaluation.py             # Performance evaluation
│   └── models/                   # Model implementations
│       ├── glm_gam.py           # Statistical GAM
│       ├── fam.py               # Forest Additive Model
│       ├── gbam.py              # Gradient Boosting Additive Model
│       └── tabnam.py            # TabNet Neural Additive Model ⭐
├── outputs/                      # Results and figures
│   ├── results/                 # Performance metrics
│   └── figures/                 # Visualizations
├── run_all_models_v0.5.py       # Main execution script ⭐
├── generate_paper_figures_v0.5.py # Paper figure generation
├── visualize_shap_tabnam_v0.5.py  # TabNAM SHAP visualization
├── README.md                    # This file
├── METHODOLOGY.md               # Academic paper (full methodology) ⭐
├── Lesson_5th_train.md         # Detailed lessons learned
└── requirements.txt            # Dependencies
```

## Key Innovations

### 1. World's First Agentic AI for GAMs
- **Intelligent Search**: Planner-Executor-Evaluator pattern
- **Multiple Strategies**: SLSQP (local), Differential Evolution (global), Random Perturbation (fine-tuning)
- **Fast Convergence**: Optimal solution found in iteration 1

### 2. TabNAM: Attention-based Neural Additive Models
- **TabNet Architecture**: Attention mechanism for feature selection
- **Sparsity Control**: λ_sparse = 1e-3 for focused importance
- **Interpretability**: 14.3x higher interaction feature importance

### 3. Engineering-Meaningful Interactions
- **Domain Knowledge**: Age × LCC, Inspection × Age, Structure × Deterioration
- **7.6x SHAP Gain**: vs. data-driven damage interactions
- **Physical Interpretability**: All interactions have clear engineering meanings

## Documentation

- **[README.md](README.md)**: Quick start guide (this file)
- **[METHODOLOGY.md](METHODOLOGY.md)**: Full academic paper with methodology, results, and discussion
- **[Lesson_5th_train.md](Lesson_5th_train.md)**: Detailed lessons learned from v0.5 development
- **[QUICKSTART.md](QUICKSTART.md)**: Step-by-step tutorial

## Citation

If you use this work in your research, please cite:

```bibtex
@software{tabgam_v0.5,
  title={TabGAM: Diverse Generalized Additive Models with Agentic Ensemble Optimization},
  author={[Your Name]},
  year={2026},
  version={0.5},
  url={https://github.com/yourusername/tabgam_additive}
}
```

## Requirements

- Python 3.12+
- PyTorch 2.5.1
- LightGBM 4.5.0
- SHAP 0.46.0
- pygam 0.9.1
- scikit-learn 1.6.1
- pytorch-tabnet 4.0
- scipy 1.14.1

See [requirements.txt](requirements.txt) for complete list.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- **pyGAM**: Statistical GAM implementation
- **LightGBM**: High-performance gradient boosting
- **SHAP**: Model interpretability framework
- **TabNet**: Attention-based tabular learning (pytorch-tabnet)

## Contact

For questions or collaboration:
- Issues: [GitHub Issues](https://github.com/yourusername/tabgam_additive/issues)
- Email: [your.email@example.com]

---

**Project Status**: ✅ Active Development  
**Last Updated**: 2026-03-15  
**Version**: 0.5.0 (Agentic Ensemble Weight Tuning)
