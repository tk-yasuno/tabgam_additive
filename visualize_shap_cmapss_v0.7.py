"""
SHAP Analysis for Jet Engine Degradation (v0.7)
CMAPSS FD004: RUL Prediction with Multi-Mechanism Degradation

This script performs SHAP analysis to interpret model predictions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pygam import GAM, s, l
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
from data_preprocessing_cmapss_v07 import load_and_preprocess_data_v07

# Output directory
os.makedirs('outputs/figures/cmapss_v0.7', exist_ok=True)

def train_gam_for_shap(X_train, y_train, n_splines=10):
    """Train GAM for SHAP analysis"""
    print("\nTraining GAM for SHAP analysis...")
    
    # Convert DataFrame to ndarray if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    
    # Build GAM formula
    n_features = X_train.shape[1]
    print(f"  Number of features: {n_features}")
    
    # Create GAM terms
    if n_features == 0:
        raise ValueError("No features found in X_train")
    
    gam_terms = s(0, n_splines=n_splines)
    for i in range(1, n_features):
        gam_terms = gam_terms + s(i, n_splines=n_splines)
    
    # Train GAM
    gam = GAM(gam_terms, max_iter=200, verbose=False)
    gam.fit(X_train, y_train)
    
    return gam

def compute_shap_values(model, X_sample, X_background):
    """Compute SHAP values for GAM model"""
    print("\nComputing SHAP values...")
    print(f"  Background samples: {len(X_background)}")
    print(f"  Explanation samples: {len(X_sample)}")
    
    # Create SHAP explainer
    explainer = shap.Explainer(model.predict, X_background)
    shap_values = explainer(X_sample)
    
    return shap_values

def plot_shap_summary(shap_values, X_sample, feature_names, output_path):
    """Create SHAP summary plot"""
    print(f"\nCreating SHAP summary plot: {output_path}")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance - Jet Engine RUL Prediction', fontsize=14, weight='bold')
    plt.xlabel('SHAP Value (impact on RUL)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")

def plot_shap_bar(shap_values, feature_names, output_path):
    """Create SHAP bar plot (mean absolute importance)"""
    print(f"\nCreating SHAP bar plot: {output_path}")
    
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False)
    plt.title('SHAP Feature Importance - Mean Absolute Impact', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")

def plot_shap_dependence_top_features(shap_values, X_sample, feature_names, output_path, top_k=6):
    """Create SHAP dependence plots for top-k features"""
    print(f"\nCreating SHAP dependence plots for top {top_k} features: {output_path}")
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature_idx in enumerate(top_indices):
        ax = axes[idx]
        feature_name = feature_names[feature_idx]
        
        # Scatter plot of feature value vs SHAP value
        scatter = ax.scatter(
            X_sample[:, feature_idx],
            shap_values.values[:, feature_idx],
            c=X_sample[:, feature_idx],
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        
        ax.set_xlabel(f'{feature_name}', fontsize=11)
        ax.set_ylabel('SHAP Value', fontsize=11)
        ax.set_title(f'SHAP Dependence: {feature_name}', fontsize=12, weight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Feature Value', fontsize=9)
    
    plt.suptitle('SHAP Dependence Plots - Top 6 Features', fontsize=16, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")

def plot_shap_waterfall(shap_values, X_sample, feature_names, output_path, sample_idx=0):
    """Create SHAP waterfall plot for a single prediction"""
    print(f"\nCreating SHAP waterfall plot for sample {sample_idx}: {output_path}")
    
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    plt.title(f'SHAP Waterfall - Sample {sample_idx} Explanation', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")

def analyze_multi_mechanism_impact(shap_values, X_sample, feature_names):
    """Analyze impact of different sensor groups and operational conditions"""
    print("\n" + "="*80)
    print("MULTI-MECHANISM DEGRADATION ANALYSIS")
    print("="*80)
    
    # Define feature groups
    feature_groups = {
        'Operational Settings': [f for f in feature_names if f.startswith('op_setting')],
        'Core Sensors': [f for f in feature_names if any(s in f for s in ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc'])],
        'Temperature Sensors': [f for f in feature_names if 'T' in f and 'roll' not in f],
        'Pressure Sensors': [f for f in feature_names if 'P' in f and 'roll' not in f],
        'Rolling Features': [f for f in feature_names if 'roll' in f],
        'Cycle': ['cycle_norm']
    }
    
    # Calculate mean absolute SHAP for each group
    group_importance = {}
    for group_name, group_features in feature_groups.items():
        group_indices = [i for i, f in enumerate(feature_names) if f in group_features]
        if group_indices:
            group_shap = np.abs(shap_values.values[:, group_indices]).mean()
            group_importance[group_name] = group_shap
            print(f"\n{group_name}:")
            print(f"  Mean |SHAP|: {group_shap:.4f}")
            print(f"  Features: {len(group_features)}")
    
    # Top individual features
    print("\n" + "-"*80)
    print("TOP 10 INDIVIDUAL FEATURES")
    print("-"*80)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank:2d}. {feature_names[idx]:30s} | SHAP: {mean_abs_shap[idx]:.4f}")
    
    return group_importance

def main():
    print("="*80)
    print("SHAP ANALYSIS FOR JET ENGINE DEGRADATION (v0.7)")
    print("CMAPSS FD004: RUL Prediction")
    print("="*80)
    
    # Load and preprocess data
    print("\n--- Loading Data ---")
    data, preprocessor = load_and_preprocess_data_v07()
    
    X_train_full = data['X_train']
    y_train_full = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    print(f"Training set: {X_train_full.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Sample for SHAP analysis (use subset for speed)
    n_background = 1000
    n_explain = 500
    
    np.random.seed(42)
    bg_indices = np.random.choice(len(X_train_full), n_background, replace=False)
    explain_indices = np.random.choice(len(X_test), n_explain, replace=False)
    
    # Handle both DataFrame and ndarray
    if hasattr(X_train_full, 'iloc'):
        X_background = X_train_full.iloc[bg_indices].values
        X_explain = X_test.iloc[explain_indices].values
    else:
        X_background = X_train_full[bg_indices]
        X_explain = X_test[explain_indices]
    y_explain = y_test[explain_indices]
    
    print(f"\nSHAP Analysis Sample:")
    print(f"  Background: {n_background} samples")
    print(f"  Explain: {n_explain} samples")
    
    # Train GAM
    gam = train_gam_for_shap(X_train_full, y_train_full)
    
    # Evaluate GAM
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    y_pred_test = gam.predict(X_test_array)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\nGAM Performance on Test Set:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE = {mae:.4f}")
    
    # Compute SHAP values
    shap_values = compute_shap_values(gam, X_explain, X_background)
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING SHAP VISUALIZATIONS")
    print("="*80)
    
    plot_shap_summary(
        shap_values, X_explain, feature_names,
        'outputs/figures/cmapss_v0.7/shap_summary.png'
    )
    
    plot_shap_bar(
        shap_values, feature_names,
        'outputs/figures/cmapss_v0.7/shap_bar.png'
    )
    
    plot_shap_dependence_top_features(
        shap_values, X_explain, feature_names,
        'outputs/figures/cmapss_v0.7/shap_dependence_top6.png',
        top_k=6
    )
    
    # Waterfall plots for representative samples
    # Find high, medium, low RUL samples
    rul_quantiles = np.percentile(y_explain, [10, 50, 90])
    sample_indices = []
    for q in rul_quantiles:
        idx = np.argmin(np.abs(y_explain - q))
        sample_indices.append(idx)
    
    for i, idx in enumerate(sample_indices):
        plot_shap_waterfall(
            shap_values, X_explain, feature_names,
            f'outputs/figures/cmapss_v0.7/shap_waterfall_sample_{i+1}.png',
            sample_idx=idx
        )
    
    # Multi-mechanism analysis
    group_importance = analyze_multi_mechanism_impact(shap_values, X_explain, feature_names)
    
    # Save feature importance table
    print("\n--- Saving Feature Importance Table ---")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    output_csv = 'outputs/results/shap_feature_importance_v0.7_cmapss.csv'
    importance_df.to_csv(output_csv, index=False)
    print(f"  ✅ Saved: {output_csv}")
    
    print("\n" + "="*80)
    print("SHAP ANALYSIS COMPLETED")
    print("="*80)
    print(f"\nVisualization files saved in: outputs/figures/cmapss_v0.7/")
    print(f"Feature importance saved in: {output_csv}")

if __name__ == '__main__':
    main()
