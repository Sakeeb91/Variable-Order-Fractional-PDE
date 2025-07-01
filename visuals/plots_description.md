# Plots and Visualizations Documentation

This document provides comprehensive descriptions of all plots and figures generated throughout the Variable-Order Fractional PDE discovery project.

## Directory Structure

```
visuals/
├── plots_description.md          # This file
├── loss_curves/                  # Training loss evolution plots
├── solution_fields/              # Solution u(x,t) visualization
├── fractional_order/             # Discovered α(x) functions
├── comparisons/                  # Ground truth vs predicted comparisons
├── ablation_studies/             # Regularization impact analysis
└── convergence_analysis/         # Numerical convergence studies
```

## Plot Categories

### 1. Training Dynamics (`loss_curves/`)

#### `training_loss_evolution.png`
- **Description**: Multi-panel plot showing evolution of all loss components during training
- **Panels**: 
  - Total loss (L_total)
  - Data mismatch loss (L_data) 
  - PDE residual loss (L_res)
  - Regularization loss (L_reg)
- **X-axis**: Training iterations
- **Y-axis**: Loss magnitude (log scale)
- **Purpose**: Monitor training convergence and balance between loss components

#### `loss_component_breakdown.png`
- **Description**: Stacked area chart showing relative contribution of each loss component
- **Components**: Data, Residual, Smoothness Regularization, L1 Regularization
- **Purpose**: Understand which aspects dominate the optimization process

### 2. Solution Field Visualization (`solution_fields/`)

#### `solution_evolution_2d.png`
- **Description**: Heatmap series showing u(x,t) evolution over time
- **Layout**: 2x3 subplot grid for different time snapshots
- **Colormap**: Viridis (blue to yellow gradient)
- **Annotations**: Time values, spatial domain boundaries
- **Purpose**: Visualize spatio-temporal solution behavior

#### `solution_comparison_ground_truth.png`
- **Description**: Side-by-side comparison of predicted vs true solution
- **Left panel**: Ground truth u_true(x,t)
- **Right panel**: Predicted u_pred(x,t)
- **Bottom panel**: Absolute difference |u_pred - u_true|
- **Purpose**: Quantify solution accuracy visually

### 3. Fractional Order Discovery (`fractional_order/`)

#### `alpha_function_recovery.png`
- **Description**: Line plot comparing discovered α(x) with ground truth
- **Lines**: 
  - Ground truth α_true(x) (solid blue)
  - Predicted α_pred(x) (dashed red)
  - Confidence intervals (shaded region)
- **Annotations**: L2 relative error percentage
- **Purpose**: Primary validation of order discovery capability

#### `alpha_spatial_distribution.png`
- **Description**: Spatial heatmap of discovered fractional order
- **Colormap**: RdYlBu (red-yellow-blue) to highlight variation
- **Contour lines**: Iso-α curves overlaid
- **Purpose**: Visualize spatial heterogeneity in discovered order

### 4. Comparative Analysis (`comparisons/`)

#### `experiment_results_summary.png`
- **Description**: Multi-experiment comparison dashboard
- **Subplots**:
  - Constant order recovery (Experiment 1)
  - Smooth variation recovery (Experiment 2)
  - Step function recovery (challenging case)
- **Metrics displayed**: L2 error, Max error, R² correlation
- **Purpose**: Comprehensive method validation across scenarios

#### `error_convergence_analysis.png`
- **Description**: Semi-log plot of error vs training iterations
- **Lines**: Solution error, Order error, Combined error
- **Error bars**: Standard deviation across multiple runs
- **Purpose**: Demonstrate convergence properties and reliability

### 5. Ablation Studies (`ablation_studies/`)

#### `regularization_impact_comparison.png`
- **Description**: 2x2 grid comparing different regularization strategies
- **Cases**:
  - No regularization (top-left)
  - Smoothness only (top-right)
  - L1 only (bottom-left)
  - Full regularization (bottom-right)
- **Metric overlays**: Error values, smoothness measures
- **Purpose**: Demonstrate necessity and effectiveness of regularization

#### `hyperparameter_sensitivity.png`
- **Description**: Heatmap showing performance across regularization weights
- **Axes**: w_smooth (x-axis) vs w_L1 (y-axis)
- **Color**: L2 error in α(x) recovery
- **Contours**: Performance level sets
- **Purpose**: Guide hyperparameter selection

### 6. Convergence Analysis (`convergence_analysis/`)

#### `network_capacity_scaling.png`
- **Description**: Performance vs network size analysis
- **X-axis**: Number of parameters (log scale)
- **Y-axis**: Recovery accuracy (R²)
- **Separate curves**: Solution network and order network
- **Purpose**: Determine optimal network architectures

#### `data_efficiency_analysis.png`
- **Description**: Accuracy vs number of training data points
- **Multiple scenarios**: Different noise levels and spatial distributions
- **Error bars**: Statistical variation across random seeds
- **Purpose**: Understand data requirements for reliable discovery

## Visualization Standards

### Color Schemes
- **Sequential data**: Viridis, Plasma, or Cividis colormaps
- **Diverging data**: RdBu or RdYlBu colormaps
- **Categorical data**: Set1 or Dark2 color palettes
- **Comparison plots**: Blue vs Red for ground truth vs predicted

### Layout Guidelines
- **Figure size**: Minimum 10x8 inches for multi-panel plots
- **Font sizes**: Title (16pt), Axes labels (14pt), Tick labels (12pt)
- **Line widths**: 2pt for main data, 1pt for reference lines
- **Markers**: 8pt size, 50% transparency for scatter plots

### File Formats
- **Publication quality**: PDF and SVG formats
- **Web display**: PNG at 300 DPI
- **Interactive versions**: HTML with Plotly when applicable

## Plot Generation Scripts

Each plot category has corresponding generation scripts in `src/visualization/`:
- `plot_training_dynamics.py`
- `plot_solution_fields.py`
- `plot_fractional_order.py`
- `plot_comparisons.py`
- `plot_ablation_studies.py`
- `plot_convergence_analysis.py`

## Usage Notes

- All plots include timestamp and commit hash for reproducibility
- Raw data for plots stored in `data/processed/` with corresponding metadata
- Interactive versions available for detailed exploration
- Batch plot generation available via `generate_all_plots.py`

## Dependencies

Required packages for visualization:
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- plotly >= 5.0.0 (interactive plots)
- numpy >= 1.21.0
- scipy >= 1.7.0