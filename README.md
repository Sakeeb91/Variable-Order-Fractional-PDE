# Data-Driven Discovery of Variable-Order Fractional PDEs

## Overview

This project implements a novel approach to discover spatially varying fractional orders in partial differential equations using Physics-Informed Neural Networks (PINNs). Instead of assuming a constant fractional order α, we learn the function α(x) that governs the heterogeneous anomalous diffusion behavior across the spatial domain.

## Problem Formulation

We aim to discover the variable fractional order α(x) in the governing equation:

```
c(-Δ)^α(x)/2 u(x,t) + v · ∇u(x,t) - f(x,t) = 0
```

This represents a significant advancement from traditional fractional PDEs with constant order, enabling modeling of:
- **Subsurface Flow**: Spatially varying aquifer properties
- **Viscoelastic Materials**: Composite materials with heterogeneous memory effects  
- **Medical Imaging**: Variable tissue properties and signal attenuation

## Methodology

### Dual-Network Architecture

1. **Solution Network (u_NN)**: Approximates the solution u(x,t)
   - Input: Spatio-temporal coordinates (x,t)
   - Output: Predicted solution u
   - Parameters: θ_u

2. **Order Network (α_NN)**: Learns the variable fractional order
   - Input: Spatial coordinate x
   - Output: Predicted fractional order α
   - Parameters: θ_α

### Loss Function Components

The networks are trained simultaneously using a composite loss function:

```
L_total = w_res * L_res + w_data * L_data + w_reg * L_reg
```

- **L_data**: Data mismatch loss (MSE between predictions and observations)
- **L_res**: PDE residual loss (enforces physics constraints)
- **L_reg**: Regularization loss (promotes smoothness and simplicity)

### Key Innovations

1. **Variable Discretization**: Grünwald-Letnikov coefficients computed point-wise based on local α(x)
2. **Regularization Strategy**: Combined smoothness and L1 penalties to ensure physically meaningful solutions
3. **Constrained Optimization**: Bounded activation functions ensure α ∈ (1,2) for fractional Laplacian

## Experimental Design

### Validation Experiments

1. **Sanity Check**: Recovery of constant fractional order
2. **Smooth Variation**: Recovery of sinusoidally varying α(x)
3. **Ablation Study**: Impact of different regularization components

### Success Metrics

- Relative L2 error between predicted and true α(x)
- Solution accuracy for u(x,t)
- Regularization effectiveness analysis

## Repository Structure

```
├── src/                    # Source code
│   ├── models/            # Neural network architectures
│   ├── training/          # Training loops and optimization
│   ├── utils/             # Utility functions and helpers
│   └── experiments/       # Experimental configurations
├── data/                  # Dataset and synthetic data generation
├── visuals/               # Plots, figures, and visualization outputs
│   └── plots_description.md  # Detailed description of all plots
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# Generate all synthetic datasets
python run_experiments.py --generate-data

# Run individual experiments
python run_experiments.py --experiment experiment1  # Constant α recovery
python run_experiments.py --experiment experiment2  # Smooth α(x) variation  
python run_experiments.py --experiment experiment3  # Step function α(x)

# Run all experiments
python run_experiments.py --all

# Run ablation study
python run_experiments.py --ablation

# Interactive tutorial
python examples/quick_start_tutorial.py --all
```

## Key Features

- **Simultaneous Discovery**: Learn both solution and fractional order simultaneously
- **Physics-Informed**: Incorporates PDE constraints directly into training
- **Regularized Learning**: Prevents overfitting through smoothness and simplicity priors
- **Comprehensive Validation**: Multiple experimental scenarios to validate methodology

## Applications

This framework enables modeling of complex physical systems with spatially varying anomalous diffusion:

- Heterogeneous porous media flow
- Composite material behavior
- Biological tissue modeling
- Financial market dynamics with varying memory effects

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{variable_order_fpde_2024,
  title={Data-Driven Discovery of Variable-Order Fractional PDEs},
  author={Sakeeb Rahman},
  year={2024},
  url={https://github.com/Sakeeb91/Variable-Order-Fractional-PDE}
}
```

This work builds upon the foundational fPINNs framework:

```bibtex
@article{Pang_2019,
   title={fPINNs: Fractional Physics-Informed Neural Networks},
   volume={41},
   ISSN={1095-7197},
   url={http://dx.doi.org/10.1137/18M1229845},
   DOI={10.1137/18m1229845},
   number={4},
   journal={SIAM Journal on Scientific Computing},
   publisher={Society for Industrial & Applied Mathematics (SIAM)},
   author={Pang, Guofei and Lu, Lu and Karniadakis, George Em},
   year={2019},
   month=jan, pages={A2603–A2626}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.