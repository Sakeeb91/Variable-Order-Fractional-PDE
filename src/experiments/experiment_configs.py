"""
Experiment Configurations for Variable-Order Fractional PDE Discovery

This module defines the three main validation experiments:
1. Constant order recovery (sanity check)
2. Smooth variation recovery 
3. Step function recovery (challenging case)

Author: Sakeeb Rahman
Date: 2025
"""

import torch
import numpy as np
from typing import Dict, Any


def get_base_config() -> Dict[str, Any]:
    """Get base configuration shared across all experiments."""
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'domain': {
            'x_bounds': (0.0, 1.0),
            't_bounds': (0.0, 1.0)
        },
        'fractional_operator': {
            'domain_bounds': (0.0, 1.0),
            'n_grid': 101
        },
        'forcing_term': {
            'type': 'zero'
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'type': 'plateau',
            'factor': 0.5,
            'patience': 100,
            'verbose': True
        },
        'n_collocation_points': 1000,
        'log_freq': 100,
        'val_freq': 50,
        'checkpoint_freq': 500,
        'checkpoint_dir': 'experiments/checkpoints'
    }


def get_experiment1_config() -> Dict[str, Any]:
    """
    Experiment 1: Constant fractional order recovery (sanity check).
    
    Goal: Verify that the method can recover a known constant α = 1.5
    Expected: α_NN should predict a nearly flat line at 1.5
    """
    config = get_base_config()
    
    config.update({
        'experiment_name': 'experiment1_constant_alpha',
        'description': 'Constant fractional order α = 1.5 recovery',
        'n_epochs': 2000,
        
        # Network architectures
        'solution_network': {
            'type': 'basic',
            'hidden_layers': 4,
            'neurons_per_layer': 50,
            'activation': 'tanh',
            'initialization': 'xavier_normal'
        },
        
        'order_network': {
            'type': 'basic',
            'hidden_layers': 3,
            'neurons_per_layer': 30,
            'activation': 'tanh',
            'alpha_bounds': (1.0, 2.0),
            'constraint_type': 'tanh',
            'initialization': 'xavier_normal'
        },
        
        # Loss function configuration
        'loss_function': {
            'data_loss': {
                'weight': 1.0,
                'loss_type': 'mse'
            },
            'residual_loss': {
                'weight': 1.0,
                'pde_form': 'fractional_diffusion'
            },
            'regularization': {
                'weight': 0.01,
                'smoothness_weight': 0.1,  # Low smoothness weight for constant case
                'l1_weight': 1.0,          # High L1 to encourage constant
                'l2_weight': 0.01
            },
            'adaptive_weights': False
        },
        
        # Data configuration
        'data_config': {
            'dataset_file': 'data/processed/experiment1_constant_alpha.npz',
            'train_split': 0.8,
            'noise_level': 0.01
        },
        
        # Validation metrics
        'success_criteria': {
            'alpha_variance_threshold': 0.01,  # α should be nearly constant
            'alpha_mean_error_threshold': 0.05,  # Close to true value
            'solution_mse_threshold': 1e-3
        }
    })
    
    return config


def get_experiment2_config() -> Dict[str, Any]:
    """
    Experiment 2: Smoothly varying fractional order recovery.
    
    Goal: Recover α(x) = 0.25 * sin(2πx) + 1.5
    Expected: Smooth sinusoidal variation in the discovered α(x)
    """
    config = get_base_config()
    
    config.update({
        'experiment_name': 'experiment2_smooth_alpha',
        'description': 'Smooth varying α(x) = 0.25*sin(2πx) + 1.5',
        'n_epochs': 3000,
        
        # Enhanced network architectures for variable order
        'solution_network': {
            'type': 'adaptive',
            'hidden_layers': 5,
            'neurons_per_layer': 60,
            'activation': 'tanh',
            'use_layer_norm': True,
            'use_residual': True,
            'dropout_rate': 0.0
        },
        
        'order_network': {
            'type': 'basic',
            'hidden_layers': 4,
            'neurons_per_layer': 40,
            'activation': 'tanh',
            'alpha_bounds': (1.0, 2.0),
            'constraint_type': 'tanh',
            'initialization': 'xavier_normal'
        },
        
        # Balanced loss function
        'loss_function': {
            'data_loss': {
                'weight': 1.0,
                'loss_type': 'mse',
                'robust_loss': True  # Handle potential outliers
            },
            'residual_loss': {
                'weight': 1.0,
                'pde_form': 'fractional_diffusion'
            },
            'regularization': {
                'weight': 0.05,
                'smoothness_weight': 1.0,  # Encourage smoothness
                'l1_weight': 0.1,          # Moderate L1 penalty
                'l2_weight': 0.01,
                'tv_weight': 0.01         # Total variation for smoothness
            },
            'adaptive_weights': True,
            'weight_update_freq': 200
        },
        
        # Enhanced optimizer for complex landscape
        'optimizer': {
            'type': 'adam',
            'lr': 8e-4,  # Slightly lower learning rate
            'weight_decay': 1e-4
        },
        
        'scheduler': {
            'type': 'cosine',
            'T_max': 3000,
            'eta_min': 1e-6
        },
        
        'data_config': {
            'dataset_file': 'data/processed/experiment2_smooth_alpha.npz',
            'train_split': 0.8,
            'noise_level': 0.015
        },
        
        'success_criteria': {
            'alpha_l2_error_threshold': 0.1,    # L2 error in α recovery
            'alpha_correlation_threshold': 0.9,  # Correlation with true α
            'solution_mse_threshold': 5e-3
        }
    })
    
    return config


def get_experiment3_config() -> Dict[str, Any]:
    """
    Experiment 3: Step function fractional order recovery (challenging case).
    
    Goal: Recover discontinuous α(x) with smooth transitions
    Expected: Sharp but smooth transition from α=1.3 to α=1.7
    """
    config = get_base_config()
    
    config.update({
        'experiment_name': 'experiment3_step_alpha',
        'description': 'Step function α(x): 1.3 → 1.7 with smooth transition',
        'n_epochs': 4000,  # More epochs for challenging case
        
        # Multi-scale network for capturing sharp transitions
        'solution_network': {
            'type': 'adaptive',
            'hidden_layers': 6,
            'neurons_per_layer': 80,
            'activation': 'tanh',
            'use_layer_norm': True,
            'use_residual': True,
            'dropout_rate': 0.1  # Regularization for complex case
        },
        
        'order_network': {
            'type': 'multiscale',
            'base_neurons': 25,
            'num_scales': 4,  # Multiple scales for sharp features
            'activation': 'tanh',
            'alpha_bounds': (1.0, 2.0),
            'constraint_type': 'tanh'
        },
        
        # Carefully tuned loss for discontinuous features
        'loss_function': {
            'data_loss': {
                'weight': 1.0,
                'loss_type': 'huber',  # Robust to outliers
                'huber_delta': 0.1
            },
            'residual_loss': {
                'weight': 1.5,  # Higher physics weight
                'pde_form': 'fractional_diffusion'
            },
            'regularization': {
                'weight': 0.02,
                'smoothness_weight': 0.5,  # Reduced smoothness for discontinuity
                'l1_weight': 0.05,         # Low L1 to allow variation
                'l2_weight': 0.005,
                'tv_weight': 0.1          # Total variation for edge preservation
            },
            'adaptive_weights': True,
            'weight_update_freq': 150
        },
        
        # Conservative optimizer settings
        'optimizer': {
            'type': 'adam',
            'lr': 5e-4,  # Lower learning rate for stability
            'weight_decay': 5e-5
        },
        
        'scheduler': {
            'type': 'plateau',
            'factor': 0.7,
            'patience': 200,
            'verbose': True
        },
        
        'n_collocation_points': 1500,  # More collocation points
        
        'data_config': {
            'dataset_file': 'data/processed/experiment3_step_alpha.npz',
            'train_split': 0.8,
            'noise_level': 0.02  # Higher noise for robustness
        },
        
        'success_criteria': {
            'alpha_step_detection': True,      # Detect step transition
            'transition_sharpness': 0.2,      # Width of transition region
            'plateau_values_error': 0.1,      # Error in plateau regions
            'solution_mse_threshold': 1e-2
        }
    })
    
    return config


def get_ablation_study_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for ablation study on regularization components.
    
    Returns:
        Dictionary with different regularization configurations
    """
    base_config = get_experiment2_config()  # Use smooth case as baseline
    
    configs = {}
    
    # No regularization
    config_no_reg = base_config.copy()
    config_no_reg['experiment_name'] = 'ablation_no_regularization'
    config_no_reg['loss_function']['regularization']['weight'] = 0.0
    configs['no_regularization'] = config_no_reg
    
    # Smoothness only
    config_smooth_only = base_config.copy()
    config_smooth_only['experiment_name'] = 'ablation_smoothness_only'
    config_smooth_only['loss_function']['regularization'].update({
        'smoothness_weight': 1.0,
        'l1_weight': 0.0,
        'l2_weight': 0.0,
        'tv_weight': 0.0
    })
    configs['smoothness_only'] = config_smooth_only
    
    # L1 only
    config_l1_only = base_config.copy()
    config_l1_only['experiment_name'] = 'ablation_l1_only'
    config_l1_only['loss_function']['regularization'].update({
        'smoothness_weight': 0.0,
        'l1_weight': 1.0,
        'l2_weight': 0.0,
        'tv_weight': 0.0
    })
    configs['l1_only'] = config_l1_only
    
    # Full regularization (baseline)
    config_full = base_config.copy()
    config_full['experiment_name'] = 'ablation_full_regularization'
    configs['full_regularization'] = config_full
    
    return configs


def get_hyperparameter_sweep_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for hyperparameter sensitivity analysis.
    
    Returns:
        Dictionary with different hyperparameter settings
    """
    base_config = get_experiment2_config()
    configs = {}
    
    # Different regularization weights
    reg_weights = [0.001, 0.01, 0.05, 0.1, 0.2]
    for weight in reg_weights:
        config = base_config.copy()
        config['experiment_name'] = f'hyperparam_reg_weight_{weight}'
        config['loss_function']['regularization']['weight'] = weight
        configs[f'reg_weight_{weight}'] = config
    
    # Different network sizes
    network_sizes = [(3, 20), (4, 40), (5, 60), (6, 80)]
    for layers, neurons in network_sizes:
        config = base_config.copy()
        config['experiment_name'] = f'hyperparam_network_{layers}x{neurons}'
        config['order_network'].update({
            'hidden_layers': layers,
            'neurons_per_layer': neurons
        })
        configs[f'network_{layers}x{neurons}'] = config
    
    return configs


# Utility functions
def validate_config(config: Dict[str, Any]) -> bool:
    """Validate experiment configuration."""
    required_keys = [
        'experiment_name', 'solution_network', 'order_network',
        'loss_function', 'optimizer', 'n_epochs'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate alpha bounds
    bounds = config['order_network']['alpha_bounds']
    if bounds[0] >= bounds[1]:
        raise ValueError("Invalid alpha bounds")
    
    return True


def get_config_by_name(experiment_name: str) -> Dict[str, Any]:
    """Get configuration by experiment name."""
    configs = {
        'experiment1': get_experiment1_config(),
        'experiment2': get_experiment2_config(),
        'experiment3': get_experiment3_config()
    }
    
    if experiment_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {available}")
    
    config = configs[experiment_name]
    validate_config(config)
    
    return config


# Example usage
if __name__ == "__main__":
    print("Testing experiment configurations...")
    
    # Test all main experiments
    for exp_name in ['experiment1', 'experiment2', 'experiment3']:
        config = get_config_by_name(exp_name)
        print(f"✓ {exp_name}: {config['description']}")
        
        # Print key parameters
        print(f"  Epochs: {config['n_epochs']}")
        print(f"  Alpha bounds: {config['order_network']['alpha_bounds']}")
        print(f"  Regularization weight: {config['loss_function']['regularization']['weight']}")
        print()
    
    # Test ablation study configs
    ablation_configs = get_ablation_study_configs()
    print(f"Ablation study configurations: {len(ablation_configs)}")
    for name in ablation_configs:
        print(f"  ✓ {name}")
    
    # Test hyperparameter sweep
    hyperparam_configs = get_hyperparameter_sweep_configs()
    print(f"\\nHyperparameter sweep configurations: {len(hyperparam_configs)}")
    
    print("\\nAll configurations validated successfully!")