#!/usr/bin/env python3
"""
Quick Start Tutorial for Variable-Order Fractional PDE Discovery

This tutorial demonstrates how to use the framework to discover
spatially varying fractional orders in PDEs.

Author: Sakeeb Rahman
Date: 2025
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.synthetic_data_generator import VariableOrderPDEGenerator
from models.solution_network import create_solution_network
from models.order_network import create_order_network
from utils.fractional_calculus import create_fractional_operator
from training.loss_functions import create_composite_loss
from training.trainer import create_trainer
from utils.visualization import ResultsVisualizer


def tutorial_basic_usage():
    """
    Tutorial 1: Basic usage with constant fractional order.
    
    This example shows how to set up and train the networks to recover
    a known constant fractional order.
    """
    print("=" * 60)
    print("Tutorial 1: Basic Usage - Constant Fractional Order")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic data...")
    generator = VariableOrderPDEGenerator(nx=51, nt=51)
    data = generator.constant_alpha_case(alpha=1.5)
    print(f"Generated {len(data['x_obs'])} observation points")
    
    # Step 2: Configure networks
    print("\\nStep 2: Configuring neural networks...")
    solution_config = {
        'type': 'basic',
        'hidden_layers': 3,
        'neurons_per_layer': 30,
        'activation': 'tanh'
    }
    
    order_config = {
        'type': 'basic',
        'hidden_layers': 2,
        'neurons_per_layer': 20,
        'activation': 'tanh',
        'alpha_bounds': (1.0, 2.0)
    }
    
    # Step 3: Set up training configuration
    print("Step 3: Setting up training configuration...")
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'solution_network': solution_config,
        'order_network': order_config,
        'fractional_operator': {
            'domain_bounds': (0.0, 1.0),
            'n_grid': 51
        },
        'loss_function': {
            'data_loss': {'weight': 1.0},
            'residual_loss': {'weight': 1.0},
            'regularization': {'weight': 0.01, 'l1_weight': 1.0}  # High L1 for constant
        },
        'optimizer': {'type': 'adam', 'lr': 1e-3},
        'n_collocation_points': 200,
        'domain': {'x_bounds': (0.0, 1.0), 't_bounds': (0.0, 1.0)}
    }
    
    # Step 4: Create trainer and prepare data
    print("Step 4: Creating trainer...")
    trainer = create_trainer(config)
    
    train_data = {
        'x': torch.tensor(data['x_obs'], dtype=torch.float32),
        't': torch.tensor(data['t_obs'], dtype=torch.float32),
        'u': torch.tensor(data['u_obs'], dtype=torch.float32).unsqueeze(-1)
    }
    
    # Step 5: Train the model
    print("Step 5: Training the model...")
    print("Training for 500 epochs (this may take a few minutes)...")
    history = trainer.train(train_data, n_epochs=500)
    
    # Step 6: Evaluate results
    print("\\nStep 6: Evaluating results...")
    x_eval = torch.linspace(0, 1, 51)
    t_eval = torch.zeros_like(x_eval)
    
    u_pred, alpha_pred = trainer.predict(x_eval, t_eval)
    
    alpha_mean = float(torch.mean(alpha_pred))
    alpha_std = float(torch.std(alpha_pred))
    
    print(f"True alpha: 1.5")
    print(f"Predicted alpha: {alpha_mean:.4f} ± {alpha_std:.4f}")
    print(f"Error: {abs(alpha_mean - 1.5):.4f}")
    
    # Step 7: Visualize results
    print("\\nStep 7: Creating visualizations...")
    visualizer = ResultsVisualizer(save_dir='tutorial_plots')
    
    # Plot alpha recovery
    visualizer.plot_alpha_function_recovery(
        x_eval, 
        torch.full_like(x_eval, 1.5),  # True constant alpha
        alpha_pred.flatten(),
        save_name='tutorial1_alpha_recovery.png'
    )
    
    # Plot training dynamics
    visualizer.plot_training_dynamics(history, 'tutorial1_training.png')
    
    print("Tutorial 1 completed! Check 'tutorial_plots/' for visualizations.")
    return trainer, data, history


def tutorial_variable_order():
    """
    Tutorial 2: Variable fractional order discovery.
    
    This example shows how to discover a spatially varying fractional order.
    """
    print("\\n" + "=" * 60)
    print("Tutorial 2: Variable Fractional Order Discovery")
    print("=" * 60)
    
    # Step 1: Generate data with variable alpha
    print("Step 1: Generating data with variable fractional order...")
    generator = VariableOrderPDEGenerator(nx=61, nt=51)
    data = generator.smooth_varying_alpha_case(amplitude=0.2, frequency=1)
    
    x_grid = data['x_grid']
    alpha_true = data['alpha_true']
    print(f"Alpha varies from {float(torch.min(alpha_true)):.3f} to {float(torch.max(alpha_true)):.3f}")
    
    # Step 2: Enhanced configuration for variable order
    print("\\nStep 2: Configuring enhanced networks...")
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'solution_network': {
            'type': 'adaptive',
            'hidden_layers': 4,
            'neurons_per_layer': 50,
            'activation': 'tanh',
            'use_layer_norm': True,
            'use_residual': True
        },
        'order_network': {
            'type': 'basic',
            'hidden_layers': 3,
            'neurons_per_layer': 40,
            'activation': 'tanh',
            'alpha_bounds': (1.0, 2.0)
        },
        'fractional_operator': {
            'domain_bounds': (0.0, 1.0),
            'n_grid': 61
        },
        'loss_function': {
            'data_loss': {'weight': 1.0},
            'residual_loss': {'weight': 1.0},
            'regularization': {
                'weight': 0.05,
                'smoothness_weight': 1.0,  # Encourage smoothness
                'l1_weight': 0.1          # Moderate L1 penalty
            },
            'adaptive_weights': True
        },
        'optimizer': {'type': 'adam', 'lr': 8e-4},
        'scheduler': {'type': 'plateau', 'patience': 100},
        'n_collocation_points': 500,
        'domain': {'x_bounds': (0.0, 1.0), 't_bounds': (0.0, 1.0)}
    }
    
    # Step 3: Train the model
    print("Step 3: Training for variable order discovery...")
    trainer = create_trainer(config)
    
    train_data = {
        'x': torch.tensor(data['x_obs'], dtype=torch.float32),
        't': torch.tensor(data['t_obs'], dtype=torch.float32),
        'u': torch.tensor(data['u_obs'], dtype=torch.float32).unsqueeze(-1)
    }
    
    print("Training for 1000 epochs...")
    history = trainer.train(train_data, n_epochs=1000)
    
    # Step 4: Evaluate and compare with ground truth
    print("\\nStep 4: Evaluating variable order recovery...")
    x_eval = x_grid
    t_eval = torch.zeros_like(x_eval)
    
    u_pred, alpha_pred = trainer.predict(x_eval, t_eval)
    
    # Calculate metrics
    l2_error = float(torch.sqrt(torch.mean((alpha_pred.flatten() - alpha_true)**2)))
    relative_error = l2_error / float(torch.sqrt(torch.mean(alpha_true**2))) * 100
    correlation = float(torch.corrcoef(torch.stack([alpha_pred.flatten(), alpha_true]))[0, 1])
    
    print(f"L2 Error: {l2_error:.4f}")
    print(f"Relative Error: {relative_error:.2f}%")
    print(f"Correlation: {correlation:.4f}")
    
    # Step 5: Create detailed visualizations
    print("\\nStep 5: Creating detailed visualizations...")
    visualizer = ResultsVisualizer(save_dir='tutorial_plots')
    
    # Alpha recovery comparison
    visualizer.plot_alpha_function_recovery(
        x_eval, alpha_true, alpha_pred.flatten(),
        save_name='tutorial2_alpha_recovery.png'
    )
    
    # Training dynamics
    visualizer.plot_training_dynamics(history, 'tutorial2_training.png')
    
    print("Tutorial 2 completed! Variable order successfully discovered.")
    return trainer, data, history, {'l2_error': l2_error, 'correlation': correlation}


def tutorial_custom_experiment():
    """
    Tutorial 3: Setting up a custom experiment.
    
    This example shows how to create your own experiment configuration.
    """
    print("\\n" + "=" * 60)
    print("Tutorial 3: Custom Experiment Setup")
    print("=" * 60)
    
    print("This tutorial shows the framework structure.")
    print("Key components:")
    print("  1. Data Generation: VariableOrderPDEGenerator")
    print("  2. Neural Networks: SolutionNetwork + OrderNetwork")
    print("  3. Physics: FractionalOperator (Grünwald-Letnikov)")
    print("  4. Training: CompositeLoss + VariableOrderPDETrainer")
    print("  5. Analysis: ResultsVisualizer")
    
    # Example custom configuration
    custom_config = {
        'experiment_name': 'my_custom_experiment',
        'description': 'Custom experiment with specific parameters',
        
        # Domain and discretization
        'domain': {
            'x_bounds': (0.0, 2.0),  # Larger spatial domain
            't_bounds': (0.0, 0.5)   # Shorter time domain
        },
        
        # Network architectures
        'solution_network': {
            'type': 'adaptive',
            'hidden_layers': 5,
            'neurons_per_layer': 60,
            'activation': 'gelu',  # Different activation
            'use_residual': True
        },
        
        'order_network': {
            'type': 'multiscale',  # Multi-scale for complex patterns
            'base_neurons': 20,
            'num_scales': 3,
            'alpha_bounds': (0.5, 2.5)  # Wider range
        },
        
        # Training parameters
        'loss_function': {
            'data_loss': {'weight': 1.0, 'loss_type': 'huber'},
            'residual_loss': {'weight': 2.0},  # Higher physics weight
            'regularization': {
                'weight': 0.02,
                'smoothness_weight': 0.5,
                'l1_weight': 0.1,
                'tv_weight': 0.1  # Total variation
            }
        },
        
        'optimizer': {
            'type': 'adam',
            'lr': 5e-4,
            'weight_decay': 1e-5
        },
        
        'n_epochs': 2000,
        'n_collocation_points': 1000
    }
    
    print("\\nExample custom configuration created!")
    print("You can modify any parameters to suit your specific problem.")
    print("\\nTo run your custom experiment:")
    print("  1. Modify the configuration above")
    print("  2. Generate appropriate synthetic data")
    print("  3. Create trainer with your config")
    print("  4. Train and evaluate")
    
    return custom_config


def run_all_tutorials():
    """Run all tutorials in sequence."""
    print("Running all Variable-Order Fractional PDE Discovery Tutorials")
    print("This will demonstrate the complete framework capabilities.")
    
    # Tutorial 1: Basic usage
    tutorial_basic_usage()
    
    # Tutorial 2: Variable order
    tutorial_variable_order()
    
    # Tutorial 3: Custom setup
    tutorial_custom_experiment()
    
    print("\\n" + "=" * 60)
    print("All tutorials completed successfully!")
    print("=" * 60)
    print("\\nNext steps:")
    print("1. Check the generated plots in 'tutorial_plots/'")
    print("2. Experiment with different configurations")
    print("3. Try the main execution script: python run_experiments.py")
    print("4. Read the documentation in visuals/plots_description.md")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Variable-Order Fractional PDE Tutorial")
    parser.add_argument('--tutorial', type=int, choices=[1, 2, 3], 
                       help='Run specific tutorial (1, 2, or 3)')
    parser.add_argument('--all', action='store_true', 
                       help='Run all tutorials')
    
    args = parser.parse_args()
    
    if args.tutorial == 1:
        tutorial_basic_usage()
    elif args.tutorial == 2:
        tutorial_variable_order()
    elif args.tutorial == 3:
        tutorial_custom_experiment()
    elif args.all:
        run_all_tutorials()
    else:
        print("Variable-Order Fractional PDE Discovery - Quick Start Tutorial")
        print("\\nUsage:")
        print("  python quick_start_tutorial.py --tutorial 1  # Basic usage")
        print("  python quick_start_tutorial.py --tutorial 2  # Variable order")
        print("  python quick_start_tutorial.py --tutorial 3  # Custom setup")
        print("  python quick_start_tutorial.py --all         # All tutorials")
        print("\\nFor the full framework, use: python ../run_experiments.py")