#!/usr/bin/env python3
"""
Main Execution Script for Variable-Order Fractional PDE Discovery

This script provides a unified interface for running all experiments,
generating datasets, training models, and creating visualizations.

Usage:
    python run_experiments.py --experiment experiment1
    python run_experiments.py --all
    python run_experiments.py --generate-data
    python run_experiments.py --visualize results/experiment1/

Author: Sakeeb Rahman
Date: 2025
"""

import argparse
import os
import sys
import torch
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.synthetic_data_generator import VariableOrderPDEGenerator, generate_all_datasets
from experiments.experiment_configs import get_config_by_name, get_ablation_study_configs
from training.trainer import create_trainer
from utils.visualization import ResultsVisualizer


def setup_logging(log_file: str = None) -> logging.Logger:
    """Setup logging configuration."""
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'experiments_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('ExperimentRunner')


def generate_synthetic_data(args):
    """Generate all synthetic datasets."""
    logger = logging.getLogger('DataGeneration')
    logger.info("Generating synthetic datasets...")
    
    try:
        generate_all_datasets()
        logger.info("All datasets generated successfully!")
    except Exception as e:
        logger.error(f"Error generating datasets: {e}")
        raise


def load_dataset(dataset_file: str) -> dict:
    """Load dataset from file."""
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    
    data = np.load(dataset_file)
    return {
        'x': torch.tensor(data['x_obs'], dtype=torch.float32),
        't': torch.tensor(data['t_obs'], dtype=torch.float32),
        'u': torch.tensor(data['u_obs'], dtype=torch.float32).unsqueeze(-1),
        'x_grid': torch.tensor(data['x_grid'], dtype=torch.float32),
        'alpha_true': torch.tensor(data['alpha_true'], dtype=torch.float32),
        'description': str(data['description'])
    }


def run_single_experiment(experiment_name: str, args):
    """Run a single experiment."""
    logger = logging.getLogger(f'Experiment_{experiment_name}')
    logger.info(f"Starting {experiment_name}")
    
    # Get configuration
    config = get_config_by_name(experiment_name)
    logger.info(f"Configuration loaded: {config['description']}")
    
    # Create results directory
    results_dir = f"results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    config['checkpoint_dir'] = f"{results_dir}/checkpoints"
    
    # Load dataset
    dataset_file = config['data_config']['dataset_file']
    logger.info(f"Loading dataset: {dataset_file}")
    
    try:
        dataset = load_dataset(dataset_file)
    except FileNotFoundError:
        logger.warning(f"Dataset not found. Generating datasets first...")
        generate_synthetic_data(args)
        dataset = load_dataset(dataset_file)
    
    # Split data
    n_data = len(dataset['x'])
    train_split = config['data_config']['train_split']
    n_train = int(n_data * train_split)
    
    indices = torch.randperm(n_data)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = {
        'x': dataset['x'][train_indices],
        't': dataset['t'][train_indices], 
        'u': dataset['u'][train_indices]
    }
    
    val_data = {
        'x': dataset['x'][val_indices],
        't': dataset['t'][val_indices],
        'u': dataset['u'][val_indices]
    } if len(val_indices) > 0 else None
    
    logger.info(f"Training data: {len(train_data['x'])} points")
    if val_data:
        logger.info(f"Validation data: {len(val_data['x'])} points")
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        n_epochs=config['n_epochs']
    )
    
    # Evaluate on full grid
    logger.info("Evaluating on full grid...")
    with torch.no_grad():
        x_eval = dataset['x_grid']
        t_eval = torch.zeros_like(x_eval)  # Evaluate at t=0
        
        u_pred, alpha_pred = trainer.predict(x_eval, t_eval)
        
        # Calculate metrics
        alpha_true = dataset['alpha_true']
        alpha_mse = torch.mean((alpha_pred.flatten() - alpha_true)**2).item()
        alpha_l2_error = torch.sqrt(torch.mean((alpha_pred.flatten() - alpha_true)**2)).item()
        
        # Solution metrics on validation data if available
        if val_data:
            u_val_pred, _ = trainer.predict(val_data['x'], val_data['t'])
            solution_mse = torch.mean((u_val_pred - val_data['u'])**2).item()
        else:
            solution_mse = 0.0
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'training_history': training_history,
        'final_metrics': {
            'alpha_mse': alpha_mse,
            'alpha_l2_error': alpha_l2_error,
            'solution_mse': solution_mse
        },
        'predictions': {
            'x_grid': x_eval.cpu().numpy(),
            'alpha_true': alpha_true.cpu().numpy(),
            'alpha_pred': alpha_pred.cpu().numpy(),
            'u_pred': u_pred.cpu().numpy()
        }
    }
    
    # Save results
    results_file = f"{results_dir}/results.npz"
    np.savez(results_file, **results)
    logger.info(f"Results saved to: {results_file}")
    
    # Generate visualizations
    if not args.no_plots:
        logger.info("Generating visualizations...")
        visualizer = ResultsVisualizer(save_dir=f"{results_dir}/plots")
        
        # Training dynamics
        visualizer.plot_training_dynamics(
            training_history, 
            f"{experiment_name}_training_dynamics.png"
        )
        
        # Alpha recovery
        visualizer.plot_alpha_function_recovery(
            torch.tensor(results['predictions']['x_grid']),
            torch.tensor(results['predictions']['alpha_true']),
            torch.tensor(results['predictions']['alpha_pred']),
            save_name=f"{experiment_name}_alpha_recovery.png"
        )
        
        logger.info(f"Visualizations saved to: {results_dir}/plots/")
    
    # Print summary
    logger.info("Experiment completed!")
    logger.info(f"Final metrics:")
    logger.info(f"  Alpha L2 Error: {alpha_l2_error:.6f}")
    logger.info(f"  Alpha MSE: {alpha_mse:.6f}")
    logger.info(f"  Solution MSE: {solution_mse:.6f}")
    
    return results


def run_ablation_study(args):
    """Run ablation study on regularization components."""
    logger = logging.getLogger('AblationStudy')
    logger.info("Starting ablation study...")
    
    configs = get_ablation_study_configs()
    results = {}
    
    for study_name, config in configs.items():
        logger.info(f"Running ablation case: {study_name}")
        
        # Create results directory
        results_dir = f"results/ablation/{study_name}"
        os.makedirs(results_dir, exist_ok=True)
        config['checkpoint_dir'] = f"{results_dir}/checkpoints"
        
        # Load dataset (use experiment2 data)
        dataset_file = "data/processed/experiment2_smooth_alpha.npz"
        
        try:
            dataset = load_dataset(dataset_file)
        except FileNotFoundError:
            logger.warning("Dataset not found. Generating datasets first...")
            generate_synthetic_data(args)
            dataset = load_dataset(dataset_file)
        
        # Prepare training data
        n_data = len(dataset['x'])
        train_split = 0.8
        n_train = int(n_data * train_split)
        
        indices = torch.randperm(n_data)
        train_data = {
            'x': dataset['x'][:n_train],
            't': dataset['t'][:n_train],
            'u': dataset['u'][:n_train]
        }
        
        # Create and train
        trainer = create_trainer(config)
        training_history = trainer.train(train_data, n_epochs=1000)  # Shorter for ablation
        
        # Evaluate
        x_eval = dataset['x_grid']
        t_eval = torch.zeros_like(x_eval)
        u_pred, alpha_pred = trainer.predict(x_eval, t_eval)
        
        results[study_name] = {
            'x': x_eval.cpu().numpy(),
            'alpha_true': dataset['alpha_true'].cpu().numpy(),
            'alpha_pred': alpha_pred.cpu().numpy(),
            'metrics': {
                'mse': torch.mean((alpha_pred.flatten() - dataset['alpha_true'])**2).item()
            }
        }
        
        logger.info(f"Completed {study_name}: MSE = {results[study_name]['metrics']['mse']:.6f}")
    
    # Create comparison visualization
    if not args.no_plots:
        visualizer = ResultsVisualizer(save_dir="results/ablation/plots")
        visualizer.plot_ablation_study(results, "ablation_comparison.png")
        logger.info("Ablation study plots saved to: results/ablation/plots/")
    
    logger.info("Ablation study completed!")
    return results


def visualize_results(results_dir: str, args):
    """Generate comprehensive visualizations from saved results."""
    logger = logging.getLogger('Visualization')
    logger.info(f"Generating visualizations for: {results_dir}")
    
    # Load results
    results_file = os.path.join(results_dir, "results.npz")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    data = np.load(results_file, allow_pickle=True)
    
    # Create visualizer
    plots_dir = os.path.join(results_dir, "comprehensive_plots")
    visualizer = ResultsVisualizer(save_dir=plots_dir)
    
    # Extract data
    experiment_data = {
        'x': data['predictions'].item()['x_grid'],
        'alpha_true': data['predictions'].item()['alpha_true'],
        'alpha_pred': data['predictions'].item()['alpha_pred'],
        'training_history': data['training_history'].item(),
        'metrics': data['final_metrics'].item()
    }
    
    # Generate all plots
    logger.info("Creating comprehensive publication figure...")
    visualizer.create_publication_figure(
        experiment_data, 
        'comprehensive_results.png'
    )
    
    logger.info(f"Comprehensive visualizations saved to: {plots_dir}/")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Variable-Order Fractional PDE Discovery Experiments"
    )
    
    parser.add_argument('--experiment', type=str, 
                       choices=['experiment1', 'experiment2', 'experiment3'],
                       help='Run specific experiment')
    
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study')
    
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate synthetic datasets')
    
    parser.add_argument('--visualize', type=str,
                       help='Generate visualizations for results directory')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    parser.add_argument('--log-file', type=str,
                       help='Log file path')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Computing device')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting Variable-Order Fractional PDE Discovery Experiments")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    try:
        if args.generate_data:
            generate_synthetic_data(args)
        
        elif args.experiment:
            run_single_experiment(args.experiment, args)
        
        elif args.all:
            for exp_name in ['experiment1', 'experiment2', 'experiment3']:
                logger.info(f"\\n{'='*50}")
                logger.info(f"Running {exp_name}")
                logger.info(f"{'='*50}")
                run_single_experiment(exp_name, args)
        
        elif args.ablation:
            run_ablation_study(args)
        
        elif args.visualize:
            visualize_results(args.visualize, args)
        
        else:
            parser.print_help()
            return
        
        logger.info("All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()