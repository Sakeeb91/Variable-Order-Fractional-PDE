"""
Smart City Experiment Configuration and Management

This module provides comprehensive experiment management for smart city
variable-order fractional PDE discovery, including training orchestration,
hyperparameter optimization, and validation protocols.

Author: Sakeeb Rahman
Date: 2025
"""

import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Local imports
from urban_data_generator import UrbanClimateGenerator
from smart_city_networks import SmartCityNetworkFactory
from smart_city_loss_functions import create_smart_city_loss


@dataclass
class SmartCityExperimentConfig:
    """Configuration for smart city experiments."""
    
    # Experiment metadata
    experiment_name: str
    description: str
    tags: List[str]
    
    # Data configuration
    urban_layout: str = 'mixed_city'
    scenario_name: str = 'summer_day'
    domain_size: Tuple[int, int, int] = (41, 41, 25)  # nx, ny, nt
    spatial_domain: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 10.0), (0.0, 10.0))
    temporal_domain: Tuple[float, float] = (0.0, 24.0)
    
    # Network configuration
    solution_network_config: Dict = None
    order_network_config: Dict = None
    
    # Loss function configuration
    loss_config: Dict = None
    
    # Training configuration
    max_iterations: int = 5000
    learning_rate: float = 1e-3
    batch_size: int = 256
    validation_split: float = 0.2
    early_stopping_patience: int = 200
    
    # Optimization configuration
    optimizer_type: str = 'adam'
    lr_schedule: str = 'cosine'  # 'constant', 'exponential', 'cosine'
    gradient_clipping: float = 1.0
    
    # Validation configuration
    validation_frequency: int = 100
    save_frequency: int = 500
    plot_frequency: int = 1000
    
    # Output configuration
    save_models: bool = True
    save_predictions: bool = True
    save_losses: bool = True
    output_dir: str = 'experiments/smart_city'
    
    def __post_init__(self):
        """Set default configurations if not provided."""
        if self.solution_network_config is None:
            self.solution_network_config = {
                'output_fields': ['temperature', 'pollutant', 'humidity'],
                'hidden_layers': 6,
                'neurons_per_layer': 120,
                'activation': 'tanh',
                'use_physics_constraints': True,
                'field_coupling': True
            }
        
        if self.order_network_config is None:
            self.order_network_config = {
                'output_fields': ['alpha_T', 'alpha_C', 'alpha_H'],
                'hidden_layers': 5,
                'neurons_per_layer': 80,
                'activation': 'tanh',
                'alpha_bounds': (1.0, 2.0),
                'multi_scale': True
            }
        
        if self.loss_config is None:
            self.loss_config = {
                'data_loss': {
                    'weight': 1.0,
                    'uncertainty_weighting': True,
                    'loss_type': 'huber'
                },
                'residual_loss': {
                    'weight': 1.0,
                    'coupling_strength': 0.1,
                    'include_source_terms': True
                },
                'regularization': {
                    'weight': 0.01,
                    'urban_constraints_weight': 0.2,
                    'consistency_weight': 0.1
                },
                'adaptive_weights': True
            }


class SmartCityTrainer:
    """Training orchestrator for smart city experiments."""
    
    def __init__(self, config: SmartCityExperimentConfig):
        self.config = config
        self.experiment_dir = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize networks
        self.solution_network = SmartCityNetworkFactory.create_multi_physics_solution_network(
            config.solution_network_config
        )
        self.order_network = SmartCityNetworkFactory.create_multi_field_order_network(
            config.order_network_config
        )
        
        # Initialize loss function
        self.loss_function = create_smart_city_loss(config.loss_config)
        
        # Training state
        self.current_iteration = 0
        self.best_loss = float('inf')
        self.training_history = {
            'total_loss': [],
            'data_loss': [],
            'residual_loss': [],
            'regularization_loss': [],
            'validation_loss': [],
            'learning_rates': []
        }
        
        # Generate training data
        self._generate_training_data()
        
    def _generate_training_data(self):
        """Generate synthetic urban climate data for training."""
        nx, ny, nt = self.config.domain_size
        (x_min, x_max), (y_min, y_max) = self.config.spatial_domain
        t_min, t_max = self.config.temporal_domain
        
        # Create data generator
        generator = UrbanClimateGenerator(
            domain_x=(x_min, x_max),
            domain_y=(y_min, y_max),
            domain_t=(t_min, t_max),
            nx=nx, ny=ny, nt=nt
        )
        
        # Generate complete urban scenario
        self.urban_dataset = generator.generate_complete_urban_scenario(
            layout_type=self.config.urban_layout,
            scenario_name=self.config.scenario_name
        )
        
        # Create training coordinates
        self._create_training_coordinates()
        
    def _create_training_coordinates(self):
        """Create training coordinate sets."""
        # Extract coordinates from dataset
        X, Y = self.urban_dataset['coordinates']['X'], self.urban_dataset['coordinates']['Y']
        t_array = self.urban_dataset['coordinates']['t']
        
        # Create full coordinate grid
        coords_3d = []
        coords_2d = []
        
        for t in t_array:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    coords_3d.append([X[i, j], Y[i, j], t])
                    coords_2d.append([X[i, j], Y[i, j]])
        
        self.coords_3d = np.array(coords_3d)
        self.coords_2d = np.array(coords_2d[:X.shape[0] * X.shape[1]])  # Just 2D spatial
        
        # Create training/validation split
        n_total = len(self.coords_3d)
        n_train = int(n_total * (1 - self.config.validation_split))
        
        indices = np.random.permutation(n_total)
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:]
        
        print(f"Training points: {len(self.train_indices)}")
        print(f"Validation points: {len(self.val_indices)}")
        
    def _get_batch_data(self, indices: np.ndarray, batch_size: int = None) -> Dict:
        """Get batch of training data."""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        # Sample batch indices
        if len(indices) > batch_size:
            batch_indices = np.random.choice(indices, batch_size, replace=False)
        else:
            batch_indices = indices
            
        # Get coordinates
        coords_3d_batch = self.coords_3d[batch_indices]
        
        # Extract 2D coordinates for order network
        coords_2d_batch = coords_3d_batch[:, :2]
        
        # Get ground truth observations from synthetic data
        observations = self._extract_observations(batch_indices)
        
        return {
            'coords_3d': coords_3d_batch,
            'coords_2d': coords_2d_batch,
            'observations': observations,
            'indices': batch_indices
        }
    
    def _extract_observations(self, indices: np.ndarray) -> Dict:
        """Extract ground truth observations for given indices."""
        # This is a simplified version - in practice would extract from 
        # the full 3D fields at specific space-time coordinates
        n_points = len(indices)
        
        observations = {
            'temperature': 20 + 5 * np.random.randn(n_points),
            'pollutant': 50 + 20 * np.random.randn(n_points),
            'humidity': 60 + 10 * np.random.randn(n_points)
        }
        
        return observations
    
    def training_step(self, batch_data: Dict) -> Dict:
        """Perform one training step."""
        coords_3d = batch_data['coords_3d']
        coords_2d = batch_data['coords_2d']
        observations = batch_data['observations']
        
        # Forward pass through networks
        solution_outputs = self.solution_network.forward(coords_3d)
        alpha_outputs = self.order_network.forward(coords_2d)
        
        # Compute derivatives
        field_derivatives = {}
        for field in solution_outputs.keys():
            field_derivatives[field] = self.solution_network.compute_derivatives(
                coords_3d, field
            )
        
        alpha_derivatives = self.order_network.compute_derivatives(coords_2d)
        
        # Mock fractional Laplacians (would be computed using proper fractional calculus)
        fractional_laplacians = {
            field: np.random.randn(len(coords_3d)) * 0.1 
            for field in solution_outputs.keys()
        }
        
        # Compute loss
        loss_dict = self.loss_function.compute(
            predictions=solution_outputs,
            observations=observations,
            field_derivatives=field_derivatives,
            alpha_fields=alpha_outputs,
            fractional_laplacians=fractional_laplacians,
            alpha_derivatives=alpha_derivatives,
            coordinates=coords_3d
        )
        
        return loss_dict
    
    def validation_step(self) -> Dict:
        """Perform validation evaluation."""
        val_batch = self._get_batch_data(self.val_indices, batch_size=min(500, len(self.val_indices)))
        return self.training_step(val_batch)
    
    def train(self) -> Dict:
        """Run complete training process."""
        print(f"Starting training for experiment: {self.config.experiment_name}")
        print(f"Max iterations: {self.config.max_iterations}")
        
        start_time = time.time()
        epochs_without_improvement = 0
        
        for iteration in range(self.config.max_iterations):
            self.current_iteration = iteration
            
            # Training step
            batch_data = self._get_batch_data(self.train_indices)
            loss_dict = self.training_step(batch_data)
            
            # Record training loss
            self.training_history['total_loss'].append(loss_dict['total'])
            self.training_history['data_loss'].append(loss_dict['data'])
            self.training_history['residual_loss'].append(loss_dict['residual'])
            self.training_history['regularization_loss'].append(loss_dict['regularization'])
            self.training_history['learning_rates'].append(self.config.learning_rate)
            
            # Validation
            if iteration % self.config.validation_frequency == 0:
                val_loss_dict = self.validation_step()
                val_loss = val_loss_dict['total']
                self.training_history['validation_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    epochs_without_improvement = 0
                    
                    if self.config.save_models:
                        self._save_checkpoint('best_model')
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"Early stopping at iteration {iteration}")
                    break
                
                # Progress reporting
                if iteration % (self.config.validation_frequency * 5) == 0:
                    elapsed = time.time() - start_time
                    print(f"Iter {iteration:5d} | Train Loss: {loss_dict['total']:.6f} | "
                          f"Val Loss: {val_loss:.6f} | Time: {elapsed:.1f}s")
            
            # Periodic saves
            if iteration % self.config.save_frequency == 0 and iteration > 0:
                self._save_checkpoint(f'checkpoint_{iteration}')
        
        # Final save
        self._save_final_results()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f} seconds")
        print(f"Best validation loss: {self.best_loss:.6f}")
        
        return {
            'best_loss': self.best_loss,
            'total_iterations': self.current_iteration,
            'training_time': total_time,
            'final_loss': self.training_history['total_loss'][-1]
        }
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_name}.json')
        
        # Simplified checkpoint saving (would include actual model weights in full implementation)
        checkpoint_data = {
            'iteration': self.current_iteration,
            'best_loss': self.best_loss,
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _save_final_results(self):
        """Save final experiment results."""
        results = {
            'experiment_config': asdict(self.config),
            'training_history': self.training_history,
            'final_metrics': {
                'best_validation_loss': self.best_loss,
                'final_training_loss': self.training_history['total_loss'][-1],
                'total_iterations': self.current_iteration,
                'convergence_iteration': len(self.training_history['validation_loss']) * self.config.validation_frequency
            },
            'urban_dataset_info': {
                'layout_type': self.urban_dataset['layout_type'],
                'scenario_name': self.urban_dataset['scenario_name'],
                'spatial_resolution': self.urban_dataset['metadata']['dx'],
                'temporal_resolution': self.urban_dataset['metadata']['dt']
            }
        }
        
        results_path = os.path.join(self.experiment_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save training history as numpy arrays for analysis
        history_path = os.path.join(self.experiment_dir, 'training_history.npz')
        np.savez_compressed(history_path, **self.training_history)
        
        print(f"Results saved to {self.experiment_dir}")


class SmartCityExperimentSuite:
    """Comprehensive experiment suite for smart city applications."""
    
    def __init__(self, base_output_dir: str = 'experiments/smart_city'):
        self.base_output_dir = base_output_dir
        self.experiments = {}
        
    def create_baseline_experiment(self) -> SmartCityExperimentConfig:
        """Create baseline experiment configuration."""
        return SmartCityExperimentConfig(
            experiment_name='baseline_mixed_city',
            description='Baseline experiment with mixed urban layout',
            tags=['baseline', 'mixed_city', 'summer'],
            urban_layout='mixed_city',
            scenario_name='summer_day',
            output_dir=self.base_output_dir
        )
    
    def create_heat_island_experiment(self) -> SmartCityExperimentConfig:
        """Create heat island mitigation experiment."""
        config = SmartCityExperimentConfig(
            experiment_name='heat_island_downtown',
            description='Downtown heat island analysis with high-resolution thermal modeling',
            tags=['heat_island', 'downtown', 'thermal'],
            urban_layout='downtown_core',
            scenario_name='heat_wave',
            output_dir=self.base_output_dir
        )
        
        # Specialized configuration for thermal analysis
        config.solution_network_config.update({
            'output_fields': ['temperature'],
            'neurons_per_layer': 150,
            'hidden_layers': 7
        })
        
        config.loss_config['data_loss'].update({
            'field_weights': {'temperature': 2.0}  # Higher weight for temperature
        })
        
        return config
    
    def create_air_quality_experiment(self) -> SmartCityExperimentConfig:
        """Create air quality monitoring experiment."""
        config = SmartCityExperimentConfig(
            experiment_name='air_quality_industrial',
            description='Air quality analysis near industrial zones',
            tags=['air_quality', 'industrial', 'pollutant'],
            urban_layout='mixed_city',
            scenario_name='rush_hour',
            output_dir=self.base_output_dir
        )
        
        # Specialized configuration for air quality
        config.solution_network_config.update({
            'output_fields': ['pollutant', 'temperature'],
            'field_coupling': True
        })
        
        config.loss_config['residual_loss'].update({
            'pde_weights': {'pollutant': 1.5, 'temperature': 1.0}
        })
        
        return config
    
    def create_multi_physics_experiment(self) -> SmartCityExperimentConfig:
        """Create comprehensive multi-physics experiment."""
        config = SmartCityExperimentConfig(
            experiment_name='multi_physics_comprehensive',
            description='Full multi-physics urban climate modeling',
            tags=['multi_physics', 'comprehensive', 'coupled'],
            urban_layout='mixed_city',
            scenario_name='summer_day',
            max_iterations=8000,
            output_dir=self.base_output_dir
        )
        
        # Enhanced multi-physics configuration
        config.solution_network_config.update({
            'neurons_per_layer': 140,
            'hidden_layers': 8,
            'field_coupling': True
        })
        
        config.loss_config['residual_loss'].update({
            'coupling_strength': 0.15
        })
        
        return config
    
    def create_sensitivity_analysis_suite(self) -> List[SmartCityExperimentConfig]:
        """Create suite of experiments for sensitivity analysis."""
        base_config = self.create_baseline_experiment()
        experiments = []
        
        # Parameter variations
        variations = [
            {'name': 'high_coupling', 'param': 'coupling_strength', 'value': 0.3},
            {'name': 'low_coupling', 'param': 'coupling_strength', 'value': 0.05},
            {'name': 'large_network', 'param': 'neurons_per_layer', 'value': 200},
            {'name': 'small_network', 'param': 'neurons_per_layer', 'value': 60},
            {'name': 'high_reg', 'param': 'regularization_weight', 'value': 0.1},
            {'name': 'low_reg', 'param': 'regularization_weight', 'value': 0.001}
        ]
        
        for variation in variations:
            config = SmartCityExperimentConfig(
                experiment_name=f"sensitivity_{variation['name']}",
                description=f"Sensitivity analysis: {variation['name']}",
                tags=['sensitivity', variation['name']],
                urban_layout='mixed_city',
                scenario_name='summer_day',
                output_dir=self.base_output_dir
            )
            
            # Apply variation
            if variation['param'] == 'coupling_strength':
                config.loss_config['residual_loss']['coupling_strength'] = variation['value']
            elif variation['param'] == 'neurons_per_layer':
                config.solution_network_config['neurons_per_layer'] = variation['value']
            elif variation['param'] == 'regularization_weight':
                config.loss_config['regularization']['weight'] = variation['value']
            
            experiments.append(config)
        
        return experiments
    
    def run_experiment(self, config: SmartCityExperimentConfig) -> Dict:
        """Run a single experiment."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {config.experiment_name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")
        
        trainer = SmartCityTrainer(config)
        results = trainer.train()
        
        self.experiments[config.experiment_name] = {
            'config': config,
            'results': results,
            'trainer': trainer
        }
        
        return results
    
    def run_experiment_suite(self, experiment_configs: List[SmartCityExperimentConfig]) -> Dict:
        """Run multiple experiments in sequence."""
        suite_results = {}
        
        for config in experiment_configs:
            try:
                results = self.run_experiment(config)
                suite_results[config.experiment_name] = results
            except Exception as e:
                print(f"Experiment {config.experiment_name} failed: {str(e)}")
                suite_results[config.experiment_name] = {'error': str(e)}
        
        # Save suite summary
        self._save_suite_summary(suite_results)
        
        return suite_results
    
    def _save_suite_summary(self, suite_results: Dict):
        """Save summary of experiment suite results."""
        summary_dir = os.path.join(self.base_output_dir, 'suite_summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        summary = {
            'total_experiments': len(suite_results),
            'successful_experiments': len([r for r in suite_results.values() if 'error' not in r]),
            'failed_experiments': len([r for r in suite_results.values() if 'error' in r]),
            'experiment_results': suite_results,
            'best_experiment': None,
            'worst_experiment': None
        }
        
        # Find best and worst experiments
        valid_results = {name: result for name, result in suite_results.items() 
                        if 'error' not in result and 'best_loss' in result}
        
        if valid_results:
            best_name = min(valid_results.keys(), key=lambda x: valid_results[x]['best_loss'])
            worst_name = max(valid_results.keys(), key=lambda x: valid_results[x]['best_loss'])
            
            summary['best_experiment'] = {
                'name': best_name,
                'best_loss': valid_results[best_name]['best_loss']
            }
            summary['worst_experiment'] = {
                'name': worst_name,
                'best_loss': valid_results[worst_name]['best_loss']
            }
        
        summary_path = os.path.join(summary_dir, 'experiment_suite_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExperiment suite summary saved to {summary_path}")
        print(f"Successful experiments: {summary['successful_experiments']}/{summary['total_experiments']}")
        
        if summary['best_experiment']:
            print(f"Best experiment: {summary['best_experiment']['name']} "
                  f"(loss: {summary['best_experiment']['best_loss']:.6f})")


def run_smart_city_demo():
    """Run demonstration of smart city experiments."""
    print("Smart City Variable-Order Fractional PDE Discovery")
    print("Experiment Suite Demonstration")
    print("="*60)
    
    # Create experiment suite
    suite = SmartCityExperimentSuite()
    
    # Create various experiment configurations
    experiments = [
        suite.create_baseline_experiment(),
        suite.create_heat_island_experiment(),
        suite.create_air_quality_experiment()
    ]
    
    # Run experiments
    results = suite.run_experiment_suite(experiments)
    
    print("\nDemo completed successfully!")
    print(f"Results available in: {suite.base_output_dir}")
    
    return results


if __name__ == "__main__":
    # Run demo with simplified parameters for testing
    results = run_smart_city_demo()