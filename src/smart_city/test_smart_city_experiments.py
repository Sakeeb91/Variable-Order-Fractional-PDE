"""
Test script for Smart City Experiment System

Simple validation of the experiment configuration and training pipeline.
"""

import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_city_experiments import SmartCityExperimentConfig, SmartCityTrainer, SmartCityExperimentSuite


def test_experiment_config():
    """Test experiment configuration creation."""
    print("Testing experiment configuration...")
    
    config = SmartCityExperimentConfig(
        experiment_name='test_config',
        description='Test configuration',
        tags=['test'],
        max_iterations=100,  # Small for testing
        domain_size=(21, 21, 5)  # Small domain
    )
    
    print(f"✓ Experiment config created: {config.experiment_name}")
    print(f"  Domain size: {config.domain_size}")
    print(f"  Max iterations: {config.max_iterations}")
    
    return config


def test_training_setup():
    """Test training setup without full training."""
    print("\nTesting training setup...")
    
    config = SmartCityExperimentConfig(
        experiment_name='test_training_setup',
        description='Test training setup',
        tags=['test', 'setup'],
        max_iterations=10,  # Very small for quick test
        domain_size=(11, 11, 3),  # Very small domain
        validation_frequency=5
    )
    
    try:
        trainer = SmartCityTrainer(config)
        print(f"✓ Trainer initialized successfully")
        print(f"  Training points: {len(trainer.train_indices)}")
        print(f"  Validation points: {len(trainer.val_indices)}")
        
        # Test single training step
        batch_data = trainer._get_batch_data(trainer.train_indices[:10])
        loss_dict = trainer.training_step(batch_data)
        
        print(f"✓ Single training step completed")
        print(f"  Total loss: {loss_dict['total']:.6f}")
        print(f"  Data loss: {loss_dict['data']:.6f}")
        print(f"  Residual loss: {loss_dict['residual']:.6f}")
        
        return trainer
        
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        return None


def test_experiment_suite():
    """Test experiment suite creation."""
    print("\nTesting experiment suite...")
    
    suite = SmartCityExperimentSuite()
    
    # Test baseline experiment creation
    baseline = suite.create_baseline_experiment()
    print(f"✓ Baseline experiment created: {baseline.experiment_name}")
    
    # Test specialized experiments
    heat_island = suite.create_heat_island_experiment()
    print(f"✓ Heat island experiment created: {heat_island.experiment_name}")
    
    air_quality = suite.create_air_quality_experiment()
    print(f"✓ Air quality experiment created: {air_quality.experiment_name}")
    
    return suite


def test_mini_experiment():
    """Run a very small experiment for full pipeline test."""
    print("\nRunning mini experiment...")
    
    config = SmartCityExperimentConfig(
        experiment_name='mini_test',
        description='Mini test experiment',
        tags=['test', 'mini'],
        max_iterations=20,
        domain_size=(11, 11, 3),
        batch_size=50,
        validation_frequency=10,
        early_stopping_patience=50,
        output_dir='test_experiments'
    )
    
    try:
        trainer = SmartCityTrainer(config)
        results = trainer.train()
        
        print(f"✓ Mini experiment completed successfully")
        print(f"  Final loss: {results['final_loss']:.6f}")
        print(f"  Best loss: {results['best_loss']:.6f}")
        print(f"  Total iterations: {results['total_iterations']}")
        print(f"  Training time: {results['training_time']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"✗ Mini experiment failed: {e}")
        return None


if __name__ == "__main__":
    print("Smart City Experiment System Test")
    print("=" * 50)
    
    # Run tests
    config = test_experiment_config()
    trainer = test_training_setup()
    suite = test_experiment_suite()
    
    if trainer is not None:
        results = test_mini_experiment()
        
        if results is not None:
            print("\n" + "=" * 50)
            print("All tests passed successfully! ✓")
            print("Smart city experiment system is ready for use.")
        else:
            print("\n" + "=" * 50)
            print("Some tests failed. Check the error messages above.")
    else:
        print("\n" + "=" * 50)
        print("Training setup failed. Cannot proceed with full test.")