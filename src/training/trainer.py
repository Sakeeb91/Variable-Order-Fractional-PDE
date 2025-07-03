"""
Training Orchestrator for Variable-Order Fractional PDE Discovery

This module implements the main training loop that coordinates the dual-network
architecture (solution + order networks) with the composite loss function to
discover spatially varying fractional orders.

Author: Sakeeb Rahman
Date: 2025
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import time
import os
from typing import Dict, List, Optional, Tuple, Callable
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

from ..models.solution_network import create_solution_network
from ..models.order_network import create_order_network
from ..utils.fractional_calculus import create_fractional_operator
from .loss_functions import create_composite_loss


class VariableOrderPDETrainer:
    """
    Main trainer for variable-order fractional PDE discovery.
    
    This class orchestrates the training of both solution and order networks
    simultaneously using physics-informed neural network principles.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize networks
        self.solution_network = create_solution_network(config['solution_network']).to(self.device)
        self.order_network = create_order_network(config['order_network']).to(self.device)
        
        # Initialize fractional operator
        self.fractional_operator = create_fractional_operator(
            {**config['fractional_operator'], 'device': str(self.device)}
        )
        
        # Initialize loss function
        self.loss_function = create_composite_loss(config['loss_function'])
        
        # Initialize optimizer
        self._setup_optimizer()
        
        # Initialize scheduler
        self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = defaultdict(list)
        self.validation_history = defaultdict(list)
        
        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"Solution network parameters: {sum(p.numel() for p in self.solution_network.parameters())}")
        self.logger.info(f"Order network parameters: {sum(p.numel() for p in self.order_network.parameters())}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get('log_file', 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('VariableOrderPDETrainer')
    
    def _setup_optimizer(self):
        """Setup optimizer for both networks."""
        # Combine parameters from both networks
        all_params = list(self.solution_network.parameters()) + list(self.order_network.parameters())
        
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                all_params,
                lr=optimizer_config.get('lr', 1e-3),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type.lower() == 'lbfgs':
            self.optimizer = optim.LBFGS(
                all_params,
                lr=optimizer_config.get('lr', 1.0),
                max_iter=optimizer_config.get('max_iter', 20),
                history_size=optimizer_config.get('history_size', 100)
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                all_params,
                lr=optimizer_config.get('lr', 1e-2),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'plateau')
        
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 50),
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 1000),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None
    
    def generate_collocation_points(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random collocation points for PDE residual evaluation.
        
        Args:
            n_points: Number of collocation points
            
        Returns:
            Tuple of (x_colloc, t_colloc) tensors
        """
        domain_config = self.config.get('domain', {})
        x_bounds = domain_config.get('x_bounds', (0.0, 1.0))
        t_bounds = domain_config.get('t_bounds', (0.0, 1.0))
        
        x_colloc = torch.rand(n_points, device=self.device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
        t_colloc = torch.rand(n_points, device=self.device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        
        return x_colloc, t_colloc
    
    def compute_forcing_term(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute forcing term f(x,t) for the PDE.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            Forcing term values
        """
        # Default: zero forcing (can be overridden for specific problems)
        forcing_config = self.config.get('forcing_term', {'type': 'zero'})
        
        if forcing_config['type'] == 'zero':
            return torch.zeros_like(x).unsqueeze(-1)
        elif forcing_config['type'] == 'sinusoidal':
            freq_x = forcing_config.get('freq_x', 1.0)
            freq_t = forcing_config.get('freq_t', 1.0)
            amplitude = forcing_config.get('amplitude', 1.0)
            return amplitude * torch.sin(np.pi * freq_x * x) * torch.sin(np.pi * freq_t * t)
        else:
            raise ValueError(f"Unknown forcing term type: {forcing_config['type']}")
    
    def forward_pass(self, 
                    x_data: torch.Tensor, 
                    t_data: torch.Tensor,
                    x_colloc: torch.Tensor, 
                    t_colloc: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform forward pass through both networks.
        
        Args:
            x_data: Data point spatial coordinates
            t_data: Data point temporal coordinates
            x_colloc: Collocation point spatial coordinates
            t_colloc: Collocation point temporal coordinates
            
        Returns:
            Dictionary with network outputs and derivatives
        """
        results = {}
        
        # Solution network forward pass on data points
        results['u_pred_data'] = self.solution_network(x_data, t_data)
        
        # Solution network forward pass on collocation points (with gradients)
        u_derivatives = self.solution_network.compute_derivatives(x_colloc, t_colloc)
        results.update(u_derivatives)
        
        # Order network forward pass
        results['alpha_pred'] = self.order_network(x_colloc)
        
        # Order network derivatives (for regularization)
        alpha_derivatives = self.order_network.compute_derivatives(x_colloc)
        results['alpha_derivatives'] = alpha_derivatives
        
        # Compute fractional Laplacian
        results['fractional_laplacian'] = self.fractional_operator.compute_fractional_laplacian(
            x_colloc, results['u'].flatten(), results['alpha_pred'].flatten()
        )
        
        # Compute forcing term
        results['forcing_term'] = self.compute_forcing_term(x_colloc, t_colloc)
        
        return results
    
    def train_step(self, 
                   x_data: torch.Tensor, 
                   t_data: torch.Tensor, 
                   u_data: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            x_data: Observed spatial coordinates
            t_data: Observed temporal coordinates  
            u_data: Observed solution values
            
        Returns:
            Dictionary with loss components
        """
        # Generate collocation points
        n_colloc = self.config.get('n_collocation_points', 1000)
        x_colloc, t_colloc = self.generate_collocation_points(n_colloc)
        
        def closure():
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward_pass(x_data, t_data, x_colloc, t_colloc)
            
            # Compute loss
            loss_dict = self.loss_function.compute(
                u_pred=outputs['u_pred_data'],
                u_observed=u_data,
                u_derivatives=outputs,
                alpha_values=outputs['alpha_pred'],
                fractional_laplacian=outputs['fractional_laplacian'],
                forcing_term=outputs['forcing_term'],
                alpha_derivatives=outputs['alpha_derivatives']
            )
            
            # Backward pass
            loss_dict['total'].backward()
            
            return loss_dict['total']
        
        if isinstance(self.optimizer, optim.LBFGS):
            # LBFGS requires closure
            loss = self.optimizer.step(closure)
            
            # Get loss components for logging
            with torch.no_grad():
                outputs = self.forward_pass(x_data, t_data, x_colloc, t_colloc)
                loss_dict = self.loss_function.compute(
                    u_pred=outputs['u_pred_data'],
                    u_observed=u_data,
                    u_derivatives=outputs,
                    alpha_values=outputs['alpha_pred'],
                    fractional_laplacian=outputs['fractional_laplacian'],
                    forcing_term=outputs['forcing_term'],
                    alpha_derivatives=outputs['alpha_derivatives']
                )
        else:
            # Standard optimizers
            loss_dict = closure()
            self.optimizer.step()
        
        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items() if k != 'weights'}
    
    def validate(self, 
                 x_val: torch.Tensor, 
                 t_val: torch.Tensor, 
                 u_val: torch.Tensor) -> Dict[str, float]:
        """
        Perform validation step.
        
        Args:
            x_val: Validation spatial coordinates
            t_val: Validation temporal coordinates
            u_val: Validation solution values
            
        Returns:
            Dictionary with validation metrics
        """
        self.solution_network.eval()
        self.order_network.eval()
        
        with torch.no_grad():
            # Prediction
            u_pred_val = self.solution_network(x_val, t_val)
            
            # Compute metrics
            mse = torch.mean((u_pred_val - u_val)**2)
            mae = torch.mean(torch.abs(u_pred_val - u_val))
            
            # Relative error
            relative_error = torch.mean(torch.abs(u_pred_val - u_val) / (torch.abs(u_val) + 1e-8))
            
            # Order network statistics
            alpha_pred = self.order_network(x_val)
            alpha_stats = self.order_network.get_statistics(x_val)
        
        self.solution_network.train()
        self.order_network.train()
        
        return {
            'val_mse': mse.item(),
            'val_mae': mae.item(),
            'val_relative_error': relative_error.item(),
            **{f'alpha_{k}': v for k, v in alpha_stats.items()}
        }
    
    def train(self, 
              train_data: Dict[str, torch.Tensor],
              val_data: Optional[Dict[str, torch.Tensor]] = None,
              n_epochs: int = 1000) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_data: Training dataset {'x', 't', 'u'}
            val_data: Optional validation dataset
            n_epochs: Number of training epochs
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {n_epochs} epochs...")
        
        # Move data to device
        x_train = train_data['x'].to(self.device)
        t_train = train_data['t'].to(self.device)
        u_train = train_data['u'].to(self.device)
        
        if val_data:
            x_val = val_data['x'].to(self.device)
            t_val = val_data['t'].to(self.device)
            u_val = val_data['u'].to(self.device)
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            
            # Training step
            loss_dict = self.train_step(x_train, t_train, u_train)
            
            # Update training history
            for key, value in loss_dict.items():
                self.training_history[key].append(value)
            
            # Validation step
            if val_data and epoch % self.config.get('val_freq', 10) == 0:
                val_metrics = self.validate(x_val, t_val, u_val)
                for key, value in val_metrics.items():
                    self.validation_history[key].append(value)
                
                # Check for best model
                if val_metrics['val_mse'] < self.best_loss:
                    self.best_loss = val_metrics['val_mse']
                    self.save_checkpoint('best_model.pth')
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss_dict['total'])
                else:
                    self.scheduler.step()
            
            # Logging
            if epoch % self.config.get('log_freq', 100) == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch:6d} | "
                    f"Total: {loss_dict['total']:.6f} | "
                    f"Data: {loss_dict['data']:.6f} | "
                    f"Residual: {loss_dict['residual']:.6f} | "
                    f"Reg: {loss_dict['regularization']:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Checkpointing
            if epoch % self.config.get('checkpoint_freq', 500) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        
        return dict(self.training_history)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'solution_network_state_dict': self.solution_network.state_dict(),
            'order_network_state_dict': self.order_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'training_history': dict(self.training_history),
            'validation_history': dict(self.validation_history),
            'config': self.config
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.solution_network.load_state_dict(checkpoint['solution_network_state_dict'])
        self.order_network.load_state_dict(checkpoint['order_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_loss = checkpoint['best_loss']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        self.validation_history = defaultdict(list, checkpoint['validation_history'])
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
    
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with trained networks.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            Tuple of (predicted_solution, predicted_alpha)
        """
        self.solution_network.eval()
        self.order_network.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            t = t.to(self.device)
            
            u_pred = self.solution_network(x, t)
            alpha_pred = self.order_network(x)
        
        return u_pred, alpha_pred
    
    def get_training_summary(self) -> Dict[str, float]:
        """Get training summary statistics."""
        if not self.training_history:
            return {}
        
        summary = {}
        for key, values in self.training_history.items():
            if values:
                summary[f'{key}_final'] = values[-1]
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_mean'] = np.mean(values[-100:])  # Last 100 epochs
        
        return summary


def create_trainer(config: Dict) -> VariableOrderPDETrainer:
    """
    Factory function to create trainer.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured trainer instance
    """
    return VariableOrderPDETrainer(config)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Variable Order PDE Trainer...")
    
    # Example configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'solution_network': {
            'type': 'basic',
            'hidden_layers': 4,
            'neurons_per_layer': 50,
            'activation': 'tanh'
        },
        'order_network': {
            'type': 'basic',
            'hidden_layers': 3,
            'neurons_per_layer': 30,
            'activation': 'tanh',
            'alpha_bounds': (1.0, 2.0)
        },
        'fractional_operator': {
            'domain_bounds': (0.0, 1.0),
            'n_grid': 101
        },
        'loss_function': {
            'data_loss': {'weight': 1.0},
            'residual_loss': {'weight': 1.0},
            'regularization': {'weight': 0.01}
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3
        },
        'scheduler': {
            'type': 'plateau',
            'patience': 50
        },
        'n_collocation_points': 500,
        'domain': {
            'x_bounds': (0.0, 1.0),
            't_bounds': (0.0, 1.0)
        },
        'log_freq': 10,
        'val_freq': 5
    }
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Create dummy training data
    n_data = 100
    x_train = torch.rand(n_data)
    t_train = torch.rand(n_data)
    u_train = torch.sin(np.pi * x_train) * torch.exp(-t_train)
    
    train_data = {
        'x': x_train,
        't': t_train,
        'u': u_train.unsqueeze(-1)
    }
    
    # Test short training run
    print("Running short training test...")
    history = trainer.train(train_data, n_epochs=50)
    
    print(f"Training completed. Final losses:")
    for key, values in history.items():
        if values:
            print(f"  {key}: {values[-1]:.6f}")
    
    # Test prediction
    x_test = torch.linspace(0, 1, 20)
    t_test = torch.full_like(x_test, 0.5)
    u_pred, alpha_pred = trainer.predict(x_test, t_test)
    
    print(f"Prediction shapes: u={u_pred.shape}, alpha={alpha_pred.shape}")
    print(f"Alpha range: [{float(torch.min(alpha_pred)):.3f}, {float(torch.max(alpha_pred)):.3f}]")
    
    print("All tests passed!")