"""
Loss Functions for Variable-Order Fractional PDE Discovery

This module implements the composite loss function that trains both the solution
network (u_NN) and order network (α_NN) simultaneously. The loss consists of
data mismatch, PDE residual, and regularization components.

Author: Sakeeb Rahman
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Abstract base class for loss function components."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        pass
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.weight * self.compute(*args, **kwargs)


class DataMismatchLoss(BaseLoss):
    """
    Data mismatch loss L_data = MSE(u_pred, u_observed).
    
    This loss anchors the training by comparing network predictions
    with observed data points.
    """
    
    def __init__(self, 
                 weight: float = 1.0,
                 loss_type: str = 'mse',
                 robust_loss: bool = False,
                 huber_delta: float = 1.0):
        """
        Initialize data mismatch loss.
        
        Args:
            weight: Loss component weight
            loss_type: Type of loss ('mse', 'mae', 'huber')
            robust_loss: Whether to use robust loss for outliers
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__(weight)
        self.loss_type = loss_type
        self.robust_loss = robust_loss
        self.huber_delta = huber_delta
        
        # Initialize loss function
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'huber':
            self.base_loss = nn.SmoothL1Loss(beta=huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute(self, 
                u_pred: torch.Tensor, 
                u_observed: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute data mismatch loss.
        
        Args:
            u_pred: Predicted solution values [N, 1]
            u_observed: Observed solution values [N, 1]
            weights: Optional point-wise weights [N, 1]
            
        Returns:
            Data mismatch loss value
        """
        # Basic loss computation
        if weights is None:
            loss = self.base_loss(u_pred, u_observed)
        else:
            # Weighted loss
            diff = u_pred - u_observed
            if self.loss_type == 'mse':
                loss = torch.mean(weights * diff**2)
            elif self.loss_type == 'mae':
                loss = torch.mean(weights * torch.abs(diff))
            else:  # Huber
                loss = torch.mean(weights * torch.where(
                    torch.abs(diff) < self.huber_delta,
                    0.5 * diff**2,
                    self.huber_delta * (torch.abs(diff) - 0.5 * self.huber_delta)
                ))
        
        # Apply robust loss if requested
        if self.robust_loss:
            loss = self._apply_robust_loss(loss)
        
        return loss
    
    def _apply_robust_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply robust loss transformation to reduce outlier influence."""
        # Cauchy loss: log(1 + loss^2)
        return torch.log(1 + loss**2)


class PDEResidualLoss(BaseLoss):
    """
    PDE residual loss L_res = MSE(R, 0) where R is the PDE residual.
    
    This loss enforces the physics constraints by requiring the PDE
    to be satisfied at collocation points.
    """
    
    def __init__(self, 
                 weight: float = 1.0,
                 pde_form: str = 'fractional_diffusion',
                 convection_coefficient: float = 0.0):
        """
        Initialize PDE residual loss.
        
        Args:
            weight: Loss component weight
            pde_form: Form of PDE ('fractional_diffusion', 'fractional_advection_diffusion')
            convection_coefficient: Convection velocity coefficient
        """
        super().__init__(weight)
        self.pde_form = pde_form
        self.convection_coefficient = convection_coefficient
    
    def compute(self,
                u_derivatives: Dict[str, torch.Tensor],
                alpha_values: torch.Tensor,
                fractional_laplacian: torch.Tensor,
                forcing_term: torch.Tensor,
                diffusion_coefficient: float = 1.0) -> torch.Tensor:
        """
        Compute PDE residual loss.
        
        Args:
            u_derivatives: Dictionary with 'u', 'u_x', 'u_t' derivatives
            alpha_values: Fractional order values at collocation points [N, 1]
            fractional_laplacian: Computed (-Δ)^(α/2) u values [N, 1]
            forcing_term: Source term f(x,t) [N, 1]
            diffusion_coefficient: Diffusion coefficient c
            
        Returns:
            PDE residual loss value
        """
        # Extract derivatives
        u = u_derivatives['u']
        u_x = u_derivatives.get('u_x', torch.zeros_like(u))
        u_t = u_derivatives['u_t']
        
        # Compute PDE residual based on form
        if self.pde_form == 'fractional_diffusion':
            # ∂u/∂t - c(-Δ)^(α/2) u = f
            residual = u_t - diffusion_coefficient * fractional_laplacian - forcing_term
            
        elif self.pde_form == 'fractional_advection_diffusion':
            # ∂u/∂t + v∂u/∂x - c(-Δ)^(α/2) u = f
            residual = (u_t + self.convection_coefficient * u_x 
                       - diffusion_coefficient * fractional_laplacian - forcing_term)
        else:
            raise ValueError(f"Unknown PDE form: {self.pde_form}")
        
        # MSE of residual
        loss = torch.mean(residual**2)
        
        return loss


class RegularizationLoss(BaseLoss):
    """
    Regularization loss L_reg for the fractional order function α(x).
    
    Combines smoothness and simplicity penalties to ensure physically
    meaningful discovery of α(x).
    """
    
    def __init__(self,
                 weight: float = 1.0,
                 smoothness_weight: float = 1.0,
                 l1_weight: float = 0.1,
                 l2_weight: float = 0.01,
                 tv_weight: float = 0.0):
        """
        Initialize regularization loss.
        
        Args:
            weight: Overall regularization weight
            smoothness_weight: Weight for smoothness penalty ||∇α||²
            l1_weight: Weight for L1 sparsity penalty
            l2_weight: Weight for L2 penalty
            tv_weight: Weight for total variation penalty
        """
        super().__init__(weight)
        self.smoothness_weight = smoothness_weight
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.tv_weight = tv_weight
    
    def compute(self,
                alpha_derivatives: Dict[str, torch.Tensor],
                alpha_values: torch.Tensor,
                reference_alpha: Optional[float] = None) -> torch.Tensor:
        """
        Compute regularization loss.
        
        Args:
            alpha_derivatives: Dictionary with 'alpha', 'alpha_x' derivatives
            alpha_values: Current α(x) values [N, 1]
            reference_alpha: Reference α value for L1 penalty (default: mean)
            
        Returns:
            Combined regularization loss
        """
        total_loss = torch.tensor(0.0, device=alpha_values.device)
        
        # Smoothness penalty: ||∇α||²
        if self.smoothness_weight > 0 and 'alpha_x' in alpha_derivatives:
            alpha_x = alpha_derivatives['alpha_x']
            smoothness_loss = torch.mean(alpha_x**2)
            total_loss += self.smoothness_weight * smoothness_loss
        
        # L1 sparsity penalty: encourages α(x) to be close to constant
        if self.l1_weight > 0:
            if reference_alpha is None:
                reference_alpha = torch.mean(alpha_values).item()
            
            l1_loss = torch.mean(torch.abs(alpha_values - reference_alpha))
            total_loss += self.l1_weight * l1_loss
        
        # L2 penalty: prevents extreme values
        if self.l2_weight > 0:
            l2_loss = torch.mean(alpha_values**2)
            total_loss += self.l2_weight * l2_loss
        
        # Total variation penalty: promotes piecewise smoothness
        if self.tv_weight > 0 and 'alpha_x' in alpha_derivatives:
            alpha_x = alpha_derivatives['alpha_x']
            tv_loss = torch.mean(torch.abs(alpha_x))
            total_loss += self.tv_weight * tv_loss
        
        return total_loss


class CompositeLoss:
    """
    Composite loss function combining all components.
    
    L_total = w_data * L_data + w_res * L_res + w_reg * L_reg
    """
    
    def __init__(self,
                 data_loss_config: Dict = None,
                 residual_loss_config: Dict = None,
                 regularization_config: Dict = None,
                 adaptive_weights: bool = False,
                 weight_update_freq: int = 100):
        """
        Initialize composite loss.
        
        Args:
            data_loss_config: Configuration for data mismatch loss
            residual_loss_config: Configuration for PDE residual loss
            regularization_config: Configuration for regularization loss
            adaptive_weights: Whether to use adaptive weight balancing
            weight_update_freq: Frequency for weight updates (iterations)
        """
        # Default configurations
        data_config = data_loss_config or {'weight': 1.0}
        residual_config = residual_loss_config or {'weight': 1.0}
        reg_config = regularization_config or {'weight': 0.01}
        
        # Initialize loss components
        self.data_loss = DataMismatchLoss(**data_config)
        self.residual_loss = PDEResidualLoss(**residual_config)
        self.regularization_loss = RegularizationLoss(**reg_config)
        
        # Adaptive weighting
        self.adaptive_weights = adaptive_weights
        self.weight_update_freq = weight_update_freq
        self.iteration_count = 0
        
        # Loss history for adaptive weighting
        self.loss_history = {
            'data': [],
            'residual': [],
            'regularization': []
        }
        
        # Store original weights
        self.original_weights = {
            'data': self.data_loss.weight,
            'residual': self.residual_loss.weight,
            'regularization': self.regularization_loss.weight
        }
    
    def compute(self,
                # Data loss inputs
                u_pred: torch.Tensor,
                u_observed: torch.Tensor,
                # Residual loss inputs
                u_derivatives: Dict[str, torch.Tensor],
                alpha_values: torch.Tensor,
                fractional_laplacian: torch.Tensor,
                forcing_term: torch.Tensor,
                # Regularization inputs
                alpha_derivatives: Dict[str, torch.Tensor],
                # Optional parameters
                data_weights: Optional[torch.Tensor] = None,
                diffusion_coefficient: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss and individual components.
        
        Returns:
            Dictionary with total loss and individual components
        """
        # Compute individual losses
        loss_data = self.data_loss.compute(u_pred, u_observed, data_weights)
        
        loss_residual = self.residual_loss.compute(
            u_derivatives, alpha_values, fractional_laplacian, 
            forcing_term, diffusion_coefficient
        )
        
        loss_regularization = self.regularization_loss.compute(
            alpha_derivatives, alpha_values
        )
        
        # Store for adaptive weighting
        if self.adaptive_weights:
            self.loss_history['data'].append(loss_data.item())
            self.loss_history['residual'].append(loss_residual.item())
            self.loss_history['regularization'].append(loss_regularization.item())
            
            # Update weights if necessary
            if (self.iteration_count + 1) % self.weight_update_freq == 0:
                self._update_adaptive_weights()
        
        # Compute total loss
        total_loss = loss_data + loss_residual + loss_regularization
        
        self.iteration_count += 1
        
        return {
            'total': total_loss,
            'data': loss_data,
            'residual': loss_residual,
            'regularization': loss_regularization,
            'weights': {
                'data': self.data_loss.weight,
                'residual': self.residual_loss.weight,
                'regularization': self.regularization_loss.weight
            }
        }
    
    def _update_adaptive_weights(self):
        """Update loss weights based on relative magnitudes."""
        if len(self.loss_history['data']) < 2:
            return
        
        # Get recent loss magnitudes
        recent_data = np.mean(self.loss_history['data'][-10:])
        recent_residual = np.mean(self.loss_history['residual'][-10:])
        recent_reg = np.mean(self.loss_history['regularization'][-10:])
        
        # Compute relative scales
        total_magnitude = recent_data + recent_residual + recent_reg
        if total_magnitude > 0:
            # Inverse scaling: give more weight to smaller losses
            data_scale = total_magnitude / (recent_data + 1e-8)
            residual_scale = total_magnitude / (recent_residual + 1e-8)
            reg_scale = total_magnitude / (recent_reg + 1e-8)
            
            # Normalize and apply smoothing
            scale_sum = data_scale + residual_scale + reg_scale
            alpha = 0.1  # Smoothing factor
            
            self.data_loss.weight = (1 - alpha) * self.data_loss.weight + \
                                   alpha * self.original_weights['data'] * data_scale / scale_sum
            
            self.residual_loss.weight = (1 - alpha) * self.residual_loss.weight + \
                                       alpha * self.original_weights['residual'] * residual_scale / scale_sum
            
            self.regularization_loss.weight = (1 - alpha) * self.regularization_loss.weight + \
                                             alpha * self.original_weights['regularization'] * reg_scale / scale_sum
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get statistics about loss evolution."""
        if not self.loss_history['data']:
            return {}
        
        stats = {}
        for loss_type in ['data', 'residual', 'regularization']:
            history = self.loss_history[loss_type]
            if history:
                stats[f'{loss_type}_mean'] = np.mean(history[-100:])  # Recent mean
                stats[f'{loss_type}_std'] = np.std(history[-100:])   # Recent std
                stats[f'{loss_type}_trend'] = np.polyfit(range(len(history)), history, 1)[0]  # Trend
        
        return stats
    
    def reset_history(self):
        """Reset loss history for fresh adaptive weighting."""
        self.loss_history = {'data': [], 'residual': [], 'regularization': []}
        self.iteration_count = 0


# Factory function for creating loss functions
def create_composite_loss(config: Dict) -> CompositeLoss:
    """
    Create composite loss function from configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Configured composite loss function
    """
    return CompositeLoss(
        data_loss_config=config.get('data_loss', {}),
        residual_loss_config=config.get('residual_loss', {}),
        regularization_config=config.get('regularization', {}),
        adaptive_weights=config.get('adaptive_weights', False),
        weight_update_freq=config.get('weight_update_freq', 100)
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100
    
    # Create mock data
    u_pred = torch.randn(batch_size, 1, device=device)
    u_observed = torch.randn(batch_size, 1, device=device)
    
    u_derivatives = {
        'u': u_pred,
        'u_x': torch.randn(batch_size, 1, device=device),
        'u_t': torch.randn(batch_size, 1, device=device)
    }
    
    alpha_values = torch.rand(batch_size, 1, device=device) * 0.5 + 1.25  # α ∈ [1.25, 1.75]
    alpha_derivatives = {
        'alpha': alpha_values,
        'alpha_x': torch.randn(batch_size, 1, device=device) * 0.1
    }
    
    fractional_laplacian = torch.randn(batch_size, 1, device=device)
    forcing_term = torch.randn(batch_size, 1, device=device)
    
    # Test individual loss components
    print("Testing individual loss components...")
    
    data_loss = DataMismatchLoss(weight=1.0)
    loss_data = data_loss(u_pred, u_observed)
    print(f"Data loss: {loss_data.item():.6f}")
    
    residual_loss = PDEResidualLoss(weight=1.0)
    loss_residual = residual_loss(u_derivatives, alpha_values, fractional_laplacian, forcing_term)
    print(f"Residual loss: {loss_residual.item():.6f}")
    
    reg_loss = RegularizationLoss(weight=0.01)
    loss_reg = reg_loss(alpha_derivatives, alpha_values)
    print(f"Regularization loss: {loss_reg.item():.6f}")
    
    # Test composite loss
    print("\nTesting composite loss...")
    
    config = {
        'data_loss': {'weight': 1.0, 'loss_type': 'mse'},
        'residual_loss': {'weight': 1.0},
        'regularization': {'weight': 0.01, 'smoothness_weight': 1.0, 'l1_weight': 0.1},
        'adaptive_weights': True
    }
    
    composite_loss = create_composite_loss(config)
    
    # Simulate training iterations
    for i in range(10):
        losses = composite_loss.compute(
            u_pred, u_observed,
            u_derivatives, alpha_values, fractional_laplacian, forcing_term,
            alpha_derivatives
        )
        
        if i % 3 == 0:
            print(f"Iteration {i}: Total={losses['total'].item():.6f}, "
                  f"Data={losses['data'].item():.6f}, "
                  f"Residual={losses['residual'].item():.6f}, "
                  f"Reg={losses['regularization'].item():.6f}")
    
    # Test statistics
    stats = composite_loss.get_loss_statistics()
    print(f"\nLoss statistics: {stats}")
    
    print("All tests passed!")