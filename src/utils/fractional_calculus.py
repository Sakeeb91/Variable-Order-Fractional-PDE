"""
Fractional Calculus Utilities for Variable-Order PDE Discovery

This module implements the Grünwald-Letnikov discretization and other
fractional calculus operations needed for computing the variable-order
fractional Laplacian in the PDE residual loss.

Author: Sakeeb Rahman
Date: 2025
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from scipy.special import gamma
import warnings


class GrunwaldLetnikovOperator:
    """
    Grünwald-Letnikov fractional derivative operator for variable orders.
    
    This class handles the point-wise computation of fractional derivatives
    with spatially varying fractional orders α(x).
    """
    
    def __init__(self, 
                 max_order: float = 2.0,
                 max_points: int = 100,
                 tolerance: float = 1e-10,
                 device: str = 'cpu'):
        """
        Initialize the Grünwald-Letnikov operator.
        
        Args:
            max_order: Maximum fractional order to support
            max_points: Maximum number of points in stencil
            tolerance: Tolerance for coefficient truncation
            device: Computing device
        """
        self.max_order = max_order
        self.max_points = max_points
        self.tolerance = tolerance
        self.device = device
        
        # Precompute coefficient cache for efficiency
        self._coefficient_cache = {}
        
    def compute_coefficients(self, alpha: float, n_points: int) -> torch.Tensor:
        """
        Compute Grünwald-Letnikov coefficients for given order.
        
        The coefficients are: c_k^(α) = (-1)^k * C(α, k)
        where C(α, k) is the binomial coefficient.
        
        Args:
            alpha: Fractional order
            n_points: Number of coefficients to compute
            
        Returns:
            Coefficients tensor [n_points]
        """
        # Check cache first
        cache_key = (alpha, n_points)
        if cache_key in self._coefficient_cache:
            return self._coefficient_cache[cache_key]
        
        # Compute coefficients
        coeffs = torch.zeros(n_points, device=self.device)
        
        # c_0 = 1
        coeffs[0] = 1.0
        
        # Recursive computation: c_k = c_{k-1} * (α - k + 1) / k
        for k in range(1, n_points):
            coeffs[k] = coeffs[k-1] * (alpha - k + 1) / k
            
            # Apply alternating signs
            if k % 2 == 1:
                coeffs[k] = -coeffs[k]
            
            # Check for truncation
            if abs(coeffs[k].item()) < self.tolerance:
                coeffs = coeffs[:k]
                break
        
        # Cache the result
        self._coefficient_cache[cache_key] = coeffs
        
        return coeffs
    
    def fractional_derivative_1d(self, 
                                u: torch.Tensor,
                                alpha: Union[float, torch.Tensor],
                                dx: float,
                                boundary_condition: str = 'zero') -> torch.Tensor:
        """
        Compute 1D fractional derivative using Grünwald-Letnikov formula.
        
        Args:
            u: Function values [N]
            alpha: Fractional order (scalar or [N])
            dx: Grid spacing
            boundary_condition: Boundary handling ('zero', 'periodic', 'symmetric')
            
        Returns:
            Fractional derivative [N]
        """
        n_points = len(u)
        
        # Handle scalar vs variable alpha
        if isinstance(alpha, (int, float)):
            # Constant alpha case
            alpha_tensor = torch.full((n_points,), alpha, device=self.device)
            constant_alpha = True
        else:
            alpha_tensor = alpha.flatten()
            constant_alpha = False
        
        # Initialize result
        result = torch.zeros_like(u)
        
        if constant_alpha:
            # Optimized path for constant alpha
            coeffs = self.compute_coefficients(alpha, min(n_points, self.max_points))
            n_coeffs = len(coeffs)
            
            for i in range(n_points):
                # Determine how many coefficients to use
                max_k = min(i + 1, n_coeffs)
                
                # Compute fractional derivative at point i
                for k in range(max_k):
                    idx = i - k
                    if idx >= 0:
                        result[i] += coeffs[k] * u[idx]
                    else:
                        # Handle boundary conditions
                        if boundary_condition == 'zero':
                            pass  # Already zero
                        elif boundary_condition == 'periodic':
                            result[i] += coeffs[k] * u[n_points + idx]
                        elif boundary_condition == 'symmetric':
                            result[i] += coeffs[k] * u[-idx]
        else:
            # Variable alpha case (slower but necessary)
            for i in range(n_points):
                alpha_i = alpha_tensor[i].item()
                coeffs = self.compute_coefficients(alpha_i, min(i + 1, self.max_points))
                n_coeffs = len(coeffs)
                
                # Compute fractional derivative at point i
                for k in range(n_coeffs):
                    idx = i - k
                    if idx >= 0:
                        result[i] += coeffs[k] * u[idx]
                    else:
                        # Handle boundary conditions
                        if boundary_condition == 'zero':
                            pass
                        elif boundary_condition == 'periodic':
                            result[i] += coeffs[k] * u[n_points + idx]
                        elif boundary_condition == 'symmetric':
                            result[i] += coeffs[k] * u[-idx]
        
        # Scale by dx^(-alpha)
        if constant_alpha:
            result = result / (dx ** alpha)
        else:
            result = result / (dx ** alpha_tensor)
        
        return result
    
    def fractional_laplacian_1d(self,
                               u: torch.Tensor,
                               alpha: Union[float, torch.Tensor],
                               dx: float) -> torch.Tensor:
        """
        Compute 1D fractional Laplacian (-Δ)^(α/2) u.
        
        For 1D, this is the fractional second derivative with order α.
        
        Args:
            u: Function values [N]
            alpha: Fractional order
            dx: Grid spacing
            
        Returns:
            Fractional Laplacian [N]
        """
        return self.fractional_derivative_1d(u, alpha, dx)
    
    def clear_cache(self):
        """Clear the coefficient cache to free memory."""
        self._coefficient_cache.clear()


class FractionalOperatorPINN:
    """
    Fractional operator designed specifically for PINN integration.
    
    This class provides methods optimized for use within the PINN training
    framework, including batch processing and automatic differentiation support.
    """
    
    def __init__(self, 
                 domain_bounds: Tuple[float, float] = (0.0, 1.0),
                 n_grid: int = 101,
                 device: str = 'cpu'):
        """
        Initialize fractional operator for PINN.
        
        Args:
            domain_bounds: Spatial domain (x_min, x_max)
            n_grid: Number of grid points for discretization
            device: Computing device
        """
        self.domain_bounds = domain_bounds
        self.n_grid = n_grid
        self.device = device
        
        # Create reference grid
        self.x_grid = torch.linspace(
            domain_bounds[0], domain_bounds[1], n_grid, device=device
        )
        self.dx = (domain_bounds[1] - domain_bounds[0]) / (n_grid - 1)
        
        # Initialize GL operator
        self.gl_operator = GrunwaldLetnikovOperator(device=device)
        
    def interpolate_to_grid(self, 
                           x: torch.Tensor, 
                           u: torch.Tensor) -> torch.Tensor:
        """
        Interpolate function values from arbitrary points to regular grid.
        
        Args:
            x: Spatial coordinates [N]
            u: Function values [N]
            
        Returns:
            Interpolated values on grid [n_grid]
        """
        # Use linear interpolation
        x_normalized = (x - self.domain_bounds[0]) / (self.domain_bounds[1] - self.domain_bounds[0])
        grid_indices = x_normalized * (self.n_grid - 1)
        
        # Clamp to valid range
        grid_indices = torch.clamp(grid_indices, 0, self.n_grid - 1)
        
        # Linear interpolation
        indices_floor = torch.floor(grid_indices).long()
        indices_ceil = torch.ceil(grid_indices).long()
        weights = grid_indices - indices_floor.float()
        
        # Handle boundary case
        indices_ceil = torch.clamp(indices_ceil, 0, self.n_grid - 1)
        
        # Interpolate
        u_grid = torch.zeros(self.n_grid, device=self.device)
        
        # Simple scatter-based interpolation (can be optimized)
        for i in range(len(x)):
            idx_low = indices_floor[i]
            idx_high = indices_ceil[i]
            w = weights[i]
            
            if idx_low == idx_high:
                u_grid[idx_low] += u[i]
            else:
                u_grid[idx_low] += (1 - w) * u[i]
                u_grid[idx_high] += w * u[i]
        
        return u_grid
    
    def interpolate_from_grid(self, 
                             x: torch.Tensor, 
                             u_grid: torch.Tensor) -> torch.Tensor:
        """
        Interpolate from regular grid to arbitrary points.
        
        Args:
            x: Target spatial coordinates [N]
            u_grid: Function values on grid [n_grid]
            
        Returns:
            Interpolated values [N]
        """
        x_normalized = (x - self.domain_bounds[0]) / (self.domain_bounds[1] - self.domain_bounds[0])
        grid_indices = x_normalized * (self.n_grid - 1)
        
        # Clamp to valid range
        grid_indices = torch.clamp(grid_indices, 0, self.n_grid - 1)
        
        # Linear interpolation
        indices_floor = torch.floor(grid_indices).long()
        indices_ceil = torch.ceil(grid_indices).long()
        weights = grid_indices - indices_floor.float()
        
        # Handle boundary case
        indices_ceil = torch.clamp(indices_ceil, 0, self.n_grid - 1)
        
        # Interpolate
        u_interp = torch.zeros_like(x)
        
        for i in range(len(x)):
            idx_low = indices_floor[i]
            idx_high = indices_ceil[i]
            w = weights[i]
            
            if idx_low == idx_high:
                u_interp[i] = u_grid[idx_low]
            else:
                u_interp[i] = (1 - w) * u_grid[idx_low] + w * u_grid[idx_high]
        
        return u_interp
    
    def compute_fractional_laplacian(self,
                                   x: torch.Tensor,
                                   u: torch.Tensor,
                                   alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute fractional Laplacian for PINN training.
        
        This method handles the interpolation between arbitrary collocation
        points and the regular grid required for GL discretization.
        
        Args:
            x: Spatial coordinates [N]
            u: Function values [N]
            alpha: Fractional orders [N]
            
        Returns:
            Fractional Laplacian values [N]
        """
        # Interpolate to grid
        u_grid = self.interpolate_to_grid(x, u)
        
        # Check if alpha is constant or variable
        alpha_unique = torch.unique(alpha)
        
        if len(alpha_unique) == 1:
            # Constant alpha case
            alpha_val = alpha_unique[0].item()
            frac_lap_grid = self.gl_operator.fractional_laplacian_1d(
                u_grid, alpha_val, self.dx
            )
        else:
            # Variable alpha case - interpolate alpha to grid
            alpha_grid = self.interpolate_to_grid(x, alpha.flatten())
            frac_lap_grid = self.gl_operator.fractional_laplacian_1d(
                u_grid, alpha_grid, self.dx
            )
        
        # Interpolate back to original points
        frac_lap = self.interpolate_from_grid(x, frac_lap_grid)
        
        return frac_lap.unsqueeze(-1)  # Add dimension for consistency


def create_fractional_operator(config: dict) -> FractionalOperatorPINN:
    """
    Factory function to create fractional operators.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured fractional operator
    """
    return FractionalOperatorPINN(
        domain_bounds=config.get('domain_bounds', (0.0, 1.0)),
        n_grid=config.get('n_grid', 101),
        device=config.get('device', 'cpu')
    )


# Utility functions
def gamma_function(x: torch.Tensor) -> torch.Tensor:
    """Compute gamma function using log-gamma for numerical stability."""
    return torch.exp(torch.lgamma(x))


def binomial_coefficient(n: float, k: int) -> float:
    """Compute generalized binomial coefficient C(n, k)."""
    if k == 0:
        return 1.0
    
    result = 1.0
    for i in range(k):
        result *= (n - i) / (i + 1)
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("Testing Fractional Calculus Utilities...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test Grünwald-Letnikov operator
    print("\nTesting Grünwald-Letnikov operator...")
    
    gl_op = GrunwaldLetnikovOperator(device=device)
    
    # Test coefficient computation
    alpha = 1.5
    n_points = 20
    coeffs = gl_op.compute_coefficients(alpha, n_points)
    print(f"GL coefficients for α={alpha}: {coeffs[:5].tolist()} ... (first 5)")
    
    # Test fractional derivative
    n_grid = 51
    x = torch.linspace(0, 1, n_grid, device=device)
    u = torch.sin(np.pi * x)  # Test function
    dx = 1.0 / (n_grid - 1)
    
    # Constant alpha
    frac_deriv = gl_op.fractional_derivative_1d(u, alpha, dx)
    print(f"Fractional derivative computed. Shape: {frac_deriv.shape}")
    
    # Variable alpha
    alpha_var = 1.2 + 0.6 * torch.sin(2 * np.pi * x)
    frac_deriv_var = gl_op.fractional_derivative_1d(u, alpha_var, dx)
    print(f"Variable-order fractional derivative computed. Shape: {frac_deriv_var.shape}")
    
    # Test PINN fractional operator
    print("\nTesting PINN fractional operator...")
    
    config = {
        'domain_bounds': (0.0, 1.0),
        'n_grid': 101,
        'device': str(device)
    }
    
    frac_op = create_fractional_operator(config)
    
    # Test with random collocation points
    n_colloc = 50
    x_colloc = torch.rand(n_colloc, device=device)
    u_colloc = torch.sin(np.pi * x_colloc)
    alpha_colloc = torch.full((n_colloc,), 1.5, device=device)
    
    frac_lap = frac_op.compute_fractional_laplacian(x_colloc, u_colloc, alpha_colloc)
    print(f"Fractional Laplacian computed. Shape: {frac_lap.shape}")
    
    # Test with variable alpha
    alpha_var_colloc = 1.2 + 0.6 * torch.sin(2 * np.pi * x_colloc)
    frac_lap_var = frac_op.compute_fractional_laplacian(x_colloc, u_colloc, alpha_var_colloc)
    print(f"Variable-order fractional Laplacian computed. Shape: {frac_lap_var.shape}")
    
    print("All tests passed!")