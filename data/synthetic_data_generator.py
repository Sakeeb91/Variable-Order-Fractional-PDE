"""
Synthetic Data Generation for Variable-Order Fractional PDE Discovery

This module generates synthetic datasets for validating the variable-order
fractional PDE discovery methodology. It creates ground truth solutions
and corresponding fractional order functions for controlled experiments.

Author: Sakeeb Rahman
Date: 2024
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Optional
import os


class VariableOrderPDEGenerator:
    """Generate synthetic data for variable-order fractional PDE experiments."""
    
    def __init__(self, 
                 domain_x: Tuple[float, float] = (0.0, 1.0),
                 domain_t: Tuple[float, float] = (0.0, 1.0),
                 nx: int = 101,
                 nt: int = 101,
                 device: str = 'cpu'):
        """
        Initialize the synthetic data generator.
        
        Args:
            domain_x: Spatial domain bounds (x_min, x_max)
            domain_t: Temporal domain bounds (t_min, t_max)
            nx: Number of spatial grid points
            nt: Number of temporal grid points
            device: Computing device ('cpu' or 'cuda')
        """
        self.domain_x = domain_x
        self.domain_t = domain_t
        self.nx = nx
        self.nt = nt
        self.device = device
        
        # Create coordinate grids
        self.x = torch.linspace(domain_x[0], domain_x[1], nx, device=device)
        self.t = torch.linspace(domain_t[0], domain_t[1], nt, device=device)
        self.X, self.T = torch.meshgrid(self.x, self.t, indexing='ij')
        
        # Grid spacing
        self.dx = (domain_x[1] - domain_x[0]) / (nx - 1)
        self.dt = (domain_t[1] - domain_t[0]) / (nt - 1)
        
    def constant_alpha_case(self, alpha: float = 1.5) -> dict:
        """
        Generate data for constant fractional order case (sanity check).
        
        Args:
            alpha: Constant fractional order value
            
        Returns:
            Dictionary containing ground truth data
        """
        # Ground truth fractional order (constant)
        alpha_true = torch.full_like(self.x, alpha)
        
        # Simple analytical solution: u(x,t) = sin(πx) * exp(-t)
        u_true = torch.sin(np.pi * self.X) * torch.exp(-self.T)
        
        # Generate sparse observation points
        n_obs = min(50, self.nx * self.nt // 10)
        obs_indices = torch.randperm(self.nx * self.nt)[:n_obs]
        x_obs = self.X.flatten()[obs_indices]
        t_obs = self.T.flatten()[obs_indices]
        u_obs = u_true.flatten()[obs_indices]
        
        return {
            'alpha_true': alpha_true,
            'u_true': u_true,
            'x_obs': x_obs,
            't_obs': t_obs,
            'u_obs': u_obs,
            'x_grid': self.x,
            't_grid': self.t,
            'description': f'Constant fractional order α = {alpha}'
        }
    
    def smooth_varying_alpha_case(self, 
                                amplitude: float = 0.25,
                                frequency: int = 2) -> dict:
        """
        Generate data for smoothly varying fractional order.
        
        Args:
            amplitude: Amplitude of α(x) variation
            frequency: Frequency of sinusoidal variation
            
        Returns:
            Dictionary containing ground truth data
        """
        # Ground truth fractional order: α(x) = amplitude * sin(2πfx) + 1.5
        alpha_true = amplitude * torch.sin(2 * np.pi * frequency * self.x) + 1.5
        
        # Ensure α stays in valid range (1, 2)
        alpha_true = torch.clamp(alpha_true, 1.1, 1.9)
        
        # Generate corresponding solution using manufactured solution approach
        # u(x,t) = sin(πx) * g(t) where g(t) = exp(-t)
        u_base = torch.sin(np.pi * self.X)
        g_t = torch.exp(-self.T)
        u_true = u_base * g_t
        
        # Generate sparse observation points with noise
        n_obs = min(100, self.nx * self.nt // 8)
        obs_indices = torch.randperm(self.nx * self.nt)[:n_obs]
        x_obs = self.X.flatten()[obs_indices]
        t_obs = self.T.flatten()[obs_indices]
        u_obs = u_true.flatten()[obs_indices]
        
        # Add noise to observations
        noise_std = 0.01 * torch.std(u_obs)
        u_obs += torch.randn_like(u_obs) * noise_std
        
        return {
            'alpha_true': alpha_true,
            'u_true': u_true,
            'x_obs': x_obs,
            't_obs': t_obs,
            'u_obs': u_obs,
            'x_grid': self.x,
            't_grid': self.t,
            'noise_std': noise_std.item(),
            'description': f'Smooth varying α(x) with amplitude {amplitude}'
        }
    
    def step_function_alpha_case(self, 
                               alpha_left: float = 1.3,
                               alpha_right: float = 1.7,
                               transition_center: float = 0.5,
                               transition_width: float = 0.1) -> dict:
        """
        Generate data for step-function-like fractional order (challenging case).
        
        Args:
            alpha_left: Fractional order in left region
            alpha_right: Fractional order in right region
            transition_center: Center of transition region
            transition_width: Width of smooth transition
            
        Returns:
            Dictionary containing ground truth data
        """
        # Ground truth fractional order with smooth step transition
        alpha_true = alpha_left + (alpha_right - alpha_left) * \
                    torch.sigmoid((self.x - transition_center) / transition_width)
        
        # Generate solution assuming piecewise behavior
        u_left = torch.sin(np.pi * self.X) * torch.exp(-self.T)
        u_right = torch.sin(2 * np.pi * self.X) * torch.exp(-1.2 * self.T)
        
        # Blend solutions based on spatial location
        weight = torch.sigmoid((self.X - transition_center) / transition_width)
        u_true = (1 - weight) * u_left + weight * u_right
        
        # Generate sparse observation points
        n_obs = min(80, self.nx * self.nt // 12)
        obs_indices = torch.randperm(self.nx * self.nt)[:n_obs]
        x_obs = self.X.flatten()[obs_indices]
        t_obs = self.T.flatten()[obs_indices]
        u_obs = u_true.flatten()[obs_indices]
        
        # Add noise
        noise_std = 0.015 * torch.std(u_obs)
        u_obs += torch.randn_like(u_obs) * noise_std
        
        return {
            'alpha_true': alpha_true,
            'u_true': u_true,
            'x_obs': x_obs,
            't_obs': t_obs,
            'u_obs': u_obs,
            'x_grid': self.x,
            't_grid': self.t,
            'noise_std': noise_std.item(),
            'description': f'Step function α(x): {alpha_left} → {alpha_right}'
        }
    
    def save_dataset(self, data: dict, filename: str, 
                    save_dir: str = 'data/processed') -> None:
        """
        Save generated dataset to file.
        
        Args:
            data: Dataset dictionary
            filename: Output filename
            save_dir: Directory to save data
        """
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        # Convert tensors to numpy for saving
        data_np = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                data_np[key] = value.cpu().numpy()
            else:
                data_np[key] = value
        
        np.savez(filepath, **data_np)
        print(f"Dataset saved to {filepath}.npz")
    
    def visualize_dataset(self, data: dict, save_path: Optional[str] = None) -> None:
        """
        Create visualization of generated dataset.
        
        Args:
            data: Dataset dictionary
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot fractional order function
        axes[0, 0].plot(data['x_grid'].cpu(), data['alpha_true'].cpu(), 
                       'b-', linewidth=2, label='α(x)')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('α(x)')
        axes[0, 0].set_title('Ground Truth Fractional Order')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot solution field
        X_np = data['X'].cpu() if 'X' in data else self.X.cpu()
        T_np = data['T'].cpu() if 'T' in data else self.T.cpu()
        u_np = data['u_true'].cpu()
        
        im = axes[0, 1].contourf(X_np, T_np, u_np, levels=20, cmap='viridis')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('t')
        axes[0, 1].set_title('Ground Truth Solution u(x,t)')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot observation points
        axes[1, 0].scatter(data['x_obs'].cpu(), data['t_obs'].cpu(), 
                          c=data['u_obs'].cpu(), cmap='viridis', s=20)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('t')
        axes[1, 0].set_title('Observation Points')
        
        # Plot solution at different times
        t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i, t_val in enumerate(t_slices):
            t_idx = int(t_val * (self.nt - 1))
            axes[1, 1].plot(data['x_grid'].cpu(), u_np[t_idx, :], 
                           label=f't = {t_val}', alpha=0.8)
        
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('u(x,t)')
        axes[1, 1].set_title('Solution Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def generate_all_datasets():
    """Generate all standard datasets for experiments."""
    generator = VariableOrderPDEGenerator(nx=101, nt=101)
    
    # Experiment 1: Constant order (sanity check)
    print("Generating Experiment 1: Constant fractional order...")
    data1 = generator.constant_alpha_case(alpha=1.5)
    generator.save_dataset(data1, 'experiment1_constant_alpha')
    generator.visualize_dataset(data1, 'visuals/experiment1_dataset.png')
    
    # Experiment 2: Smooth variation
    print("Generating Experiment 2: Smooth varying fractional order...")
    data2 = generator.smooth_varying_alpha_case(amplitude=0.25, frequency=2)
    generator.save_dataset(data2, 'experiment2_smooth_alpha')
    generator.visualize_dataset(data2, 'visuals/experiment2_dataset.png')
    
    # Experiment 3: Step function (challenging)
    print("Generating Experiment 3: Step function fractional order...")
    data3 = generator.step_function_alpha_case(alpha_left=1.3, alpha_right=1.7)
    generator.save_dataset(data3, 'experiment3_step_alpha')
    generator.visualize_dataset(data3, 'visuals/experiment3_dataset.png')
    
    print("All datasets generated successfully!")


if __name__ == "__main__":
    generate_all_datasets()