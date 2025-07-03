"""
Visualization Utilities for Variable-Order Fractional PDE Discovery

This module provides comprehensive plotting and visualization tools for
analyzing training results, comparing predictions with ground truth,
and generating publication-quality figures.

Author: Sakeeb Rahman
Date: 2025
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsVisualizer:
    """
    Comprehensive visualization toolkit for variable-order fractional PDE results.
    
    This class provides methods for creating all the plots described in the
    plots_description.md documentation.
    """
    
    def __init__(self, 
                 save_dir: str = 'visuals',
                 dpi: int = 300,
                 figsize_base: Tuple[int, int] = (12, 8),
                 style: str = 'publication'):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save plots
            dpi: Resolution for saved figures
            figsize_base: Base figure size
            style: Visualization style ('publication', 'presentation', 'notebook')
        """
        self.save_dir = save_dir
        self.dpi = dpi
        self.figsize_base = figsize_base
        self.style = style
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Configure style
        self._configure_style()
        
        # Color schemes
        self.colors = {
            'ground_truth': '#1f77b4',  # Blue
            'predicted': '#ff7f0e',     # Orange
            'error': '#d62728',         # Red
            'residual': '#2ca02c',      # Green
            'regularization': '#9467bd' # Purple
        }
    
    def _configure_style(self):
        """Configure matplotlib style based on target output."""
        if self.style == 'publication':
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 18,
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'text.usetex': False,  # Set to True if LaTeX is available
                'axes.linewidth': 1.2,
                'grid.alpha': 0.3
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'axes.labelsize': 16,
                'axes.titlesize': 18,
                'legend.fontsize': 14,
                'figure.titlesize': 20,
                'lines.linewidth': 3
            })
    
    def plot_training_dynamics(self, 
                             training_history: Dict[str, List[float]],
                             save_name: str = 'training_loss_evolution.png') -> None:
        """
        Plot training loss evolution over epochs.
        
        Args:
            training_history: Dictionary with loss history
            save_name: Filename for saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Loss Evolution', fontsize=18, fontweight='bold')
        
        epochs = range(len(training_history['total']))
        
        # Total loss
        axes[0, 0].semilogy(epochs, training_history['total'], 
                           color=self.colors['ground_truth'], linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (log scale)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Data mismatch loss
        axes[0, 1].semilogy(epochs, training_history['data'], 
                           color=self.colors['predicted'], linewidth=2)
        axes[0, 1].set_title('Data Mismatch Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # PDE residual loss
        axes[1, 0].semilogy(epochs, training_history['residual'], 
                           color=self.colors['residual'], linewidth=2)
        axes[1, 0].set_title('PDE Residual Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log scale)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Regularization loss
        axes[1, 1].semilogy(epochs, training_history['regularization'], 
                           color=self.colors['regularization'], linewidth=2)
        axes[1, 1].set_title('Regularization Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log scale)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()
    
    def plot_loss_component_breakdown(self,
                                    training_history: Dict[str, List[float]],
                                    save_name: str = 'loss_component_breakdown.png') -> None:
        """
        Plot stacked area chart of loss component contributions.
        
        Args:
            training_history: Dictionary with loss history
            save_name: Filename for saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize_base)
        
        epochs = range(len(training_history['total']))
        
        # Normalize components to show relative contributions
        data_contrib = np.array(training_history['data'])
        residual_contrib = np.array(training_history['residual'])
        reg_contrib = np.array(training_history['regularization'])
        
        total = data_contrib + residual_contrib + reg_contrib
        data_norm = data_contrib / total
        residual_norm = residual_contrib / total
        reg_norm = reg_contrib / total
        
        # Create stacked area plot
        ax.fill_between(epochs, 0, data_norm, 
                       alpha=0.7, color=self.colors['predicted'], label='Data Mismatch')
        ax.fill_between(epochs, data_norm, data_norm + residual_norm,
                       alpha=0.7, color=self.colors['residual'], label='PDE Residual')
        ax.fill_between(epochs, data_norm + residual_norm, 1.0,
                       alpha=0.7, color=self.colors['regularization'], label='Regularization')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative Contribution')
        ax.set_title('Loss Component Breakdown Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()
    
    def plot_solution_evolution(self,
                               x_grid: torch.Tensor,
                               t_grid: torch.Tensor,
                               u_true: torch.Tensor,
                               u_pred: Optional[torch.Tensor] = None,
                               save_name: str = 'solution_evolution_2d.png') -> None:
        """
        Plot 2D heatmap of solution evolution over time.
        
        Args:
            x_grid: Spatial coordinates
            t_grid: Temporal coordinates  
            u_true: Ground truth solution
            u_pred: Predicted solution (optional)
            save_name: Filename for saved plot
        """
        n_plots = 3 if u_pred is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 2:
            axes = [axes] if not hasattr(axes, '__len__') else axes
        
        # Convert to numpy
        X = x_grid.cpu().numpy() if torch.is_tensor(x_grid) else x_grid
        T = t_grid.cpu().numpy() if torch.is_tensor(t_grid) else t_grid
        U_true = u_true.cpu().numpy() if torch.is_tensor(u_true) else u_true
        
        # Ground truth
        im1 = axes[0].contourf(X, T, U_true, levels=20, cmap='viridis')
        axes[0].set_title('Ground Truth u(x,t)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('t')
        plt.colorbar(im1, ax=axes[0])
        
        if u_pred is not None:
            U_pred = u_pred.cpu().numpy() if torch.is_tensor(u_pred) else u_pred
            
            # Predicted
            im2 = axes[1].contourf(X, T, U_pred, levels=20, cmap='viridis')
            axes[1].set_title('Predicted u(x,t)')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('t')
            plt.colorbar(im2, ax=axes[1])
            
            # Error
            error = np.abs(U_pred - U_true)
            im3 = axes[2].contourf(X, T, error, levels=20, cmap='Reds')
            axes[2].set_title('Absolute Error |u_pred - u_true|')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('t')
            plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()
    
    def plot_alpha_function_recovery(self,
                                   x: torch.Tensor,
                                   alpha_true: torch.Tensor,
                                   alpha_pred: torch.Tensor,
                                   confidence_interval: Optional[torch.Tensor] = None,
                                   save_name: str = 'alpha_function_recovery.png') -> None:
        """
        Plot comparison of discovered α(x) with ground truth.
        
        Args:
            x: Spatial coordinates
            alpha_true: Ground truth fractional order
            alpha_pred: Predicted fractional order
            confidence_interval: Optional confidence bounds
            save_name: Filename for saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize_base)
        
        # Convert to numpy
        x_np = x.cpu().numpy() if torch.is_tensor(x) else x
        alpha_true_np = alpha_true.cpu().numpy() if torch.is_tensor(alpha_true) else alpha_true
        alpha_pred_np = alpha_pred.cpu().numpy() if torch.is_tensor(alpha_pred) else alpha_pred
        
        # Plot ground truth
        ax.plot(x_np, alpha_true_np, 'b-', linewidth=3, 
               label='Ground Truth α(x)', color=self.colors['ground_truth'])
        
        # Plot prediction
        ax.plot(x_np, alpha_pred_np, 'r--', linewidth=2, 
               label='Predicted α(x)', color=self.colors['predicted'])
        
        # Add confidence interval if provided
        if confidence_interval is not None:
            ci_np = confidence_interval.cpu().numpy() if torch.is_tensor(confidence_interval) else confidence_interval
            ax.fill_between(x_np, alpha_pred_np - ci_np, alpha_pred_np + ci_np,
                           alpha=0.3, color=self.colors['predicted'], label='Confidence Interval')
        
        # Calculate and display error
        l2_error = np.sqrt(np.mean((alpha_pred_np - alpha_true_np)**2))
        relative_error = l2_error / np.sqrt(np.mean(alpha_true_np**2)) * 100
        
        ax.text(0.05, 0.95, f'L2 Error: {l2_error:.4f}\\nRelative: {relative_error:.2f}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('x')
        ax.set_ylabel('α(x)')
        ax.set_title('Fractional Order Recovery')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()
    
    def plot_experiment_comparison(self,
                                 results: Dict[str, Dict],
                                 save_name: str = 'experiment_results_summary.png') -> None:
        """
        Plot comparison across multiple experiments.
        
        Args:
            results: Dictionary with experiment results
            save_name: Filename for saved plot
        """
        n_experiments = len(results)
        fig, axes = plt.subplots(2, n_experiments, figsize=(5*n_experiments, 8))
        
        if n_experiments == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Experiment Results Comparison', fontsize=18, fontweight='bold')
        
        for i, (exp_name, data) in enumerate(results.items()):
            # Plot α(x) recovery
            x = data['x']
            alpha_true = data['alpha_true']
            alpha_pred = data['alpha_pred']
            
            axes[0, i].plot(x, alpha_true, 'b-', linewidth=2, label='True')
            axes[0, i].plot(x, alpha_pred, 'r--', linewidth=2, label='Predicted')
            axes[0, i].set_title(f'{exp_name}: α(x) Recovery')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('α(x)')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot error metrics
            metrics = data.get('metrics', {})
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[1, i].bar(range(len(metric_names)), metric_values, 
                                 color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(metric_names)])
            axes[1, i].set_title(f'{exp_name}: Error Metrics')
            axes[1, i].set_xticks(range(len(metric_names)))
            axes[1, i].set_xticklabels(metric_names, rotation=45)
            axes[1, i].set_ylabel('Error Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[1, i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()
    
    def plot_ablation_study(self,
                           ablation_results: Dict[str, Dict],
                           save_name: str = 'regularization_impact_comparison.png') -> None:
        """
        Plot ablation study results showing regularization impact.
        
        Args:
            ablation_results: Dictionary with ablation study results
            save_name: Filename for saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Regularization Impact Analysis', fontsize=16, fontweight='bold')
        
        cases = ['no_regularization', 'smoothness_only', 'l1_only', 'full_regularization']
        titles = ['No Regularization', 'Smoothness Only', 'L1 Only', 'Full Regularization']
        
        for i, (case, title) in enumerate(zip(cases, titles)):
            if case not in ablation_results:
                continue
                
            row, col = i // 2, i % 2
            data = ablation_results[case]
            
            x = data['x']
            alpha_true = data['alpha_true']
            alpha_pred = data['alpha_pred']
            
            axes[row, col].plot(x, alpha_true, 'b-', linewidth=2, label='True')
            axes[row, col].plot(x, alpha_pred, 'r--', linewidth=2, label='Predicted')
            
            # Calculate error
            error = np.mean((alpha_pred - alpha_true)**2)
            axes[row, col].set_title(f'{title}\\nMSE: {error:.4f}')
            axes[row, col].set_xlabel('x')
            axes[row, col].set_ylabel('α(x)')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()
    
    def plot_convergence_analysis(self,
                                training_histories: Dict[str, Dict],
                                save_name: str = 'error_convergence_analysis.png') -> None:
        """
        Plot convergence analysis across multiple runs.
        
        Args:
            training_histories: Dictionary with training histories from multiple runs
            save_name: Filename for saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Loss convergence
        for run_name, history in training_histories.items():
            epochs = range(len(history['total']))
            ax1.semilogy(epochs, history['total'], label=run_name, alpha=0.7)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss (log scale)')
        ax1.set_title('Loss Convergence Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final loss distribution
        final_losses = [history['total'][-1] for history in training_histories.values()]
        ax2.hist(final_losses, bins=20, alpha=0.7, color=self.colors['ground_truth'])
        ax2.axvline(np.mean(final_losses), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(final_losses):.4f}')
        ax2.set_xlabel('Final Total Loss')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Final Loss Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()
    
    def create_publication_figure(self,
                                experiment_data: Dict,
                                save_name: str = 'publication_main_figure.png') -> None:
        """
        Create a comprehensive publication-quality figure.
        
        Args:
            experiment_data: Complete experiment data
            save_name: Filename for saved plot
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Variable-Order Fractional PDE Discovery Results', 
                    fontsize=20, fontweight='bold')
        
        # A: Training dynamics
        ax1 = fig.add_subplot(gs[0, :2])
        history = experiment_data['training_history']
        epochs = range(len(history['total']))
        ax1.semilogy(epochs, history['total'], 'b-', linewidth=2, label='Total')
        ax1.semilogy(epochs, history['data'], 'r-', linewidth=2, label='Data')
        ax1.semilogy(epochs, history['residual'], 'g-', linewidth=2, label='Residual')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_title('(A) Training Loss Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # B: α(x) recovery
        ax2 = fig.add_subplot(gs[0, 2:])
        x = experiment_data['x']
        alpha_true = experiment_data['alpha_true']
        alpha_pred = experiment_data['alpha_pred']
        ax2.plot(x, alpha_true, 'b-', linewidth=3, label='Ground Truth')
        ax2.plot(x, alpha_pred, 'r--', linewidth=2, label='Discovered')
        ax2.set_xlabel('x')
        ax2.set_ylabel('α(x)')
        ax2.set_title('(B) Fractional Order Discovery')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # C: Solution comparison
        ax3 = fig.add_subplot(gs[1, :2])
        X = experiment_data['X']
        T = experiment_data['T']
        u_true = experiment_data['u_true']
        im = ax3.contourf(X, T, u_true, levels=20, cmap='viridis')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_title('(C) Ground Truth Solution u(x,t)')
        plt.colorbar(im, ax=ax3)
        
        # D: Prediction error
        ax4 = fig.add_subplot(gs[1, 2:])
        u_pred = experiment_data['u_pred']
        error = np.abs(u_pred - u_true)
        im2 = ax4.contourf(X, T, error, levels=20, cmap='Reds')
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        ax4.set_title('(D) Prediction Error |u_pred - u_true|')
        plt.colorbar(im2, ax=ax4)
        
        # E: Error metrics
        ax5 = fig.add_subplot(gs[2, :2])
        metrics = experiment_data['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        bars = ax5.bar(metric_names, metric_values, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(metrics)])
        ax5.set_ylabel('Error Value')
        ax5.set_title('(E) Quantitative Metrics')
        plt.setp(ax5.get_xticklabels(), rotation=45)
        
        # F: Regularization components
        ax6 = fig.add_subplot(gs[2, 2:])
        reg_history = {
            'Smoothness': history.get('smoothness', [0] * len(epochs)),
            'L1': history.get('l1', [0] * len(epochs)),
            'Total Reg': history['regularization']
        }
        for name, values in reg_history.items():
            ax6.semilogy(epochs, values, linewidth=2, label=name)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Regularization Loss (log scale)')
        ax6.set_title('(F) Regularization Components')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add timestamp and commit info
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, alpha=0.7)
        
        self._save_figure(fig, save_name)
        plt.show()
    
    def _save_figure(self, fig, filename: str) -> None:
        """Save figure in multiple formats."""
        base_name = os.path.splitext(filename)[0]
        
        # Save as PNG
        fig.savefig(os.path.join(self.save_dir, f'{base_name}.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        
        # Save as PDF for publications
        fig.savefig(os.path.join(self.save_dir, f'{base_name}.pdf'), 
                   bbox_inches='tight')
        
        # Save as SVG for editing
        fig.savefig(os.path.join(self.save_dir, f'{base_name}.svg'), 
                   bbox_inches='tight')


# Utility functions
def create_custom_colormap(colors: List[str], name: str = 'custom') -> LinearSegmentedColormap:
    """Create custom colormap from list of colors."""
    return LinearSegmentedColormap.from_list(name, colors)


def add_timestamp_watermark(fig, position: str = 'bottom_right') -> None:
    """Add timestamp watermark to figure."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if position == 'bottom_right':
        fig.text(0.95, 0.02, timestamp, fontsize=8, alpha=0.5, 
                ha='right', va='bottom')
    elif position == 'bottom_left':
        fig.text(0.05, 0.02, timestamp, fontsize=8, alpha=0.5, 
                ha='left', va='bottom')


def export_data_for_plots(data: Dict, filename: str) -> None:
    """Export data used for plots in CSV format for reproducibility."""
    # Convert tensors to numpy arrays
    export_data = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            export_data[key] = value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            export_data[key] = value
        else:
            export_data[key] = value
    
    # Save as CSV if possible, otherwise as numpy
    try:
        df = pd.DataFrame(export_data)
        df.to_csv(filename.replace('.npz', '.csv'), index=False)
    except:
        np.savez(filename, **export_data)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Visualization Utilities...")
    
    # Create test visualizer
    visualizer = ResultsVisualizer(save_dir='test_visuals')
    
    # Generate test data
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 50)
    X, T = np.meshgrid(x, t, indexing='ij')
    
    # Test solution evolution plot
    u_true = np.sin(np.pi * X) * np.exp(-T)
    u_pred = u_true + 0.01 * np.random.randn(*u_true.shape)
    
    print("Testing solution evolution plot...")
    visualizer.plot_solution_evolution(
        torch.tensor(X), torch.tensor(T), 
        torch.tensor(u_true), torch.tensor(u_pred),
        save_name='test_solution_evolution.png'
    )
    
    # Test alpha recovery plot
    alpha_true = 0.25 * np.sin(2 * np.pi * x) + 1.5
    alpha_pred = alpha_true + 0.05 * np.random.randn(len(x))
    
    print("Testing alpha recovery plot...")
    visualizer.plot_alpha_function_recovery(
        torch.tensor(x), torch.tensor(alpha_true), torch.tensor(alpha_pred),
        save_name='test_alpha_recovery.png'
    )
    
    # Test training dynamics
    training_history = {
        'total': np.exp(-np.linspace(0, 5, 1000)) + 0.001,
        'data': np.exp(-np.linspace(0, 4, 1000)) + 0.001,
        'residual': np.exp(-np.linspace(0, 6, 1000)) + 0.001,
        'regularization': np.exp(-np.linspace(0, 3, 1000)) + 0.001
    }
    
    print("Testing training dynamics plot...")
    visualizer.plot_training_dynamics(training_history, 'test_training_dynamics.png')
    
    print("All visualization tests completed successfully!")
    print(f"Test plots saved in: test_visuals/")