"""
Order Network (α_NN) for Variable-Order Fractional PDE Discovery

This module implements the neural network architecture that learns the
spatially varying fractional order α(x) in the variable-order fractional PDE.
The network includes constrained outputs to ensure α remains in physically
meaningful ranges.

Author: Sakeeb Rahman
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class OrderNetwork(nn.Module):
    """
    Neural network for learning spatially varying fractional order α(x).
    
    This network serves as the α_NN component in the dual-network architecture.
    It takes spatial coordinates as input and outputs the predicted fractional
    order with proper physical constraints.
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_layers: int = 3,
                 neurons_per_layer: int = 30,
                 activation: str = 'tanh',
                 alpha_bounds: Tuple[float, float] = (1.0, 2.0),
                 constraint_type: str = 'tanh',
                 initialization: str = 'xavier_normal'):
        """
        Initialize the order network.
        
        Args:
            input_dim: Input dimension (1 for spatial coordinate x)
            hidden_layers: Number of hidden layers
            neurons_per_layer: Number of neurons in each hidden layer
            activation: Activation function
            alpha_bounds: (min, max) bounds for fractional order
            constraint_type: Method to enforce bounds ('tanh', 'sigmoid', 'softplus')
            initialization: Weight initialization scheme
        """
        super(OrderNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation
        self.alpha_bounds = alpha_bounds
        self.constraint_type = constraint_type
        self.initialization = initialization
        
        # Validate bounds
        if alpha_bounds[0] >= alpha_bounds[1]:
            raise ValueError("alpha_bounds[0] must be less than alpha_bounds[1]")
        
        # Build the network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self._get_activation(activation))
        
        # Output layer (unconstrained)
        layers.append(nn.Linear(neurons_per_layer, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Input normalization parameters
        self.register_buffer('x_mean', torch.tensor(0.5))
        self.register_buffer('x_std', torch.tensor(0.5))
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'elu': nn.ELU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using specified scheme."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                if self.initialization == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight)
                elif self.initialization == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight)
                elif self.initialization == 'kaiming_normal':
                    nn.init.kaiming_normal_(layer.weight, 
                                          nonlinearity=self.activation_name)
                elif self.initialization == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(layer.weight, 
                                           nonlinearity=self.activation_name)
                else:
                    # Default: small random initialization
                    nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                
                # Initialize biases to produce middle of range initially
                if layer == list(self.network.modules())[-1]:  # Output layer
                    mid_value = (self.alpha_bounds[0] + self.alpha_bounds[1]) / 2
                    nn.init.constant_(layer.bias, self._inverse_constraint(mid_value))
                else:
                    nn.init.zeros_(layer.bias)
    
    def _inverse_constraint(self, alpha: float) -> float:
        """Compute inverse of constraint function for initialization."""
        a, b = self.alpha_bounds
        
        if self.constraint_type == 'tanh':
            # alpha = (b-a)/2 * tanh(z) + (a+b)/2
            # z = atanh(2*(alpha - (a+b)/2)/(b-a))
            normalized = 2 * (alpha - (a + b) / 2) / (b - a)
            return np.arctanh(np.clip(normalized, -0.99, 0.99))
        
        elif self.constraint_type == 'sigmoid':
            # alpha = (b-a) * sigmoid(z) + a
            # z = logit((alpha - a)/(b - a))
            normalized = (alpha - a) / (b - a)
            return np.log(normalized / (1 - normalized + 1e-8))
        
        else:  # Default case
            return 0.0
    
    def set_normalization(self, x_stats: Tuple[float, float]) -> None:
        """
        Set input normalization parameters.
        
        Args:
            x_stats: (mean, std) for spatial coordinate
        """
        self.x_mean = torch.tensor(x_stats[0], device=self.x_mean.device)
        self.x_std = torch.tensor(x_stats[1], device=self.x_std.device)
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input coordinates for numerical stability."""
        return (x - self.x_mean) / (self.x_std + 1e-8)
    
    def apply_constraints(self, alpha_raw: torch.Tensor) -> torch.Tensor:
        """
        Apply constraints to ensure α(x) stays within physical bounds.
        
        Args:
            alpha_raw: Unconstrained network output
            
        Returns:
            Constrained fractional order values
        """
        a, b = self.alpha_bounds
        
        if self.constraint_type == 'tanh':
            # Maps (-∞, ∞) to (a, b)
            alpha = (b - a) / 2 * torch.tanh(alpha_raw) + (a + b) / 2
            
        elif self.constraint_type == 'sigmoid':
            # Maps (-∞, ∞) to (a, b)
            alpha = (b - a) * torch.sigmoid(alpha_raw) + a
            
        elif self.constraint_type == 'softplus':
            # Maps (-∞, ∞) to (a, ∞), then clip to (a, b)
            alpha = torch.clamp(a + torch.nn.functional.softplus(alpha_raw), a, b)
            
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
        
        return alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the order network.
        
        Args:
            x: Spatial coordinates [batch_size]
            
        Returns:
            Constrained fractional order values [batch_size, 1]
        """
        # Ensure input is 1D tensor
        if x.dim() == 0:
            x = x.unsqueeze(0)
        
        # Normalize input
        x_norm = self.normalize_input(x)
        
        # Add dimension for network input
        if x_norm.dim() == 1:
            x_norm = x_norm.unsqueeze(-1)
        
        # Forward pass (unconstrained)
        alpha_raw = self.network(x_norm)
        
        # Apply constraints
        alpha = self.apply_constraints(alpha_raw)
        
        return alpha
    
    def compute_derivatives(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spatial derivatives of α(x) using automatic differentiation.
        
        Args:
            x: Spatial coordinates requiring gradients
            
        Returns:
            Dictionary containing α, α_x, α_xx derivatives
        """
        # Ensure input requires gradients
        x = x.requires_grad_(True)
        
        # Forward pass
        alpha = self.forward(x)
        
        # First derivative
        alpha_x = torch.autograd.grad(
            outputs=alpha, inputs=x,
            grad_outputs=torch.ones_like(alpha),
            create_graph=True, retain_graph=True
        )[0]
        
        # Second derivative
        alpha_xx = torch.autograd.grad(
            outputs=alpha_x, inputs=x,
            grad_outputs=torch.ones_like(alpha_x),
            create_graph=True, retain_graph=True
        )[0]
        
        return {
            'alpha': alpha,
            'alpha_x': alpha_x,
            'alpha_xx': alpha_xx
        }
    
    def get_statistics(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics of the predicted fractional order.
        
        Args:
            x: Spatial coordinates for evaluation
            
        Returns:
            Dictionary with statistics
        """
        with torch.no_grad():
            alpha = self.forward(x)
            
            stats = {
                'mean': float(torch.mean(alpha)),
                'std': float(torch.std(alpha)),
                'min': float(torch.min(alpha)),
                'max': float(torch.max(alpha)),
                'range': float(torch.max(alpha) - torch.min(alpha))
            }
            
        return stats


class MultiScaleOrderNetwork(OrderNetwork):
    """
    Multi-scale order network for learning α(x) with different frequency components.
    
    This enhanced version uses multiple pathways to capture both smooth variations
    and sharp transitions in the fractional order function.
    """
    
    def __init__(self,
                 input_dim: int = 1,
                 base_neurons: int = 20,
                 num_scales: int = 3,
                 activation: str = 'tanh',
                 alpha_bounds: Tuple[float, float] = (1.0, 2.0),
                 constraint_type: str = 'tanh'):
        """
        Initialize multi-scale order network.
        
        Args:
            base_neurons: Base number of neurons per scale
            num_scales: Number of different scales/pathways
        """
        # Initialize parent class without building network
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.base_neurons = base_neurons
        self.num_scales = num_scales
        self.activation_name = activation
        self.alpha_bounds = alpha_bounds
        self.constraint_type = constraint_type
        
        # Build multi-scale network
        self._build_multiscale_network()
        
        # Input normalization parameters
        self.register_buffer('x_mean', torch.tensor(0.5))
        self.register_buffer('x_std', torch.tensor(0.5))
    
    def _build_multiscale_network(self) -> None:
        """Build multi-scale network architecture."""
        self.scale_networks = nn.ModuleList()
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        # Create networks for different scales
        for i in range(self.num_scales):
            # Each scale has different architecture complexity
            layers = []
            current_neurons = self.base_neurons * (2 ** i)
            num_layers = 2 + i  # Increasing complexity
            
            # Input layer
            layers.append(nn.Linear(self.input_dim, current_neurons))
            layers.append(self._get_activation(self.activation_name))
            
            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(current_neurons, current_neurons))
                layers.append(self._get_activation(self.activation_name))
            
            # Output layer
            layers.append(nn.Linear(current_neurons, 1))
            
            self.scale_networks.append(nn.Sequential(*layers))
        
        # Final combination layer
        self.combination_layer = nn.Sequential(
            nn.Linear(self.num_scales, self.num_scales),
            nn.Tanh(),
            nn.Linear(self.num_scales, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale forward pass."""
        # Normalize input
        x_norm = self.normalize_input(x)
        if x_norm.dim() == 1:
            x_norm = x_norm.unsqueeze(-1)
        
        # Get outputs from all scales
        scale_outputs = []
        for i, network in enumerate(self.scale_networks):
            # Apply different input transformations for different scales
            x_transformed = x_norm * (2 ** i)  # Different frequency scaling
            output = network(x_transformed)
            scale_outputs.append(output)
        
        # Combine scale outputs
        combined = torch.cat(scale_outputs, dim=-1)
        alpha_raw = self.combination_layer(combined)
        
        # Apply constraints
        alpha = self.apply_constraints(alpha_raw)
        
        return alpha


def create_order_network(config: dict) -> OrderNetwork:
    """
    Factory function to create order networks based on configuration.
    
    Args:
        config: Dictionary containing network configuration
        
    Returns:
        Configured order network
    """
    network_type = config.get('type', 'basic')
    
    if network_type == 'basic':
        return OrderNetwork(
            input_dim=config.get('input_dim', 1),
            hidden_layers=config.get('hidden_layers', 3),
            neurons_per_layer=config.get('neurons_per_layer', 30),
            activation=config.get('activation', 'tanh'),
            alpha_bounds=config.get('alpha_bounds', (1.0, 2.0)),
            constraint_type=config.get('constraint_type', 'tanh'),
            initialization=config.get('initialization', 'xavier_normal')
        )
    
    elif network_type == 'multiscale':
        return MultiScaleOrderNetwork(
            input_dim=config.get('input_dim', 1),
            base_neurons=config.get('base_neurons', 20),
            num_scales=config.get('num_scales', 3),
            activation=config.get('activation', 'tanh'),
            alpha_bounds=config.get('alpha_bounds', (1.0, 2.0)),
            constraint_type=config.get('constraint_type', 'tanh')
        )
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test basic order network
    print("Testing Order Network...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create network
    config = {
        'type': 'basic',
        'hidden_layers': 3,
        'neurons_per_layer': 30,
        'activation': 'tanh',
        'alpha_bounds': (1.0, 2.0),
        'constraint_type': 'tanh'
    }
    
    net = create_order_network(config).to(device)
    print(f"Network created with {sum(p.numel() for p in net.parameters())} parameters")
    
    # Test forward pass
    batch_size = 100
    x = torch.linspace(0, 1, batch_size, device=device)
    
    # Forward pass
    with torch.no_grad():
        alpha = net(x)
        print(f"Forward pass successful. Output shape: {alpha.shape}")
        print(f"Alpha range: [{float(torch.min(alpha)):.3f}, {float(torch.max(alpha)):.3f}]")
    
    # Test derivatives
    derivatives = net.compute_derivatives(x)
    print("Derivative computation successful:")
    for key, value in derivatives.items():
        print(f"  {key}: {value.shape}")
    
    # Test statistics
    stats = net.get_statistics(x)
    print("Statistics:", stats)
    
    # Test multi-scale network
    print("\nTesting Multi-Scale Order Network...")
    config_multiscale = {
        'type': 'multiscale',
        'base_neurons': 20,
        'num_scales': 3,
        'alpha_bounds': (1.0, 2.0)
    }
    
    net_multiscale = create_order_network(config_multiscale).to(device)
    print(f"Multi-scale network created with {sum(p.numel() for p in net_multiscale.parameters())} parameters")
    
    with torch.no_grad():
        alpha_multiscale = net_multiscale(x)
        print(f"Multi-scale forward pass successful. Output shape: {alpha_multiscale.shape}")
        print(f"Alpha range: [{float(torch.min(alpha_multiscale)):.3f}, {float(torch.max(alpha_multiscale)):.3f}]")
    
    print("All tests passed!")