"""
Solution Network (u_NN) for Variable-Order Fractional PDE Discovery

This module implements the neural network architecture that approximates
the solution u(x,t) of the variable-order fractional PDE. The network
is designed to work with automatic differentiation for computing spatial
and temporal derivatives required in the physics-informed training.

Author: Sakeeb Rahman
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List


class SolutionNetwork(nn.Module):
    """
    Neural network for approximating PDE solution u(x,t).
    
    This network serves as the u_NN component in the dual-network architecture
    for variable-order fractional PDE discovery. It takes spatio-temporal
    coordinates as input and outputs the predicted solution value.
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 hidden_layers: int = 4,
                 neurons_per_layer: int = 50,
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None,
                 initialization: str = 'xavier_normal'):
        """
        Initialize the solution network.
        
        Args:
            input_dim: Input dimension (2 for (x,t))
            hidden_layers: Number of hidden layers
            neurons_per_layer: Number of neurons in each hidden layer
            activation: Activation function ('tanh', 'relu', 'gelu', 'swish')
            output_activation: Output layer activation (None for linear)
            initialization: Weight initialization scheme
        """
        super(SolutionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation
        self.initialization = initialization
        
        # Build the network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self._get_activation(activation))
        
        # Output layer
        layers.append(nn.Linear(neurons_per_layer, 1))
        if output_activation:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Input normalization parameters (for numerical stability)
        self.register_buffer('x_mean', torch.tensor(0.5))
        self.register_buffer('x_std', torch.tensor(0.5))
        self.register_buffer('t_mean', torch.tensor(0.5))
        self.register_buffer('t_std', torch.tensor(0.5))
        
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
                
                # Initialize biases to zero
                nn.init.zeros_(layer.bias)
    
    def set_normalization(self, x_stats: Tuple[float, float], 
                         t_stats: Tuple[float, float]) -> None:
        """
        Set input normalization parameters.
        
        Args:
            x_stats: (mean, std) for spatial coordinate
            t_stats: (mean, std) for temporal coordinate
        """
        self.x_mean = torch.tensor(x_stats[0], device=self.x_mean.device)
        self.x_std = torch.tensor(x_stats[1], device=self.x_std.device)
        self.t_mean = torch.tensor(t_stats[0], device=self.t_mean.device)
        self.t_std = torch.tensor(t_stats[1], device=self.t_std.device)
    
    def normalize_input(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Normalize input coordinates for numerical stability."""
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        t_norm = (t - self.t_mean) / (self.t_std + 1e-8)
        return torch.stack([x_norm, t_norm], dim=-1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the solution network.
        
        Args:
            x: Spatial coordinates [batch_size]
            t: Temporal coordinates [batch_size]
            
        Returns:
            Predicted solution values [batch_size, 1]
        """
        # Ensure inputs are 1D tensors
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Normalize inputs
        inputs = self.normalize_input(x, t)
        
        # Forward pass
        output = self.network(inputs)
        
        return output
    
    def compute_derivatives(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        """
        Compute spatial and temporal derivatives using automatic differentiation.
        
        Args:
            x: Spatial coordinates requiring gradients
            t: Temporal coordinates requiring gradients
            
        Returns:
            Dictionary containing u, u_x, u_xx, u_t derivatives
        """
        # Ensure inputs require gradients
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        
        # Forward pass
        u = self.forward(x, t)
        
        # First derivatives
        u_x = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_t = torch.autograd.grad(
            outputs=u, inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        # Second spatial derivative
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        return {
            'u': u,
            'u_x': u_x,
            'u_xx': u_xx,
            'u_t': u_t
        }


class AdaptiveSolutionNetwork(SolutionNetwork):
    """
    Adaptive solution network with residual connections and normalization.
    
    Enhanced version of the basic solution network with modern deep learning
    techniques for improved training stability and performance.
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 hidden_layers: int = 4,
                 neurons_per_layer: int = 50,
                 activation: str = 'tanh',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = True,
                 use_residual: bool = True,
                 dropout_rate: float = 0.0):
        """
        Initialize adaptive solution network.
        
        Args:
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            dropout_rate: Dropout probability (0.0 for no dropout)
        """
        # Initialize parent class without building network
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        
        # Build enhanced network
        self._build_adaptive_network()
        self._initialize_weights()
        
        # Input normalization parameters
        self.register_buffer('x_mean', torch.tensor(0.5))
        self.register_buffer('x_std', torch.tensor(0.5))
        self.register_buffer('t_mean', torch.tensor(0.5))
        self.register_buffer('t_std', torch.tensor(0.5))
    
    def _build_adaptive_network(self) -> None:
        """Build the adaptive network architecture."""
        # Input projection
        self.input_layer = nn.Linear(self.input_dim, self.neurons_per_layer)
        
        # Hidden layers with normalization and residual connections
        self.hidden_layers_list = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        for i in range(self.hidden_layers):
            # Main layer
            layer = nn.Linear(self.neurons_per_layer, self.neurons_per_layer)
            self.hidden_layers_list.append(layer)
            
            # Normalization
            if self.use_batch_norm:
                norm = nn.BatchNorm1d(self.neurons_per_layer)
            elif self.use_layer_norm:
                norm = nn.LayerNorm(self.neurons_per_layer)
            else:
                norm = nn.Identity()
            self.norm_layers.append(norm)
            
            # Residual connection (identity mapping)
            if self.use_residual and i > 0:
                residual = nn.Linear(self.neurons_per_layer, self.neurons_per_layer)
            else:
                residual = nn.Identity()
            self.residual_layers.append(residual)
        
        # Output layer
        self.output_layer = nn.Linear(self.neurons_per_layer, 1)
        
        # Activation and dropout
        self.activation = self._get_activation(self.activation_name)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with residual connections."""
        # Normalize inputs
        inputs = self.normalize_input(x, t)
        
        # Input layer
        h = self.activation(self.input_layer(inputs))
        h = self.dropout(h)
        
        # Hidden layers with residual connections
        for i, (layer, norm, residual) in enumerate(zip(
            self.hidden_layers_list, self.norm_layers, self.residual_layers)):
            
            h_new = self.activation(norm(layer(h)))
            h_new = self.dropout(h_new)
            
            # Add residual connection
            if self.use_residual and i > 0:
                h = h_new + residual(h)
            else:
                h = h_new
        
        # Output layer
        output = self.output_layer(h)
        
        return output


def create_solution_network(config: dict) -> SolutionNetwork:
    """
    Factory function to create solution networks based on configuration.
    
    Args:
        config: Dictionary containing network configuration
        
    Returns:
        Configured solution network
    """
    network_type = config.get('type', 'basic')
    
    if network_type == 'basic':
        return SolutionNetwork(
            input_dim=config.get('input_dim', 2),
            hidden_layers=config.get('hidden_layers', 4),
            neurons_per_layer=config.get('neurons_per_layer', 50),
            activation=config.get('activation', 'tanh'),
            initialization=config.get('initialization', 'xavier_normal')
        )
    
    elif network_type == 'adaptive':
        return AdaptiveSolutionNetwork(
            input_dim=config.get('input_dim', 2),
            hidden_layers=config.get('hidden_layers', 4),
            neurons_per_layer=config.get('neurons_per_layer', 50),
            activation=config.get('activation', 'tanh'),
            use_batch_norm=config.get('use_batch_norm', False),
            use_layer_norm=config.get('use_layer_norm', True),
            use_residual=config.get('use_residual', True),
            dropout_rate=config.get('dropout_rate', 0.0)
        )
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test basic solution network
    print("Testing Solution Network...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create network
    config = {
        'type': 'basic',
        'hidden_layers': 4,
        'neurons_per_layer': 50,
        'activation': 'tanh'
    }
    
    net = create_solution_network(config).to(device)
    print(f"Network created with {sum(p.numel() for p in net.parameters())} parameters")
    
    # Test forward pass
    batch_size = 100
    x = torch.linspace(0, 1, batch_size, device=device)
    t = torch.linspace(0, 1, batch_size, device=device)
    
    # Forward pass
    with torch.no_grad():
        u = net(x, t)
        print(f"Forward pass successful. Output shape: {u.shape}")
    
    # Test derivatives
    derivatives = net.compute_derivatives(x, t)
    print("Derivative computation successful:")
    for key, value in derivatives.items():
        print(f"  {key}: {value.shape}")
    
    # Test adaptive network
    print("\nTesting Adaptive Solution Network...")
    config_adaptive = {
        'type': 'adaptive',
        'hidden_layers': 4,
        'neurons_per_layer': 50,
        'use_residual': True,
        'use_layer_norm': True
    }
    
    net_adaptive = create_solution_network(config_adaptive).to(device)
    print(f"Adaptive network created with {sum(p.numel() for p in net_adaptive.parameters())} parameters")
    
    with torch.no_grad():
        u_adaptive = net_adaptive(x, t)
        print(f"Adaptive forward pass successful. Output shape: {u_adaptive.shape}")
    
    print("All tests passed!")