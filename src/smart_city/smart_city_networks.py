"""
Smart City Neural Networks for 2D Multi-Physics Systems

This module extends the existing neural network architectures to handle
2D spatial domains and multi-physics coupling for smart city applications.
Supports temperature, air quality, and humidity field discovery with
spatially varying fractional orders.

Author: Sakeeb Rahman  
Date: 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.solution_network import SolutionNetwork, AdaptiveSolutionNetwork
    from models.order_network import OrderNetwork, MultiScaleOrderNetwork
except ImportError:
    # Fallback if imports fail
    print("Warning: Could not import base network classes. Using simplified implementations.")


class SmartCitySolutionNetwork:
    """
    Multi-field solution network for smart city applications.
    
    Handles coupled temperature, pollutant concentration, and humidity fields
    in 2D spatial + temporal domains (x, y, t).
    """
    
    def __init__(self,
                 input_dim: int = 3,  # (x, y, t)
                 output_fields: List[str] = ['temperature', 'pollutant', 'humidity'],
                 hidden_layers: int = 5,
                 neurons_per_layer: int = 80,
                 activation: str = 'tanh',
                 use_physics_constraints: bool = True,
                 field_coupling: bool = True):
        """
        Initialize multi-field solution network.
        
        Args:
            input_dim: Input dimension (3 for (x,y,t))
            output_fields: List of field names to predict
            hidden_layers: Number of hidden layers
            neurons_per_layer: Neurons per hidden layer
            activation: Activation function
            use_physics_constraints: Apply physical constraints to outputs
            field_coupling: Enable coupling between different fields
        """
        self.input_dim = input_dim
        self.output_fields = output_fields
        self.n_fields = len(output_fields)
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation
        self.use_physics_constraints = use_physics_constraints
        self.field_coupling = field_coupling
        
        # Initialize network architecture
        self._build_network()
        
        # Field-specific constraints
        self.field_constraints = {
            'temperature': {'min': -10.0, 'max': 50.0},  # Celsius
            'pollutant': {'min': 0.0, 'max': 500.0},     # μg/m³
            'humidity': {'min': 0.0, 'max': 100.0}       # %
        }
        
        # Coupling coefficients (learned during training)
        self.coupling_strength = 0.1
        
    def _build_network(self):
        """Build the multi-field neural network architecture."""
        # Simplified implementation using basic Python structures
        # In full implementation, this would use PyTorch/TensorFlow
        
        # Shared feature extraction layers
        self.shared_layers = []
        layer_sizes = [self.input_dim] + [self.neurons_per_layer] * (self.hidden_layers - 1)
        
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1,
                'biases': np.zeros(layer_sizes[i+1]),
                'activation': self.activation_name
            }
            self.shared_layers.append(layer)
        
        # Field-specific output layers
        self.field_layers = {}
        for field in self.output_fields:
            self.field_layers[field] = {
                'weights': np.random.randn(self.neurons_per_layer, 1) * 0.1,
                'biases': np.zeros(1)
            }
        
        # Coupling layers (if enabled)
        if self.field_coupling:
            self.coupling_layers = {}
            for i, field1 in enumerate(self.output_fields):
                self.coupling_layers[field1] = {}
                for j, field2 in enumerate(self.output_fields):
                    if i != j:
                        self.coupling_layers[field1][field2] = np.random.randn() * 0.01
    
    def _activation_function(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'linear':
            return x
        else:
            return np.tanh(x)  # Default
    
    def forward(self, coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass through the multi-field network.
        
        Args:
            coordinates: Input coordinates [N, 3] for (x, y, t)
            
        Returns:
            Dictionary with predicted fields
        """
        # Normalize coordinates
        x_norm = coordinates.copy()
        x_norm[:, 0] = (x_norm[:, 0] - 5.0) / 5.0  # Assuming domain [0, 10] km
        x_norm[:, 1] = (x_norm[:, 1] - 5.0) / 5.0  # Assuming domain [0, 10] km  
        x_norm[:, 2] = (x_norm[:, 2] - 12.0) / 12.0  # Assuming domain [0, 24] hours
        
        # Forward through shared layers
        h = x_norm
        for layer in self.shared_layers:
            h = h @ layer['weights'] + layer['biases']
            h = self._activation_function(h, layer['activation'])
        
        # Field-specific outputs
        field_outputs = {}
        for field in self.output_fields:
            output = h @ self.field_layers[field]['weights'] + self.field_layers[field]['biases']
            
            # Apply physical constraints
            if self.use_physics_constraints and field in self.field_constraints:
                constraints = self.field_constraints[field]
                output = constraints['min'] + (constraints['max'] - constraints['min']) * \
                        self._activation_function(output, 'sigmoid')
            
            field_outputs[field] = output.flatten()
        
        # Apply field coupling
        if self.field_coupling:
            coupled_outputs = field_outputs.copy()
            for field1 in self.output_fields:
                coupling_effect = 0.0
                for field2 in self.output_fields:
                    if field1 != field2:
                        coupling_coeff = self.coupling_layers[field1][field2]
                        coupling_effect += coupling_coeff * np.mean(field_outputs[field2])
                
                coupled_outputs[field1] += self.coupling_strength * coupling_effect
            
            field_outputs = coupled_outputs
        
        return field_outputs
    
    def compute_derivatives(self, coordinates: np.ndarray, 
                          field_name: str = 'temperature') -> Dict[str, np.ndarray]:
        """
        Compute spatial and temporal derivatives using finite differences.
        
        Args:
            coordinates: Input coordinates [N, 3]
            field_name: Which field to compute derivatives for
            
        Returns:
            Dictionary with derivatives (u, u_x, u_y, u_t, u_xx, u_yy)
        """
        h = 1e-5  # Finite difference step
        
        # Base prediction
        base_output = self.forward(coordinates)
        u = base_output[field_name]
        
        # Spatial derivatives
        coords_x_plus = coordinates.copy()
        coords_x_plus[:, 0] += h
        u_x = (self.forward(coords_x_plus)[field_name] - u) / h
        
        coords_y_plus = coordinates.copy()
        coords_y_plus[:, 1] += h
        u_y = (self.forward(coords_y_plus)[field_name] - u) / h
        
        # Temporal derivative
        coords_t_plus = coordinates.copy()
        coords_t_plus[:, 2] += h
        u_t = (self.forward(coords_t_plus)[field_name] - u) / h
        
        # Second derivatives
        coords_x_minus = coordinates.copy()
        coords_x_minus[:, 0] -= h
        u_x_minus = (self.forward(coords_x_minus)[field_name] - u) / (-h)
        u_xx = (u_x - u_x_minus) / h
        
        coords_y_minus = coordinates.copy()
        coords_y_minus[:, 1] -= h
        u_y_minus = (self.forward(coords_y_minus)[field_name] - u) / (-h)
        u_yy = (u_y - u_y_minus) / h
        
        return {
            'u': u,
            'u_x': u_x,
            'u_y': u_y,
            'u_t': u_t,
            'u_xx': u_xx,
            'u_yy': u_yy
        }


class SmartCityOrderNetwork:
    """
    2D Fractional order network for smart city applications.
    
    Predicts spatially varying fractional orders α(x,y) for different
    physical processes (thermal, pollutant, humidity transport).
    """
    
    def __init__(self,
                 input_dim: int = 2,  # (x, y)
                 output_fields: List[str] = ['alpha_T', 'alpha_C', 'alpha_H'],
                 hidden_layers: int = 4,
                 neurons_per_layer: int = 60,
                 activation: str = 'tanh',
                 alpha_bounds: Tuple[float, float] = (1.0, 2.0),
                 multi_scale: bool = True):
        """
        Initialize 2D fractional order network.
        
        Args:
            input_dim: Input dimension (2 for (x,y))
            output_fields: List of fractional order fields
            hidden_layers: Number of hidden layers
            neurons_per_layer: Neurons per layer
            activation: Activation function
            alpha_bounds: Physical bounds for fractional orders
            multi_scale: Use multi-scale architecture
        """
        self.input_dim = input_dim
        self.output_fields = output_fields
        self.n_fields = len(output_fields)
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_name = activation
        self.alpha_bounds = alpha_bounds
        self.multi_scale = multi_scale
        
        # Build network
        self._build_network()
        
    def _build_network(self):
        """Build the 2D fractional order network."""
        if self.multi_scale:
            # Multi-scale architecture for capturing different spatial frequencies
            self.scales = [1, 2, 4]  # Different spatial scales
            self.scale_networks = []
            
            for scale in self.scales:
                network = []
                layer_sizes = [self.input_dim] + [self.neurons_per_layer // scale] * self.hidden_layers
                
                for i in range(len(layer_sizes) - 1):
                    layer = {
                        'weights': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1 / scale,
                        'biases': np.zeros(layer_sizes[i+1]),
                        'scale': scale
                    }
                    network.append(layer)
                
                self.scale_networks.append(network)
            
            # Combination layer
            total_features = sum(self.neurons_per_layer // scale for scale in self.scales)
            self.combination_weights = np.random.randn(total_features, self.n_fields) * 0.1
            self.combination_biases = np.zeros(self.n_fields)
            
        else:
            # Standard architecture
            self.main_network = []
            layer_sizes = [self.input_dim] + [self.neurons_per_layer] * self.hidden_layers + [self.n_fields]
            
            for i in range(len(layer_sizes) - 1):
                layer = {
                    'weights': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1,
                    'biases': np.zeros(layer_sizes[i+1])
                }
                self.main_network.append(layer)
    
    def _apply_constraints(self, alpha_raw: np.ndarray) -> np.ndarray:
        """Apply physical constraints to fractional orders."""
        a, b = self.alpha_bounds
        # Use tanh to map (-∞, ∞) to (a, b)
        alpha_constrained = (b - a) / 2 * np.tanh(alpha_raw) + (a + b) / 2
        return alpha_constrained
    
    def forward(self, coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass through the 2D order network.
        
        Args:
            coordinates: Input coordinates [N, 2] for (x, y)
            
        Returns:
            Dictionary with predicted fractional order fields
        """
        # Normalize coordinates
        coords_norm = coordinates.copy()
        coords_norm[:, 0] = (coords_norm[:, 0] - 5.0) / 5.0  # Domain [0, 10] km
        coords_norm[:, 1] = (coords_norm[:, 1] - 5.0) / 5.0  # Domain [0, 10] km
        
        if self.multi_scale:
            # Multi-scale processing
            scale_features = []
            
            for scale, network in zip(self.scales, self.scale_networks):
                # Apply spatial scaling to inputs
                scaled_coords = coords_norm * scale
                
                h = scaled_coords
                for layer in network:
                    h = h @ layer['weights'] + layer['biases']
                    h = self._activation_function(h, self.activation_name)
                
                scale_features.append(h)
            
            # Concatenate multi-scale features
            combined_features = np.concatenate(scale_features, axis=1)
            
            # Final combination
            alpha_raw = combined_features @ self.combination_weights + self.combination_biases
            
        else:
            # Standard forward pass
            h = coords_norm
            for layer in self.main_network[:-1]:
                h = h @ layer['weights'] + layer['biases']
                h = self._activation_function(h, self.activation_name)
            
            # Output layer (no activation)
            alpha_raw = h @ self.main_network[-1]['weights'] + self.main_network[-1]['biases']
        
        # Apply constraints
        alpha_constrained = self._apply_constraints(alpha_raw)
        
        # Convert to field dictionary
        alpha_fields = {}
        for i, field in enumerate(self.output_fields):
            alpha_fields[field] = alpha_constrained[:, i]
        
        return alpha_fields
    
    def _activation_function(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        else:
            return np.tanh(x)
    
    def compute_derivatives(self, coordinates: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute spatial derivatives of fractional order fields.
        
        Args:
            coordinates: Input coordinates [N, 2]
            
        Returns:
            Nested dictionary with derivatives for each field
        """
        h = 1e-5
        
        # Base prediction
        base_output = self.forward(coordinates)
        
        derivatives = {}
        for field in self.output_fields:
            # Spatial derivatives
            coords_x_plus = coordinates.copy()
            coords_x_plus[:, 0] += h
            alpha_x = (self.forward(coords_x_plus)[field] - base_output[field]) / h
            
            coords_y_plus = coordinates.copy()
            coords_y_plus[:, 1] += h
            alpha_y = (self.forward(coords_y_plus)[field] - base_output[field]) / h
            
            # Second derivatives
            coords_x_minus = coordinates.copy()
            coords_x_minus[:, 0] -= h
            alpha_x_minus = (self.forward(coords_x_minus)[field] - base_output[field]) / (-h)
            alpha_xx = (alpha_x - alpha_x_minus) / h
            
            coords_y_minus = coordinates.copy()
            coords_y_minus[:, 1] -= h
            alpha_y_minus = (self.forward(coords_y_minus)[field] - base_output[field]) / (-h)
            alpha_yy = (alpha_y - alpha_y_minus) / h
            
            derivatives[field] = {
                'alpha': base_output[field],
                'alpha_x': alpha_x,
                'alpha_y': alpha_y,
                'alpha_xx': alpha_xx,
                'alpha_yy': alpha_yy
            }
        
        return derivatives


class SmartCityNetworkFactory:
    """Factory for creating smart city networks with predefined configurations."""
    
    @staticmethod
    def create_temperature_solution_network(config: Dict = None) -> SmartCitySolutionNetwork:
        """Create network optimized for temperature field prediction."""
        default_config = {
            'output_fields': ['temperature'],
            'hidden_layers': 5,
            'neurons_per_layer': 80,
            'activation': 'tanh',
            'use_physics_constraints': True,
            'field_coupling': False
        }
        
        if config:
            default_config.update(config)
        
        return SmartCitySolutionNetwork(**default_config)
    
    @staticmethod
    def create_air_quality_solution_network(config: Dict = None) -> SmartCitySolutionNetwork:
        """Create network optimized for air quality prediction."""
        default_config = {
            'output_fields': ['pollutant'],
            'hidden_layers': 6,
            'neurons_per_layer': 100,
            'activation': 'gelu',
            'use_physics_constraints': True,
            'field_coupling': False
        }
        
        if config:
            default_config.update(config)
        
        return SmartCitySolutionNetwork(**default_config)
    
    @staticmethod
    def create_multi_physics_solution_network(config: Dict = None) -> SmartCitySolutionNetwork:
        """Create network for coupled multi-physics prediction."""
        default_config = {
            'output_fields': ['temperature', 'pollutant', 'humidity'],
            'hidden_layers': 6,
            'neurons_per_layer': 120,
            'activation': 'tanh',
            'use_physics_constraints': True,
            'field_coupling': True
        }
        
        if config:
            default_config.update(config)
        
        return SmartCitySolutionNetwork(**default_config)
    
    @staticmethod
    def create_thermal_order_network(config: Dict = None) -> SmartCityOrderNetwork:
        """Create network for thermal fractional order discovery."""
        default_config = {
            'output_fields': ['alpha_T'],
            'hidden_layers': 4,
            'neurons_per_layer': 60,
            'activation': 'tanh',
            'alpha_bounds': (1.0, 2.0),
            'multi_scale': True
        }
        
        if config:
            default_config.update(config)
        
        return SmartCityOrderNetwork(**default_config)
    
    @staticmethod
    def create_multi_field_order_network(config: Dict = None) -> SmartCityOrderNetwork:
        """Create network for multi-field fractional order discovery."""
        default_config = {
            'output_fields': ['alpha_T', 'alpha_C', 'alpha_H'],
            'hidden_layers': 5,
            'neurons_per_layer': 80,
            'activation': 'tanh',
            'alpha_bounds': (1.0, 2.0),
            'multi_scale': True
        }
        
        if config:
            default_config.update(config)
        
        return SmartCityOrderNetwork(**default_config)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Smart City Neural Networks...")
    
    # Test multi-physics solution network
    solution_net = SmartCityNetworkFactory.create_multi_physics_solution_network()
    print(f"✓ Multi-physics solution network created")
    
    # Test coordinates
    n_points = 100
    coords_3d = np.random.rand(n_points, 3) * [10, 10, 24]  # (x, y, t)
    coords_2d = coords_3d[:, :2]  # (x, y)
    
    # Test solution network
    solution_output = solution_net.forward(coords_3d)
    print(f"✓ Solution network forward pass: {list(solution_output.keys())}")
    for field, values in solution_output.items():
        print(f"  {field}: range [{np.min(values):.2f}, {np.max(values):.2f}]")
    
    # Test derivatives
    temp_derivatives = solution_net.compute_derivatives(coords_3d, 'temperature')
    print(f"✓ Temperature derivatives computed: {list(temp_derivatives.keys())}")
    
    # Test multi-field order network
    order_net = SmartCityNetworkFactory.create_multi_field_order_network()
    print(f"✓ Multi-field order network created")
    
    # Test order network
    order_output = order_net.forward(coords_2d)
    print(f"✓ Order network forward pass: {list(order_output.keys())}")
    for field, values in order_output.items():
        print(f"  {field}: range [{np.min(values):.3f}, {np.max(values):.3f}]")
    
    # Test order derivatives
    order_derivatives = order_net.compute_derivatives(coords_2d)
    print(f"✓ Order derivatives computed for: {list(order_derivatives.keys())}")
    
    print("\\nAll smart city neural network tests passed!")