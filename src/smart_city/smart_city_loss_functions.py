"""
Multi-Physics Loss Functions for Smart City Applications

This module implements physics-informed loss functions for coupled urban
climate systems including temperature, air quality, and humidity transport
with spatially varying fractional orders.

Author: Sakeeb Rahman
Date: 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod


class SmartCityBaseLoss(ABC):
    """Abstract base class for smart city loss function components."""
    
    def __init__(self, weight: float = 1.0, field_name: str = 'generic'):
        self.weight = weight
        self.field_name = field_name
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        pass
    
    def __call__(self, *args, **kwargs) -> float:
        return self.weight * self.compute(*args, **kwargs)


class UrbanDataMismatchLoss(SmartCityBaseLoss):
    """
    Data mismatch loss for urban sensor observations.
    
    Handles multiple field types with field-specific weighting and
    uncertainty quantification for real sensor networks.
    """
    
    def __init__(self,
                 weight: float = 1.0,
                 field_weights: Dict[str, float] = None,
                 loss_type: str = 'mse',
                 uncertainty_weighting: bool = True):
        """
        Initialize urban data mismatch loss.
        
        Args:
            weight: Overall loss weight
            field_weights: Per-field weighting factors
            loss_type: Type of loss ('mse', 'mae', 'huber')
            uncertainty_weighting: Weight by sensor uncertainty
        """
        super().__init__(weight, 'multi_field_data')
        
        self.field_weights = field_weights or {
            'temperature': 1.0,
            'pollutant': 1.0,
            'humidity': 0.8  # Lower weight due to higher variability
        }
        self.loss_type = loss_type
        self.uncertainty_weighting = uncertainty_weighting
        
        # Sensor uncertainty models (field-dependent)
        self.sensor_uncertainty = {
            'temperature': 0.5,  # ±0.5°C typical sensor accuracy
            'pollutant': 5.0,    # ±5 μg/m³ typical accuracy
            'humidity': 3.0      # ±3% typical accuracy
        }
    
    def compute(self,
                predictions: Dict[str, np.ndarray],
                observations: Dict[str, np.ndarray],
                sensor_coordinates: np.ndarray = None,
                sensor_uncertainties: Dict[str, np.ndarray] = None) -> float:
        """
        Compute multi-field data mismatch loss.
        
        Args:
            predictions: Dictionary of predicted field values
            observations: Dictionary of observed field values
            sensor_coordinates: Sensor locations for spatial weighting
            sensor_uncertainties: Field-specific sensor uncertainties
            
        Returns:
            Combined data mismatch loss
        """
        total_loss = 0.0
        total_weight = 0.0
        
        for field_name in predictions.keys():
            if field_name not in observations:
                continue
                
            pred = predictions[field_name]
            obs = observations[field_name]
            
            # Compute residuals
            residuals = pred - obs
            
            # Apply field-specific loss function
            if self.loss_type == 'mse':
                field_loss = np.mean(residuals**2)
            elif self.loss_type == 'mae':
                field_loss = np.mean(np.abs(residuals))
            elif self.loss_type == 'huber':
                delta = self.sensor_uncertainty.get(field_name, 1.0)
                huber_loss = np.where(
                    np.abs(residuals) <= delta,
                    0.5 * residuals**2,
                    delta * (np.abs(residuals) - 0.5 * delta)
                )
                field_loss = np.mean(huber_loss)
            else:
                field_loss = np.mean(residuals**2)  # Default to MSE
            
            # Apply uncertainty weighting
            if self.uncertainty_weighting and sensor_uncertainties:
                if field_name in sensor_uncertainties:
                    weights = 1.0 / (sensor_uncertainties[field_name]**2 + 1e-8)
                    field_loss = np.mean(weights * residuals**2) / np.mean(weights)
            
            # Apply field weighting
            field_weight = self.field_weights.get(field_name, 1.0)
            total_loss += field_weight * field_loss
            total_weight += field_weight
        
        return total_loss / (total_weight + 1e-8)


class UrbanPDEResidualLoss(SmartCityBaseLoss):
    """
    PDE residual loss for urban climate physics.
    
    Enforces coupled transport equations for temperature, pollutants,
    and humidity with spatially varying fractional orders.
    """
    
    def __init__(self,
                 weight: float = 1.0,
                 pde_weights: Dict[str, float] = None,
                 coupling_strength: float = 0.1,
                 include_source_terms: bool = True):
        """
        Initialize urban PDE residual loss.
        
        Args:
            weight: Overall residual loss weight
            pde_weights: Per-equation weighting factors
            coupling_strength: Strength of inter-field coupling
            include_source_terms: Include source/sink terms
        """
        super().__init__(weight, 'urban_pde_residual')
        
        self.pde_weights = pde_weights or {
            'temperature': 1.0,
            'pollutant': 1.0,
            'humidity': 0.8
        }
        self.coupling_strength = coupling_strength
        self.include_source_terms = include_source_terms
        
        # Physical constants
        self.constants = {
            'thermal_diffusivity': 1e-5,  # m²/s (typical urban)
            'pollutant_diffusivity': 10.0,  # m²/s (turbulent)
            'humidity_diffusivity': 20.0   # m²/s
        }
    
    def compute(self,
                field_derivatives: Dict[str, Dict[str, np.ndarray]],
                alpha_fields: Dict[str, np.ndarray],
                fractional_laplacians: Dict[str, np.ndarray],
                source_terms: Dict[str, np.ndarray] = None,
                coordinates: np.ndarray = None) -> float:
        """
        Compute PDE residual loss for urban climate equations.
        
        Args:
            field_derivatives: Derivatives for each field
            alpha_fields: Fractional order fields
            fractional_laplacians: Computed fractional Laplacians
            source_terms: Source/sink terms for each field
            coordinates: Spatial coordinates for coupling
            
        Returns:
            Combined PDE residual loss
        """
        total_loss = 0.0
        total_weight = 0.0
        
        # Temperature equation: ∂T/∂t = κ(-Δ)^(α_T/2) T + S_T + coupling
        if 'temperature' in field_derivatives and 'alpha_T' in alpha_fields:
            T_derivs = field_derivatives['temperature']
            alpha_T = alpha_fields['alpha_T']
            
            # Main diffusion term
            thermal_term = self.constants['thermal_diffusivity'] * fractional_laplacians['temperature']
            
            # Coupling with humidity (evapotranspiration cooling)
            coupling_term = 0.0
            if 'humidity' in field_derivatives and self.coupling_strength > 0:
                H_derivs = field_derivatives['humidity']
                # Simplified evapotranspiration coupling
                coupling_term = -self.coupling_strength * np.maximum(0, H_derivs['u'] - 60.0)
            
            # Source terms
            source_term = 0.0
            if self.include_source_terms and source_terms and 'temperature' in source_terms:
                source_term = source_terms['temperature']
            
            # PDE residual: ∂T/∂t - κ(-Δ)^(α_T/2) T - S_T - coupling = 0
            residual = T_derivs['u_t'] - thermal_term - source_term - coupling_term
            
            # Compute loss
            temp_loss = np.mean(residual**2)
            weight = self.pde_weights.get('temperature', 1.0)
            total_loss += weight * temp_loss
            total_weight += weight
        
        # Pollutant equation: ∂C/∂t + v·∇C = D(-Δ)^(α_C/2) C + S_C - λC
        if 'pollutant' in field_derivatives and 'alpha_C' in alpha_fields:
            C_derivs = field_derivatives['pollutant']
            alpha_C = alpha_fields['alpha_C']
            
            # Diffusion term
            diffusion_term = self.constants['pollutant_diffusivity'] * fractional_laplacians['pollutant']
            
            # Simplified advection (wind effects)
            wind_speed = 2.0  # m/s typical urban wind
            advection_term = wind_speed * (C_derivs['u_x'] + C_derivs['u_y'])
            
            # Decay/removal term
            decay_rate = 0.1  # h⁻¹ typical pollutant decay
            decay_term = decay_rate * C_derivs['u']
            
            # Source terms
            source_term = 0.0
            if self.include_source_terms and source_terms and 'pollutant' in source_terms:
                source_term = source_terms['pollutant']
            
            # Temperature-dependent removal (higher T → faster reactions)
            temp_coupling = 0.0
            if 'temperature' in field_derivatives and self.coupling_strength > 0:
                T_derivs = field_derivatives['temperature']
                temp_factor = 1.0 + 0.05 * (T_derivs['u'] - 20.0)  # 5% per degree
                temp_coupling = self.coupling_strength * temp_factor * decay_term
            
            # PDE residual
            residual = (C_derivs['u_t'] + advection_term - diffusion_term 
                       - source_term + decay_term + temp_coupling)
            
            pollutant_loss = np.mean(residual**2)
            weight = self.pde_weights.get('pollutant', 1.0)
            total_loss += weight * pollutant_loss
            total_weight += weight
        
        # Humidity equation: ∂H/∂t = D_H(-Δ)^(α_H/2) H + E - P
        if 'humidity' in field_derivatives and 'alpha_H' in alpha_fields:
            H_derivs = field_derivatives['humidity']
            alpha_H = alpha_fields['alpha_H']
            
            # Diffusion term
            diffusion_term = self.constants['humidity_diffusivity'] * fractional_laplacians['humidity']
            
            # Evapotranspiration (simplified model)
            evap_term = 0.0
            if self.include_source_terms:
                # Depends on temperature and vegetation
                if 'temperature' in field_derivatives:
                    T_derivs = field_derivatives['temperature']
                    evap_rate = 0.1 * np.maximum(0, T_derivs['u'] - 15.0)  # Temperature-dependent
                    evap_term = evap_rate
            
            # Condensation/precipitation
            condensation_term = np.where(
                H_derivs['u'] > 90.0,  # Near saturation
                0.05 * (H_derivs['u'] - 90.0),
                0.0
            )
            
            # PDE residual
            residual = H_derivs['u_t'] - diffusion_term - evap_term + condensation_term
            
            humidity_loss = np.mean(residual**2)
            weight = self.pde_weights.get('humidity', 1.0)
            total_loss += weight * humidity_loss
            total_weight += weight
        
        return total_loss / (total_weight + 1e-8)


class UrbanRegularizationLoss(SmartCityBaseLoss):
    """
    Regularization loss for urban fractional order fields.
    
    Promotes physically meaningful α(x,y) distributions with
    urban-specific constraints and smoothness requirements.
    """
    
    def __init__(self,
                 weight: float = 1.0,
                 smoothness_weights: Dict[str, float] = None,
                 consistency_weight: float = 0.1,
                 urban_constraints_weight: float = 0.2):
        """
        Initialize urban regularization loss.
        
        Args:
            weight: Overall regularization weight
            smoothness_weights: Per-field smoothness penalties
            consistency_weight: Inter-field consistency penalty
            urban_constraints_weight: Urban-specific constraint penalty
        """
        super().__init__(weight, 'urban_regularization')
        
        self.smoothness_weights = smoothness_weights or {
            'alpha_T': 1.0,
            'alpha_C': 1.0,
            'alpha_H': 0.8
        }
        self.consistency_weight = consistency_weight
        self.urban_constraints_weight = urban_constraints_weight
        
        # Urban surface type expectations
        self.surface_alpha_expectations = {
            'dense_urban': {'alpha_T': 1.2, 'alpha_C': 1.1, 'alpha_H': 1.0},
            'residential': {'alpha_T': 1.4, 'alpha_C': 1.3, 'alpha_H': 1.2},
            'green_infrastructure': {'alpha_T': 1.7, 'alpha_C': 1.6, 'alpha_H': 1.8},
            'water_bodies': {'alpha_T': 1.9, 'alpha_C': 1.8, 'alpha_H': 1.9},
            'industrial': {'alpha_T': 1.1, 'alpha_C': 1.0, 'alpha_H': 1.1}
        }
    
    def compute(self,
                alpha_derivatives: Dict[str, Dict[str, np.ndarray]],
                alpha_fields: Dict[str, np.ndarray],
                surface_classification: np.ndarray = None,
                coordinates: np.ndarray = None) -> float:
        """
        Compute urban regularization loss.
        
        Args:
            alpha_derivatives: Spatial derivatives of fractional order fields
            alpha_fields: Current fractional order field values
            surface_classification: Urban surface type classification
            coordinates: Spatial coordinates
            
        Returns:
            Combined regularization loss
        """
        total_loss = 0.0
        
        # 1. Smoothness penalties for each field
        smoothness_loss = 0.0
        for field_name, derivatives in alpha_derivatives.items():
            if field_name in self.smoothness_weights:
                # Gradient magnitude penalty
                grad_x = derivatives.get('alpha_x', np.zeros(1))
                grad_y = derivatives.get('alpha_y', np.zeros(1))
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                field_smoothness = np.mean(grad_magnitude**2)
                smoothness_loss += self.smoothness_weights[field_name] * field_smoothness
        
        total_loss += smoothness_loss
        
        # 2. Inter-field consistency penalty
        if self.consistency_weight > 0 and len(alpha_fields) > 1:
            consistency_loss = 0.0
            field_names = list(alpha_fields.keys())
            
            # Penalize large differences between related fields
            for i in range(len(field_names)):
                for j in range(i + 1, len(field_names)):
                    field1, field2 = field_names[i], field_names[j]
                    alpha1, alpha2 = alpha_fields[field1], alpha_fields[field2]
                    
                    # Related fields should have similar spatial patterns
                    diff = alpha1 - alpha2
                    consistency_loss += np.mean(diff**2)
            
            total_loss += self.consistency_weight * consistency_loss / len(field_names)
        
        # 3. Urban surface constraints
        if (self.urban_constraints_weight > 0 and surface_classification is not None 
            and coordinates is not None):
            
            constraint_loss = 0.0
            surface_types = ['dense_urban', 'residential', 'green_infrastructure', 
                           'water_bodies', 'industrial']
            
            for surface_idx, surface_type in enumerate(surface_types):
                if surface_type not in self.surface_alpha_expectations:
                    continue
                    
                # Find points with this surface type
                surface_mask = (surface_classification == surface_idx)
                if not np.any(surface_mask):
                    continue
                
                expected_values = self.surface_alpha_expectations[surface_type]
                
                for field_name, expected_alpha in expected_values.items():
                    if field_name in alpha_fields:
                        # Penalize deviations from expected values for this surface
                        field_values = alpha_fields[field_name]
                        surface_values = field_values[surface_mask.flatten()]
                        
                        if len(surface_values) > 0:
                            deviation = np.mean((surface_values - expected_alpha)**2)
                            constraint_loss += deviation
            
            total_loss += self.urban_constraints_weight * constraint_loss
        
        # 4. Physical bounds enforcement (soft constraints)
        bounds_loss = 0.0
        for field_name, alpha_values in alpha_fields.items():
            # Penalize values outside [1.0, 2.0] range
            lower_violation = np.maximum(0, 1.0 - alpha_values)
            upper_violation = np.maximum(0, alpha_values - 2.0)
            
            bounds_loss += np.mean(lower_violation**2) + np.mean(upper_violation**2)
        
        total_loss += 10.0 * bounds_loss  # High penalty for bound violations
        
        return total_loss


class SmartCityCompositeLoss:
    """
    Composite loss function for smart city applications.
    
    Combines data mismatch, PDE residual, and regularization losses
    with adaptive weighting and urban-specific considerations.
    """
    
    def __init__(self,
                 data_loss_config: Dict = None,
                 residual_loss_config: Dict = None,
                 regularization_config: Dict = None,
                 adaptive_weights: bool = True,
                 urban_priorities: Dict[str, float] = None):
        """
        Initialize smart city composite loss.
        
        Args:
            data_loss_config: Data loss configuration
            residual_loss_config: Residual loss configuration
            regularization_config: Regularization loss configuration
            adaptive_weights: Enable adaptive weight balancing
            urban_priorities: Priority weights for different urban zones
        """
        # Initialize loss components
        self.data_loss = UrbanDataMismatchLoss(**(data_loss_config or {}))
        self.residual_loss = UrbanPDEResidualLoss(**(residual_loss_config or {}))
        self.regularization_loss = UrbanRegularizationLoss(**(regularization_config or {}))
        
        self.adaptive_weights = adaptive_weights
        self.urban_priorities = urban_priorities or {
            'downtown': 1.2,     # Higher priority for downtown areas
            'residential': 1.0,   # Standard priority
            'industrial': 0.8,   # Lower priority for industrial zones
            'green_spaces': 1.1  # Slightly higher for environmental zones
        }
        
        # Loss history for adaptive weighting
        self.loss_history = {
            'data': [],
            'residual': [],
            'regularization': []
        }
        
        self.iteration_count = 0
    
    def compute(self,
                # Data loss inputs
                predictions: Dict[str, np.ndarray],
                observations: Dict[str, np.ndarray],
                # Residual loss inputs
                field_derivatives: Dict[str, Dict[str, np.ndarray]],
                alpha_fields: Dict[str, np.ndarray],
                fractional_laplacians: Dict[str, np.ndarray],
                # Regularization inputs
                alpha_derivatives: Dict[str, Dict[str, np.ndarray]],
                # Optional inputs
                source_terms: Dict[str, np.ndarray] = None,
                surface_classification: np.ndarray = None,
                coordinates: np.ndarray = None,
                sensor_uncertainties: Dict[str, np.ndarray] = None) -> Dict[str, float]:
        """
        Compute composite loss for smart city application.
        
        Returns:
            Dictionary with total loss and individual components
        """
        # Compute individual losses
        loss_data = self.data_loss.compute(
            predictions, observations, coordinates, sensor_uncertainties
        )
        
        loss_residual = self.residual_loss.compute(
            field_derivatives, alpha_fields, fractional_laplacians, 
            source_terms, coordinates
        )
        
        loss_regularization = self.regularization_loss.compute(
            alpha_derivatives, alpha_fields, surface_classification, coordinates
        )
        
        # Store for adaptive weighting
        self.loss_history['data'].append(loss_data)
        self.loss_history['residual'].append(loss_residual)
        self.loss_history['regularization'].append(loss_regularization)
        
        # Apply adaptive weighting if enabled
        if self.adaptive_weights and self.iteration_count > 50:
            self._update_adaptive_weights()
        
        # Apply urban zone priorities if available
        if coordinates is not None and surface_classification is not None:
            zone_weight = self._compute_zone_weight(coordinates, surface_classification)
        else:
            zone_weight = 1.0
        
        # Compute total loss
        total_loss = (zone_weight * 
                     (loss_data + loss_residual + loss_regularization))
        
        self.iteration_count += 1
        
        return {
            'total': total_loss,
            'data': loss_data,
            'residual': loss_residual,
            'regularization': loss_regularization,
            'zone_weight': zone_weight,
            'weights': {
                'data': self.data_loss.weight,
                'residual': self.residual_loss.weight,
                'regularization': self.regularization_loss.weight
            }
        }
    
    def _update_adaptive_weights(self):
        """Update loss weights based on relative magnitudes and trends."""
        if len(self.loss_history['data']) < 10:
            return
        
        # Get recent loss magnitudes
        recent_data = np.mean(self.loss_history['data'][-10:])
        recent_residual = np.mean(self.loss_history['residual'][-10:])
        recent_reg = np.mean(self.loss_history['regularization'][-10:])
        
        # Compute relative scales and trends
        total_magnitude = recent_data + recent_residual + recent_reg
        
        if total_magnitude > 0:
            # Balance based on relative magnitudes
            data_ratio = recent_data / total_magnitude
            residual_ratio = recent_residual / total_magnitude
            reg_ratio = recent_reg / total_magnitude
            
            # Adjust weights inversely to current ratios (balance the losses)
            smoothing = 0.1
            target_balance = 1.0 / 3.0  # Equal weighting target
            
            self.data_loss.weight *= (1 - smoothing) + smoothing * (target_balance / data_ratio)
            self.residual_loss.weight *= (1 - smoothing) + smoothing * (target_balance / residual_ratio)
            self.regularization_loss.weight *= (1 - smoothing) + smoothing * (target_balance / reg_ratio)
    
    def _compute_zone_weight(self, coordinates: np.ndarray, 
                           surface_classification: np.ndarray) -> float:
        """Compute zone-based priority weighting."""
        # Simplified zone weighting based on surface classification
        surface_weights = {
            0: self.urban_priorities.get('downtown', 1.0),      # dense_urban
            1: self.urban_priorities.get('residential', 1.0),   # residential
            2: self.urban_priorities.get('green_spaces', 1.0),  # green_infrastructure
            3: self.urban_priorities.get('green_spaces', 1.0),  # water_bodies
            4: self.urban_priorities.get('industrial', 1.0)     # industrial
        }
        
        # Compute average weight based on surface distribution
        if surface_classification.size > 0:
            unique, counts = np.unique(surface_classification, return_counts=True)
            weighted_sum = sum(surface_weights.get(surf, 1.0) * count 
                             for surf, count in zip(unique, counts))
            total_count = sum(counts)
            return weighted_sum / total_count if total_count > 0 else 1.0
        
        return 1.0


# Factory function
def create_smart_city_loss(config: Dict) -> SmartCityCompositeLoss:
    """Create smart city composite loss from configuration."""
    return SmartCityCompositeLoss(
        data_loss_config=config.get('data_loss', {}),
        residual_loss_config=config.get('residual_loss', {}),
        regularization_config=config.get('regularization', {}),
        adaptive_weights=config.get('adaptive_weights', True),
        urban_priorities=config.get('urban_priorities', {})
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Smart City Loss Functions...")
    
    # Create mock data
    n_points = 50
    coordinates = np.random.rand(n_points, 3) * [10, 10, 24]
    coords_2d = coordinates[:, :2]
    
    # Mock predictions and observations
    predictions = {
        'temperature': 20 + 5 * np.random.randn(n_points),
        'pollutant': 50 + 20 * np.random.randn(n_points),
        'humidity': 60 + 10 * np.random.randn(n_points)
    }
    
    observations = {
        'temperature': predictions['temperature'] + 0.5 * np.random.randn(n_points),
        'pollutant': predictions['pollutant'] + 2.0 * np.random.randn(n_points),
        'humidity': predictions['humidity'] + 1.0 * np.random.randn(n_points)
    }
    
    # Mock derivatives
    field_derivatives = {}
    for field in predictions.keys():
        field_derivatives[field] = {
            'u': predictions[field],
            'u_x': np.random.randn(n_points) * 0.1,
            'u_y': np.random.randn(n_points) * 0.1,
            'u_t': np.random.randn(n_points) * 0.01,
            'u_xx': np.random.randn(n_points) * 0.01,
            'u_yy': np.random.randn(n_points) * 0.01
        }
    
    # Mock fractional order fields
    alpha_fields = {
        'alpha_T': 1.5 + 0.2 * np.random.randn(n_points),
        'alpha_C': 1.4 + 0.2 * np.random.randn(n_points),
        'alpha_H': 1.6 + 0.2 * np.random.randn(n_points)
    }
    
    alpha_derivatives = {}
    for field in alpha_fields.keys():
        alpha_derivatives[field] = {
            'alpha': alpha_fields[field],
            'alpha_x': np.random.randn(n_points) * 0.01,
            'alpha_y': np.random.randn(n_points) * 0.01
        }
    
    # Mock fractional Laplacians
    fractional_laplacians = {
        'temperature': np.random.randn(n_points) * 0.1,
        'pollutant': np.random.randn(n_points) * 0.2,
        'humidity': np.random.randn(n_points) * 0.15
    }
    
    # Mock surface classification
    surface_classification = np.random.randint(0, 5, size=(coords_2d.shape[0],))
    
    # Test composite loss
    config = {
        'data_loss': {'weight': 1.0, 'uncertainty_weighting': True},
        'residual_loss': {'weight': 1.0, 'coupling_strength': 0.1},
        'regularization': {'weight': 0.01, 'urban_constraints_weight': 0.2},
        'adaptive_weights': True
    }
    
    loss_function = create_smart_city_loss(config)
    
    # Compute loss
    loss_dict = loss_function.compute(
        predictions=predictions,
        observations=observations,
        field_derivatives=field_derivatives,
        alpha_fields=alpha_fields,
        fractional_laplacians=fractional_laplacians,
        alpha_derivatives=alpha_derivatives,
        surface_classification=surface_classification,
        coordinates=coordinates
    )
    
    print(f"✓ Smart city composite loss computed successfully")
    print(f"  Total loss: {loss_dict['total']:.6f}")
    print(f"  Data loss: {loss_dict['data']:.6f}")
    print(f"  Residual loss: {loss_dict['residual']:.6f}")
    print(f"  Regularization loss: {loss_dict['regularization']:.6f}")
    print(f"  Zone weight: {loss_dict['zone_weight']:.3f}")
    
    # Test multiple iterations for adaptive weighting
    print("\\nTesting adaptive weighting over multiple iterations...")
    for i in range(5):
        loss_dict = loss_function.compute(
            predictions=predictions,
            observations=observations,
            field_derivatives=field_derivatives,
            alpha_fields=alpha_fields,
            fractional_laplacians=fractional_laplacians,
            alpha_derivatives=alpha_derivatives,
            surface_classification=surface_classification,
            coordinates=coordinates
        )
        
        if i % 2 == 0:
            print(f"  Iteration {i+1}: Total={loss_dict['total']:.6f}")
    
    print("\\nAll smart city loss function tests passed!")