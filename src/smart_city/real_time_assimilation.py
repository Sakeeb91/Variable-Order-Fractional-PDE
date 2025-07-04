"""
Real-Time Data Assimilation for Smart City Applications

This module implements advanced data assimilation techniques for integrating
real-time IoT sensor data with variable-order fractional PDE models,
including Kalman filtering, ensemble methods, and adaptive model updating.

Author: Sakeeb Rahman
Date: 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

from iot_integration import SensorReading, SensorNetwork


@dataclass
class AssimilationConfig:
    """Configuration for data assimilation system."""
    
    # Spatial interpolation
    interpolation_method: str = 'linear'  # 'linear', 'cubic', 'rbf'
    max_interpolation_distance: float = 2.0  # km
    
    # Temporal settings
    assimilation_window: float = 300.0  # seconds
    prediction_horizon: float = 3600.0  # seconds
    
    # Uncertainty parameters
    observation_noise_std: Dict[str, float] = None
    model_noise_std: Dict[str, float] = None
    
    # Quality control
    outlier_threshold: float = 3.0  # standard deviations
    min_sensors_for_field: int = 3
    
    # Performance
    max_memory_hours: float = 24.0
    update_frequency: float = 60.0  # seconds
    
    def __post_init__(self):
        if self.observation_noise_std is None:
            self.observation_noise_std = {
                'temperature': 0.5,  # °C
                'pollutant': 5.0,    # μg/m³ 
                'humidity': 3.0      # %
            }
        
        if self.model_noise_std is None:
            self.model_noise_std = {
                'temperature': 1.0,
                'pollutant': 10.0,
                'humidity': 5.0
            }


class SpatialInterpolator:
    """Spatial interpolation for sensor measurements."""
    
    def __init__(self, config: AssimilationConfig):
        self.config = config
        
    def interpolate_field(self,
                         sensor_locations: np.ndarray,
                         sensor_values: np.ndarray,
                         target_locations: np.ndarray,
                         field_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate sensor measurements to target locations.
        
        Args:
            sensor_locations: Sensor positions [N, 2]
            sensor_values: Sensor measurements [N]
            target_locations: Target positions [M, 2]
            field_name: Name of the field being interpolated
            
        Returns:
            Tuple of (interpolated_values [M], uncertainties [M])
        """
        if len(sensor_locations) == 0 or len(sensor_values) == 0:
            return np.full(len(target_locations), np.nan), np.full(len(target_locations), np.inf)
        
        # Remove invalid measurements
        valid_mask = ~np.isnan(sensor_values)
        if not np.any(valid_mask):
            return np.full(len(target_locations), np.nan), np.full(len(target_locations), np.inf)
        
        valid_locations = sensor_locations[valid_mask]
        valid_values = sensor_values[valid_mask]
        
        # Handle single sensor case
        if len(valid_values) == 1:
            distances = np.linalg.norm(target_locations - valid_locations[0], axis=1)
            within_range = distances <= self.config.max_interpolation_distance
            
            interpolated = np.full(len(target_locations), np.nan)
            interpolated[within_range] = valid_values[0]
            
            # Uncertainty increases with distance
            uncertainty = np.full(len(target_locations), np.inf)
            uncertainty[within_range] = self.config.observation_noise_std.get(field_name, 1.0) * (1 + distances[within_range])
            
            return interpolated, uncertainty
        
        # Compute distances between sensors and targets
        distances = cdist(target_locations, valid_locations)
        min_distances = np.min(distances, axis=1)
        
        # Only interpolate within reasonable distance
        valid_targets = min_distances <= self.config.max_interpolation_distance
        
        interpolated = np.full(len(target_locations), np.nan)
        uncertainty = np.full(len(target_locations), np.inf)
        
        if np.any(valid_targets):
            valid_target_locations = target_locations[valid_targets]
            
            try:
                if self.config.interpolation_method == 'linear':
                    values = griddata(valid_locations, valid_values, valid_target_locations, 
                                    method='linear', fill_value=np.nan)
                elif self.config.interpolation_method == 'cubic':
                    values = griddata(valid_locations, valid_values, valid_target_locations,
                                    method='cubic', fill_value=np.nan)
                else:  # Default to nearest
                    values = griddata(valid_locations, valid_values, valid_target_locations,
                                    method='nearest', fill_value=np.nan)
                
                interpolated[valid_targets] = values
                
                # Estimate uncertainty based on sensor density and distance
                base_uncertainty = self.config.observation_noise_std.get(field_name, 1.0)
                
                for i, is_valid in enumerate(valid_targets):
                    if is_valid:
                        sensor_distances = distances[i]
                        closest_sensors = np.sort(sensor_distances)[:min(3, len(closest_sensors))]
                        
                        # Uncertainty increases with distance to nearest sensors
                        distance_factor = 1 + np.mean(closest_sensors)
                        
                        # Uncertainty decreases with more nearby sensors
                        density_factor = 1 / (1 + len(closest_sensors))
                        
                        uncertainty[i] = base_uncertainty * distance_factor * (1 + density_factor)
                        
            except Exception as e:
                print(f"Interpolation failed for {field_name}: {e}")
                # Fall back to nearest neighbor
                for i, is_valid in enumerate(valid_targets):
                    if is_valid:
                        nearest_idx = np.argmin(distances[i])
                        interpolated[i] = valid_values[nearest_idx]
                        uncertainty[i] = self.config.observation_noise_std.get(field_name, 1.0) * (1 + distances[i, nearest_idx])
        
        return interpolated, uncertainty


class OnlineKalmanFilter:
    """Online Kalman filter for field estimation."""
    
    def __init__(self, field_name: str, config: AssimilationConfig):
        self.field_name = field_name
        self.config = config
        
        # Filter state
        self.state_mean = None
        self.state_covariance = None
        self.initialized = False
        
        # Noise parameters
        self.observation_noise = config.observation_noise_std.get(field_name, 1.0) ** 2
        self.process_noise = config.model_noise_std.get(field_name, 1.0) ** 2
        
        # Filter history
        self.state_history = []
        self.uncertainty_history = []
        self.timestamps = []
        
    def initialize(self, initial_state: np.ndarray, initial_uncertainty: np.ndarray):
        """Initialize filter with initial state estimate."""
        self.state_mean = initial_state.copy()
        self.state_covariance = np.diag(initial_uncertainty ** 2)
        self.initialized = True
        
        self.state_history.append(self.state_mean.copy())
        self.uncertainty_history.append(np.diag(self.state_covariance).copy())
        self.timestamps.append(datetime.now())
        
    def predict(self, dt: float):
        """Prediction step of Kalman filter."""
        if not self.initialized:
            return
        
        # Simple random walk model (could be replaced with PDE dynamics)
        # State stays the same (identity transition)
        # Covariance increases with time
        process_noise_matrix = np.eye(len(self.state_mean)) * self.process_noise * dt
        self.state_covariance += process_noise_matrix
        
    def update(self, observations: np.ndarray, observation_locations: np.ndarray,
              observation_uncertainties: np.ndarray):
        """Update step with new observations."""
        if not self.initialized:
            return
        
        # Remove invalid observations
        valid_mask = ~np.isnan(observations)
        if not np.any(valid_mask):
            return
        
        valid_obs = observations[valid_mask]
        valid_locations = observation_locations[valid_mask]
        valid_uncertainties = observation_uncertainties[valid_mask]
        
        # Observation matrix (maps state to observations)
        # This is simplified - in full implementation would use proper spatial mapping
        n_state = len(self.state_mean)
        n_obs = len(valid_obs)
        
        if n_obs > n_state:
            # More observations than state variables - use subset
            selected_indices = np.linspace(0, n_obs-1, n_state, dtype=int)
            valid_obs = valid_obs[selected_indices]
            valid_uncertainties = valid_uncertainties[selected_indices]
            n_obs = len(valid_obs)
        
        H = np.eye(min(n_obs, n_state), n_state)
        
        # Observation noise covariance
        R = np.diag(valid_uncertainties ** 2)
        
        # Kalman gain
        S = H @ self.state_covariance @ H.T + R
        try:
            K = self.state_covariance @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudo-inverse
            K = self.state_covariance @ H.T @ np.linalg.pinv(S)
        
        # Update state and covariance
        state_subset = self.state_mean[:n_obs]
        innovation = valid_obs - state_subset
        
        self.state_mean[:n_obs] += (K @ innovation)[:n_obs]
        self.state_covariance = (np.eye(n_state) - K @ H) @ self.state_covariance
        
        # Store history
        self.state_history.append(self.state_mean.copy())
        self.uncertainty_history.append(np.diag(self.state_covariance).copy())
        self.timestamps.append(datetime.now())
        
        # Limit history size
        max_history = int(self.config.max_memory_hours * 3600 / self.config.update_frequency)
        if len(self.state_history) > max_history:
            self.state_history = self.state_history[-max_history:]
            self.uncertainty_history = self.uncertainty_history[-max_history:]
            self.timestamps = self.timestamps[-max_history:]
    
    def get_current_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and uncertainty."""
        if not self.initialized:
            return None, None
        
        return self.state_mean.copy(), np.sqrt(np.diag(self.state_covariance))


class RealTimeDataAssimilator:
    """Real-time data assimilation system."""
    
    def __init__(self,
                 spatial_grid: Tuple[np.ndarray, np.ndarray],
                 config: AssimilationConfig = None):
        """
        Initialize data assimilation system.
        
        Args:
            spatial_grid: Tuple of (x_grid, y_grid) defining spatial domain
            config: Assimilation configuration
        """
        self.x_grid, self.y_grid = spatial_grid
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.grid_points = np.column_stack([self.X.flatten(), self.Y.flatten()])
        
        self.config = config or AssimilationConfig()
        
        # Interpolator
        self.interpolator = SpatialInterpolator(self.config)
        
        # Kalman filters for each field
        self.kalman_filters = {}
        
        # Data storage
        self.recent_readings = []
        self.assimilated_fields = {}
        self.field_uncertainties = {}
        
        # Performance tracking
        self.last_update_time = time.time()
        self.update_count = 0
        self.processing_times = []
        
    def add_sensor_reading(self, reading: SensorReading):
        """Add new sensor reading for assimilation."""
        self.recent_readings.append(reading)
        
        # Remove old readings outside assimilation window
        cutoff_time = reading.timestamp - timedelta(seconds=self.config.assimilation_window)
        self.recent_readings = [r for r in self.recent_readings if r.timestamp > cutoff_time]
        
        # Check if update is needed
        current_time = time.time()
        if current_time - self.last_update_time > self.config.update_frequency:
            self._perform_assimilation()
            self.last_update_time = current_time
        
        # Also trigger assimilation when we have enough new data
        if len(self.recent_readings) >= self.config.min_sensors_for_field:
            self._perform_assimilation()
    
    def _perform_assimilation(self):
        """Perform data assimilation with recent readings."""
        if len(self.recent_readings) == 0:
            return
        
        start_time = time.time()
        
        # Group readings by field
        field_data = {}
        for reading in self.recent_readings:
            for field_name, value in reading.measurements.items():
                if field_name not in field_data:
                    field_data[field_name] = {
                        'values': [],
                        'locations': [],
                        'uncertainties': [],
                        'timestamps': []
                    }
                
                field_data[field_name]['values'].append(value)
                field_data[field_name]['locations'].append(reading.location)
                uncertainty = reading.uncertainty.get(field_name, 
                                                    self.config.observation_noise_std.get(field_name, 1.0))
                field_data[field_name]['uncertainties'].append(uncertainty)
                field_data[field_name]['timestamps'].append(reading.timestamp)
        
        # Assimilate each field
        for field_name, data in field_data.items():
            if len(data['values']) < self.config.min_sensors_for_field:
                continue
                
            self._assimilate_field(field_name, data)
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.update_count += 1
        
        # Limit processing time history
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
    
    def _assimilate_field(self, field_name: str, data: Dict):
        """Assimilate data for a specific field."""
        values = np.array(data['values'])
        locations = np.array(data['locations'])
        uncertainties = np.array(data['uncertainties'])
        
        # Quality control - remove outliers
        if len(values) > 3:
            median_val = np.median(values)
            mad = np.median(np.abs(values - median_val))
            outlier_threshold = self.config.outlier_threshold
            
            outlier_mask = np.abs(values - median_val) > outlier_threshold * mad
            if np.any(outlier_mask):
                print(f"Removing {np.sum(outlier_mask)} outliers from {field_name}")
                valid_mask = ~outlier_mask
                values = values[valid_mask]
                locations = locations[valid_mask]
                uncertainties = uncertainties[valid_mask]
        
        if len(values) == 0:
            return
        
        # Spatial interpolation to grid
        interpolated_values, interpolated_uncertainties = self.interpolator.interpolate_field(
            locations, values, self.grid_points, field_name
        )
        
        # Initialize Kalman filter if needed
        if field_name not in self.kalman_filters:
            # Only use valid interpolated points for initialization
            valid_mask = ~np.isnan(interpolated_values)
            if not np.any(valid_mask):
                return
            
            initial_state = np.full(len(self.grid_points), np.nanmean(interpolated_values))
            initial_uncertainty = np.full(len(self.grid_points), np.nanstd(interpolated_values))
            
            self.kalman_filters[field_name] = OnlineKalmanFilter(field_name, self.config)
            self.kalman_filters[field_name].initialize(initial_state, initial_uncertainty)
        
        # Kalman filter prediction step
        dt = self.config.update_frequency
        self.kalman_filters[field_name].predict(dt)
        
        # Kalman filter update step
        valid_mask = ~np.isnan(interpolated_values)
        if np.any(valid_mask):
            valid_obs = interpolated_values[valid_mask]
            valid_locations = self.grid_points[valid_mask]
            valid_uncertainties = interpolated_uncertainties[valid_mask]
            
            self.kalman_filters[field_name].update(valid_obs, valid_locations, valid_uncertainties)
        
        # Store assimilated field
        state, uncertainty = self.kalman_filters[field_name].get_current_estimate()
        if state is not None:
            self.assimilated_fields[field_name] = state.reshape(self.X.shape)
            self.field_uncertainties[field_name] = uncertainty.reshape(self.X.shape)
    
    def get_assimilated_fields(self) -> Dict[str, np.ndarray]:
        """Get current assimilated field estimates."""
        return self.assimilated_fields.copy()
    
    def get_field_uncertainties(self) -> Dict[str, np.ndarray]:
        """Get current field uncertainty estimates."""
        return self.field_uncertainties.copy()
    
    def get_field_at_location(self, field_name: str, location: Tuple[float, float]) -> Tuple[float, float]:
        """Get field value and uncertainty at specific location."""
        if field_name not in self.assimilated_fields:
            return np.nan, np.inf
        
        x, y = location
        
        # Find nearest grid point (could use interpolation for better accuracy)
        distances = np.sqrt((self.X - x)**2 + (self.Y - y)**2)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        value = self.assimilated_fields[field_name][min_idx]
        uncertainty = self.field_uncertainties[field_name][min_idx]
        
        return value, uncertainty
    
    def get_performance_metrics(self) -> Dict:
        """Get assimilation performance metrics."""
        if len(self.processing_times) == 0:
            return {'no_data': True}
        
        return {
            'total_updates': self.update_count,
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'recent_readings_count': len(self.recent_readings),
            'active_fields': list(self.assimilated_fields.keys()),
            'grid_size': self.X.size
        }


def create_assimilation_demo():
    """Create demonstration of real-time data assimilation."""
    print("Real-Time Data Assimilation Demo")
    print("=" * 40)
    
    # Create spatial grid
    x_grid = np.linspace(0, 10, 21)
    y_grid = np.linspace(0, 10, 21)
    
    # Create assimilation system
    config = AssimilationConfig(
        interpolation_method='linear',
        max_interpolation_distance=3.0,
        assimilation_window=300.0,
        update_frequency=30.0
    )
    
    assimilator = RealTimeDataAssimilator((x_grid, y_grid), config)
    
    # Simulate sensor readings
    sensor_locations = [(2, 2), (5, 5), (8, 8), (2, 8), (8, 2)]
    
    print(f"Created assimilation system with {len(x_grid)}x{len(y_grid)} grid")
    print(f"Simulating readings from {len(sensor_locations)} sensors")
    
    # Generate time series of readings
    current_time = datetime.now()
    
    for t in range(10):  # 10 time steps
        print(f"\nTime step {t+1}/10:")
        
        # Generate readings for this time step
        for i, location in enumerate(sensor_locations):
            # Simulate realistic measurements with spatial and temporal variation
            base_temp = 20 + 5 * np.sin(2 * np.pi * t / 24)  # Diurnal cycle
            temp = base_temp + np.random.randn() * 0.5
            
            pollutant = 50 + 20 * np.random.randn()
            pollutant = max(0, pollutant)  # Non-negative
            
            humidity = 60 + 10 * np.random.randn()
            humidity = max(0, min(100, humidity))  # 0-100%
            
            reading = SensorReading(
                sensor_id=f"sensor_{i+1}",
                timestamp=current_time + timedelta(seconds=t*30),
                location=location,
                measurements={
                    'temperature': temp,
                    'pollutant': pollutant,
                    'humidity': humidity
                },
                uncertainty={
                    'temperature': 0.5,
                    'pollutant': 5.0,
                    'humidity': 3.0
                }
            )
            
            assimilator.add_sensor_reading(reading)
        
        # Get current assimilated fields
        fields = assimilator.get_assimilated_fields()
        uncertainties = assimilator.get_field_uncertainties()
        
        for field_name, field_data in fields.items():
            mean_value = np.nanmean(field_data)
            std_value = np.nanstd(field_data)
            mean_uncertainty = np.nanmean(uncertainties[field_name])
            
            print(f"  {field_name}: {mean_value:.2f} ± {std_value:.2f} "
                  f"(uncertainty: {mean_uncertainty:.2f})")
        
        # Test point query
        test_location = (5.0, 5.0)  # City center
        for field_name in fields.keys():
            value, uncertainty = assimilator.get_field_at_location(field_name, test_location)
            print(f"    At center: {field_name} = {value:.2f} ± {uncertainty:.2f}")
    
    # Performance metrics
    metrics = assimilator.get_performance_metrics()
    print(f"\nPerformance Summary:")
    if 'no_data' not in metrics:
        print(f"  Total updates: {metrics['total_updates']}")
        print(f"  Avg processing time: {metrics['avg_processing_time']*1000:.1f} ms")
        print(f"  Active fields: {metrics['active_fields']}")
    else:
        print(f"  No processing metrics available yet")
    
    print("\nData assimilation demo completed successfully!")
    
    return assimilator


if __name__ == "__main__":
    # Run the demonstration
    demo_assimilator = create_assimilation_demo()