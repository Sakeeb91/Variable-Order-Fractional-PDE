"""
Urban Climate Data Generator for Smart City Applications

This module generates synthetic urban climate datasets including temperature,
humidity, air quality, and fractional order fields for different urban environments.
It supports realistic urban scenarios for validating the variable-order fractional
PDE discovery methodology in smart city contexts.

Author: Sakeeb Rahman
Date: 2025
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from typing import Tuple, Dict, Optional, List
import os


class UrbanSurfaceClassifier:
    """
    Classifier for different urban surface types with associated
    fractional order characteristics.
    """
    
    SURFACE_TYPES = {
        'dense_urban': {
            'alpha_T': 1.2,  # Thermal diffusion
            'alpha_C': 1.1,  # Pollutant diffusion  
            'alpha_H': 1.0,  # Humidity diffusion
            'thermal_conductivity': 0.8,
            'emission_factor': 2.5,
            'evapotranspiration': 0.1,
            'description': 'Dense urban core with high building density'
        },
        'residential': {
            'alpha_T': 1.4,
            'alpha_C': 1.3,
            'alpha_H': 1.2,
            'thermal_conductivity': 0.6,
            'emission_factor': 1.2,
            'evapotranspiration': 0.3,
            'description': 'Residential areas with mixed surfaces'
        },
        'green_infrastructure': {
            'alpha_T': 1.7,
            'alpha_C': 1.6,
            'alpha_H': 1.8,
            'thermal_conductivity': 0.4,
            'emission_factor': 0.2,
            'evapotranspiration': 0.8,
            'description': 'Parks, green roofs, urban forests'
        },
        'water_bodies': {
            'alpha_T': 1.9,
            'alpha_C': 1.8,
            'alpha_H': 1.9,
            'thermal_conductivity': 0.3,
            'emission_factor': 0.0,
            'evapotranspiration': 1.0,
            'description': 'Rivers, lakes, fountains'
        },
        'industrial': {
            'alpha_T': 1.1,
            'alpha_C': 1.0,
            'alpha_H': 1.1,
            'thermal_conductivity': 1.0,
            'emission_factor': 4.0,
            'evapotranspiration': 0.05,
            'description': 'Industrial zones with large impervious surfaces'
        }
    }
    
    @classmethod
    def get_surface_properties(cls, surface_type: str) -> Dict:
        """Get properties for a specific surface type."""
        if surface_type not in cls.SURFACE_TYPES:
            raise ValueError(f"Unknown surface type: {surface_type}")
        return cls.SURFACE_TYPES[surface_type].copy()
    
    @classmethod
    def list_surface_types(cls) -> List[str]:
        """Get list of available surface types."""
        return list(cls.SURFACE_TYPES.keys())


class UrbanClimateGenerator:
    """
    Generate synthetic urban climate data with spatially varying
    fractional order fields for smart city applications.
    """
    
    def __init__(self,
                 domain_x: Tuple[float, float] = (0.0, 10.0),  # km
                 domain_y: Tuple[float, float] = (0.0, 10.0),  # km
                 domain_t: Tuple[float, float] = (0.0, 24.0),  # hours
                 nx: int = 51,
                 ny: int = 51,
                 nt: int = 25):
        """
        Initialize urban climate data generator.
        
        Args:
            domain_x: Spatial domain bounds in x-direction (km)
            domain_y: Spatial domain bounds in y-direction (km)
            domain_t: Temporal domain bounds (hours)
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
            nt: Number of time steps
        """
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_t = domain_t
        self.nx = nx
        self.ny = ny
        self.nt = nt
        
        # Create coordinate grids
        self.x = np.linspace(domain_x[0], domain_x[1], nx)
        self.y = np.linspace(domain_y[0], domain_y[1], ny)
        self.t = np.linspace(domain_t[0], domain_t[1], nt)
        
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.dx = (domain_x[1] - domain_x[0]) / (nx - 1)
        self.dy = (domain_y[1] - domain_y[0]) / (ny - 1)
        self.dt = (domain_t[1] - domain_t[0]) / (nt - 1)
        
        # Initialize surface classification
        self.surface_classifier = UrbanSurfaceClassifier()
        
    def create_urban_layout(self, layout_type: str = 'mixed_city') -> np.ndarray:
        """
        Create urban surface layout with different zones.
        
        Args:
            layout_type: Type of urban layout to generate
            
        Returns:
            Surface type array [nx, ny] with integer codes
        """
        surface_map = np.zeros((self.nx, self.ny), dtype=int)
        
        if layout_type == 'mixed_city':
            # Create realistic mixed urban layout
            
            # Dense urban core in center
            center_x, center_y = self.nx // 2, self.ny // 2
            radius_dense = min(self.nx, self.ny) // 6
            
            for i in range(self.nx):
                for j in range(self.ny):
                    dist_center = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    
                    if dist_center < radius_dense:
                        surface_map[i, j] = 0  # dense_urban
                    elif dist_center < 2 * radius_dense:
                        surface_map[i, j] = 1  # residential
                    else:
                        surface_map[i, j] = 2  # green_infrastructure
            
            # Add water bodies (river)
            river_y = self.ny // 4
            surface_map[:, river_y-2:river_y+3] = 3  # water_bodies
            
            # Add industrial zone
            surface_map[self.nx//8:self.nx//4, -self.ny//6:] = 4  # industrial
            
            # Add parks in residential areas
            park_locations = [
                (self.nx//4, self.ny//4),
                (3*self.nx//4, self.ny//4),
                (self.nx//2, 3*self.ny//4)
            ]
            
            for px, py in park_locations:
                park_radius = self.nx // 12
                for i in range(max(0, px-park_radius), min(self.nx, px+park_radius+1)):
                    for j in range(max(0, py-park_radius), min(self.ny, py+park_radius+1)):
                        if (i-px)**2 + (j-py)**2 <= park_radius**2:
                            surface_map[i, j] = 2  # green_infrastructure
                            
        elif layout_type == 'downtown_core':
            # Predominantly dense urban with some green spaces
            surface_map.fill(0)  # Start with dense urban
            
            # Add green corridors
            surface_map[self.nx//3:2*self.nx//3, ::4] = 2
            surface_map[::4, self.ny//3:2*self.ny//3] = 2
            
        elif layout_type == 'suburban':
            # Predominantly residential with green spaces
            surface_map.fill(1)  # Start with residential
            
            # Add green patches
            for i in range(0, self.nx, 8):
                for j in range(0, self.ny, 8):
                    if np.random.random() > 0.6:
                        surface_map[i:i+3, j:j+3] = 2
                        
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")
            
        return surface_map
    
    def generate_fractional_order_fields(self, surface_map: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate fractional order fields based on surface classification.
        
        Args:
            surface_map: Surface type classification array
            
        Returns:
            Dictionary with fractional order fields for T, C, H
        """
        surface_names = list(self.surface_classifier.SURFACE_TYPES.keys())
        
        alpha_T = np.zeros_like(surface_map, dtype=float)
        alpha_C = np.zeros_like(surface_map, dtype=float)
        alpha_H = np.zeros_like(surface_map, dtype=float)
        
        for i in range(self.nx):
            for j in range(self.ny):
                surface_type = surface_names[surface_map[i, j]]
                props = self.surface_classifier.get_surface_properties(surface_type)
                
                # Add small random variations
                noise_level = 0.05
                alpha_T[i, j] = props['alpha_T'] * (1 + noise_level * np.random.randn())
                alpha_C[i, j] = props['alpha_C'] * (1 + noise_level * np.random.randn())
                alpha_H[i, j] = props['alpha_H'] * (1 + noise_level * np.random.randn())
        
        # Smooth the fields to ensure physical continuity
        from scipy.ndimage import gaussian_filter
        sigma = 1.0
        try:
            alpha_T = gaussian_filter(alpha_T, sigma=sigma)
            alpha_C = gaussian_filter(alpha_C, sigma=sigma)
            alpha_H = gaussian_filter(alpha_H, sigma=sigma)
        except ImportError:
            # Fallback: simple averaging filter
            alpha_T = self._simple_smooth(alpha_T)
            alpha_C = self._simple_smooth(alpha_C)
            alpha_H = self._simple_smooth(alpha_H)
        
        # Ensure physical bounds
        alpha_T = np.clip(alpha_T, 1.0, 2.0)
        alpha_C = np.clip(alpha_C, 1.0, 2.0)
        alpha_H = np.clip(alpha_H, 1.0, 2.0)
        
        return {
            'alpha_T': alpha_T,
            'alpha_C': alpha_C,
            'alpha_H': alpha_H
        }
    
    def _simple_smooth(self, field: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Simple smoothing filter fallback."""
        smoothed = field.copy()
        pad = kernel_size // 2
        
        for i in range(pad, field.shape[0] - pad):
            for j in range(pad, field.shape[1] - pad):
                smoothed[i, j] = np.mean(field[i-pad:i+pad+1, j-pad:j+pad+1])
                
        return smoothed
    
    def generate_temperature_field(self, 
                                 alpha_T: np.ndarray,
                                 surface_map: np.ndarray,
                                 time_hour: float = 12.0) -> np.ndarray:
        """
        Generate realistic urban temperature field.
        
        Args:
            alpha_T: Thermal fractional order field
            surface_map: Surface classification
            time_hour: Time of day (0-24 hours)
            
        Returns:
            Temperature field [nx, ny] in Celsius
        """
        # Base temperature with diurnal cycle
        T_base = 20.0 + 8.0 * np.sin(2 * np.pi * (time_hour - 6) / 24)
        
        # Urban heat island effect
        center_x, center_y = self.nx // 2, self.ny // 2
        heat_island = np.zeros_like(self.X)
        
        for i in range(self.nx):
            for j in range(self.ny):
                dist_center = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                # Heat island intensity decreases with distance from center
                heat_island[i, j] = 5.0 * (1 - dist_center / max_dist)
        
        # Surface-specific temperature modifications
        surface_names = list(self.surface_classifier.SURFACE_TYPES.keys())
        T_field = np.full_like(self.X, T_base)
        
        for i in range(self.nx):
            for j in range(self.ny):
                surface_type = surface_names[surface_map[i, j]]
                props = self.surface_classifier.get_surface_properties(surface_type)
                
                # Modify temperature based on surface properties
                if surface_type == 'dense_urban':
                    T_field[i, j] += heat_island[i, j] + 2.0
                elif surface_type == 'industrial':
                    T_field[i, j] += heat_island[i, j] + 3.0
                elif surface_type == 'green_infrastructure':
                    T_field[i, j] += heat_island[i, j] * 0.3 - 2.0  # Cooling effect
                elif surface_type == 'water_bodies':
                    T_field[i, j] += heat_island[i, j] * 0.1 - 3.0  # Strong cooling
                else:  # residential
                    T_field[i, j] += heat_island[i, j] * 0.7
        
        # Add spatial correlation based on fractional order
        T_field += 0.5 * (alpha_T - 1.5) * np.random.randn(self.nx, self.ny)
        
        return T_field
    
    def generate_pollutant_field(self,
                               alpha_C: np.ndarray,
                               surface_map: np.ndarray,
                               time_hour: float = 12.0) -> np.ndarray:
        """
        Generate urban pollutant concentration field.
        
        Args:
            alpha_C: Pollutant fractional order field
            surface_map: Surface classification
            time_hour: Time of day for traffic patterns
            
        Returns:
            Pollutant concentration field [nx, ny] in μg/m³
        """
        # Base background concentration
        C_background = 15.0
        
        # Traffic emission pattern (higher during rush hours)
        traffic_factor = 1.0 + 0.5 * (
            np.exp(-((time_hour - 8)**2) / 4) +  # Morning rush
            np.exp(-((time_hour - 17)**2) / 4)   # Evening rush
        )
        
        surface_names = list(self.surface_classifier.SURFACE_TYPES.keys())
        C_field = np.full_like(self.X, C_background)
        
        for i in range(self.nx):
            for j in range(self.ny):
                surface_type = surface_names[surface_map[i, j]]
                props = self.surface_classifier.get_surface_properties(surface_type)
                
                # Add emissions based on surface type
                emission = props['emission_factor'] * traffic_factor
                C_field[i, j] += emission
                
                # Add dispersion effects
                if surface_type == 'green_infrastructure':
                    C_field[i, j] *= 0.7  # Vegetation filtering
                elif surface_type == 'water_bodies':
                    C_field[i, j] *= 0.8  # Enhanced mixing
        
        # Add wind dispersion effects (simplified)
        wind_direction = 45  # degrees
        wind_speed = 3.0  # m/s
        
        # Simple advection effect
        shift_x = int(wind_speed * np.cos(np.radians(wind_direction)) / self.dx)
        shift_y = int(wind_speed * np.sin(np.radians(wind_direction)) / self.dy)
        
        if abs(shift_x) < self.nx//2 and abs(shift_y) < self.ny//2:
            C_field = np.roll(C_field, shift_x, axis=0)
            C_field = np.roll(C_field, shift_y, axis=1)
        
        # Add fractional-order dependent variability
        C_field += 5.0 * (alpha_C - 1.5) * np.abs(np.random.randn(self.nx, self.ny))
        
        # Ensure non-negative concentrations
        C_field = np.maximum(C_field, 0.0)
        
        return C_field
    
    def generate_complete_urban_scenario(self, 
                                       layout_type: str = 'mixed_city',
                                       scenario_name: str = 'summer_day') -> Dict:
        """
        Generate complete urban climate scenario with all fields.
        
        Args:
            layout_type: Type of urban layout
            scenario_name: Name of the climate scenario
            
        Returns:
            Dictionary with complete urban dataset
        """
        # Generate urban layout
        surface_map = self.create_urban_layout(layout_type)
        
        # Generate fractional order fields
        alpha_fields = self.generate_fractional_order_fields(surface_map)
        
        # Generate time series of climate fields
        temperature_series = []
        pollutant_series = []
        
        for t_hour in self.t:
            T_field = self.generate_temperature_field(
                alpha_fields['alpha_T'], surface_map, t_hour
            )
            C_field = self.generate_pollutant_field(
                alpha_fields['alpha_C'], surface_map, t_hour
            )
            
            temperature_series.append(T_field)
            pollutant_series.append(C_field)
        
        # Convert to numpy arrays
        temperature_series = np.array(temperature_series)  # [nt, nx, ny]
        pollutant_series = np.array(pollutant_series)      # [nt, nx, ny]
        
        # Generate sparse observation points
        n_sensors = min(50, self.nx * self.ny // 20)
        sensor_indices = np.random.choice(self.nx * self.ny, n_sensors, replace=False)
        sensor_x = self.X.flatten()[sensor_indices]
        sensor_y = self.Y.flatten()[sensor_indices]
        
        # Extract observations at sensor locations
        sensor_T = []
        sensor_C = []
        
        for t_idx in range(self.nt):
            T_flat = temperature_series[t_idx].flatten()
            C_flat = pollutant_series[t_idx].flatten()
            
            sensor_T.append(T_flat[sensor_indices])
            sensor_C.append(C_flat[sensor_indices])
        
        sensor_T = np.array(sensor_T)  # [nt, n_sensors]
        sensor_C = np.array(sensor_C)  # [nt, n_sensors]
        
        return {
            'scenario_name': scenario_name,
            'layout_type': layout_type,
            'coordinates': {
                'x': self.x,
                'y': self.y,
                't': self.t,
                'X': self.X,
                'Y': self.Y
            },
            'surface_map': surface_map,
            'alpha_fields': alpha_fields,
            'temperature_series': temperature_series,
            'pollutant_series': pollutant_series,
            'sensor_data': {
                'x_sensors': sensor_x,
                'y_sensors': sensor_y,
                'temperature_obs': sensor_T,
                'pollutant_obs': sensor_C
            },
            'metadata': {
                'nx': self.nx,
                'ny': self.ny,
                'nt': self.nt,
                'dx': self.dx,
                'dy': self.dy,
                'dt': self.dt,
                'domain_x': self.domain_x,
                'domain_y': self.domain_y,
                'domain_t': self.domain_t
            }
        }
    
    def save_dataset(self, dataset: Dict, filename: str, save_dir: str = 'data/smart_city') -> None:
        """Save generated urban dataset."""
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename}.npz")
        
        # Flatten data for saving
        save_dict = {}
        for key, value in dataset.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    save_dict[f"{key}_{subkey}"] = subvalue
            else:
                save_dict[key] = value
        
        np.savez_compressed(filepath, **save_dict)
        print(f"Smart city dataset saved to {filepath}")
    
    def visualize_urban_scenario(self, dataset: Dict, save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of urban scenario."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for visualization")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Urban Climate Scenario: {dataset['scenario_name']}", fontsize=16)
        
        X, Y = dataset['coordinates']['X'], dataset['coordinates']['Y']
        
        # Surface classification
        im1 = axes[0, 0].contourf(X, Y, dataset['surface_map'], levels=5, cmap='Set3')
        axes[0, 0].set_title('Urban Surface Classification')
        axes[0, 0].set_xlabel('x (km)')
        axes[0, 0].set_ylabel('y (km)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Thermal fractional order
        im2 = axes[0, 1].contourf(X, Y, dataset['alpha_fields']['alpha_T'], 
                                levels=20, cmap='viridis')
        axes[0, 1].set_title('Thermal Fractional Order α_T(x,y)')
        axes[0, 1].set_xlabel('x (km)')
        axes[0, 1].set_ylabel('y (km)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Pollutant fractional order
        im3 = axes[0, 2].contourf(X, Y, dataset['alpha_fields']['alpha_C'], 
                                levels=20, cmap='plasma')
        axes[0, 2].set_title('Pollutant Fractional Order α_C(x,y)')
        axes[0, 2].set_xlabel('x (km)')
        axes[0, 2].set_ylabel('y (km)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Temperature field (noon)
        noon_idx = len(dataset['coordinates']['t']) // 2
        im4 = axes[1, 0].contourf(X, Y, dataset['temperature_series'][noon_idx], 
                                levels=20, cmap='hot')
        axes[1, 0].set_title('Temperature Field at Noon (°C)')
        axes[1, 0].set_xlabel('x (km)')
        axes[1, 0].set_ylabel('y (km)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Pollutant field (noon)
        im5 = axes[1, 1].contourf(X, Y, dataset['pollutant_series'][noon_idx], 
                                levels=20, cmap='Reds')
        axes[1, 1].set_title('Pollutant Concentration at Noon (μg/m³)')
        axes[1, 1].set_xlabel('x (km)')
        axes[1, 1].set_ylabel('y (km)')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Sensor locations
        axes[1, 2].contourf(X, Y, dataset['temperature_series'][noon_idx], 
                          levels=20, cmap='hot', alpha=0.6)
        axes[1, 2].scatter(dataset['sensor_data']['x_sensors'], 
                          dataset['sensor_data']['y_sensors'], 
                          c='blue', s=30, marker='o', label='Sensors')
        axes[1, 2].set_title('Sensor Network Layout')
        axes[1, 2].set_xlabel('x (km)')
        axes[1, 2].set_ylabel('y (km)')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def generate_smart_city_datasets():
    """Generate standard smart city datasets for validation."""
    generator = UrbanClimateGenerator(nx=41, ny=41, nt=13)  # Smaller for demo
    
    scenarios = [
        ('mixed_city', 'summer_day'),
        ('downtown_core', 'heat_wave'),
        ('suburban', 'normal_day')
    ]
    
    for layout, scenario in scenarios:
        print(f"Generating {scenario} scenario with {layout} layout...")
        dataset = generator.generate_complete_urban_scenario(layout, scenario)
        generator.save_dataset(dataset, f"smart_city_{scenario}_{layout}")
        generator.visualize_urban_scenario(dataset, 
                                        f"visuals/smart_city_{scenario}_{layout}.png")
    
    print("All smart city datasets generated successfully!")


if __name__ == "__main__":
    generate_smart_city_datasets()