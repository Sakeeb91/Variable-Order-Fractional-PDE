"""
Comprehensive Smart City Visualization System

This module provides advanced visualization capabilities for smart city
variable-order fractional PDE discovery, including real-time dashboards,
3D spatial plots, temporal analysis, and interactive city monitoring.

Author: Sakeeb Rahman
Date: 2025
"""

import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: Matplotlib not available. Visualization features limited.")

try:
    from matplotlib.patches import Circle, Rectangle, Polygon
    HAS_PATCHES = True
except ImportError:
    HAS_PATCHES = False


@dataclass
class VisualizationConfig:
    """Configuration for visualization system."""
    
    # Figure settings
    figure_size: Tuple[int, int] = (16, 12)
    dpi: int = 100
    style: str = 'default'  # matplotlib style
    
    # Color schemes
    temperature_cmap: str = 'hot'
    pollutant_cmap: str = 'Reds'
    humidity_cmap: str = 'Blues'
    alpha_cmap: str = 'viridis'
    surface_cmap: str = 'Set3'
    
    # Layout
    subplot_spacing: float = 0.3
    colorbar_shrink: float = 0.8
    
    # Animation
    animation_interval: int = 200  # milliseconds
    animation_frames: int = 50
    
    # Export
    export_format: str = 'png'
    export_dpi: int = 300
    
    # Interactive features
    enable_interactivity: bool = True
    show_sensor_locations: bool = True
    show_grid_lines: bool = False


class SmartCityVisualizer:
    """Comprehensive visualization system for smart city data."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        if HAS_MATPLOTLIB:
            # Set matplotlib style
            if self.config.style != 'default':
                plt.style.use(self.config.style)
            
            # Configure matplotlib for better output
            plt.rcParams['figure.dpi'] = self.config.dpi
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['legend.fontsize'] = 9
        
        self.figures = {}  # Store created figures
        
    def create_urban_overview_dashboard(self,
                                      urban_dataset: Dict,
                                      current_time_idx: int = None,
                                      save_path: Optional[str] = None) -> None:
        """Create comprehensive urban overview dashboard."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for visualization")
            return
        
        fig = plt.figure(figsize=self.config.figure_size)
        fig.suptitle(f"Smart City Overview: {urban_dataset.get('scenario_name', 'Unknown')}", 
                    fontsize=16, fontweight='bold')
        
        # Create subplot grid
        gs = fig.add_gridspec(3, 4, hspace=self.config.subplot_spacing, 
                             wspace=self.config.subplot_spacing)
        
        X = urban_dataset['coordinates']['X']
        Y = urban_dataset['coordinates']['Y']
        
        # Time index for current snapshot
        if current_time_idx is None:
            current_time_idx = len(urban_dataset['coordinates']['t']) // 2
        
        # 1. Urban Surface Classification
        ax1 = fig.add_subplot(gs[0, 0])
        surface_im = ax1.contourf(X, Y, urban_dataset['surface_map'], 
                                 levels=5, cmap=self.config.surface_cmap)
        ax1.set_title('Urban Surface Types')
        ax1.set_xlabel('x (km)')
        ax1.set_ylabel('y (km)')
        
        # Add surface type legend
        surface_types = ['Dense Urban', 'Residential', 'Green Infra', 'Water', 'Industrial']
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=surface_im.cmap(i/4)) 
                          for i in range(5)]
        ax1.legend(legend_elements, surface_types, loc='upper right', fontsize=8)
        
        # 2. Temperature Field
        ax2 = fig.add_subplot(gs[0, 1])
        temp_data = urban_dataset['temperature_series'][current_time_idx]
        temp_im = ax2.contourf(X, Y, temp_data, levels=20, cmap=self.config.temperature_cmap)
        ax2.set_title('Temperature (°C)')
        ax2.set_xlabel('x (km)')
        ax2.set_ylabel('y (km)')
        plt.colorbar(temp_im, ax=ax2, shrink=self.config.colorbar_shrink)
        
        # 3. Pollutant Concentration
        ax3 = fig.add_subplot(gs[0, 2])
        pollutant_data = urban_dataset['pollutant_series'][current_time_idx]
        pollutant_im = ax3.contourf(X, Y, pollutant_data, levels=20, cmap=self.config.pollutant_cmap)
        ax3.set_title('Pollutant (μg/m³)')
        ax3.set_xlabel('x (km)')
        ax3.set_ylabel('y (km)')
        plt.colorbar(pollutant_im, ax=ax3, shrink=self.config.colorbar_shrink)
        
        # 4. Thermal Fractional Order
        ax4 = fig.add_subplot(gs[0, 3])
        alpha_T_im = ax4.contourf(X, Y, urban_dataset['alpha_fields']['alpha_T'], 
                                 levels=20, cmap=self.config.alpha_cmap)
        ax4.set_title('Thermal α_T(x,y)')
        ax4.set_xlabel('x (km)')
        ax4.set_ylabel('y (km)')
        plt.colorbar(alpha_T_im, ax=ax4, shrink=self.config.colorbar_shrink)
        
        # 5. Sensor Network Layout
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.contourf(X, Y, temp_data, levels=20, cmap=self.config.temperature_cmap, alpha=0.6)
        
        if 'sensor_data' in urban_dataset and self.config.show_sensor_locations:
            sensor_x = urban_dataset['sensor_data']['x_sensors']
            sensor_y = urban_dataset['sensor_data']['y_sensors']
            ax5.scatter(sensor_x, sensor_y, c='blue', s=50, marker='o', 
                       edgecolors='white', linewidth=1, label='Sensors', zorder=5)
            ax5.legend()
        
        ax5.set_title('Sensor Network')
        ax5.set_xlabel('x (km)')
        ax5.set_ylabel('y (km)')
        
        # 6. Temporal Evolution (Temperature)
        ax6 = fig.add_subplot(gs[1, 1:3])
        t_hours = urban_dataset['coordinates']['t']
        
        # Sample locations for temporal plots
        sample_locations = [(2, 2), (5, 5), (8, 8)]  # Different urban zones
        location_names = ['Downtown', 'City Center', 'Suburban']
        
        for i, (x_loc, y_loc) in enumerate(sample_locations):
            # Find nearest grid point
            x_idx = np.argmin(np.abs(X[0, :] - x_loc))
            y_idx = np.argmin(np.abs(Y[:, 0] - y_loc))
            
            temp_series = [urban_dataset['temperature_series'][t][y_idx, x_idx] 
                          for t in range(len(t_hours))]
            
            ax6.plot(t_hours, temp_series, marker='o', label=location_names[i], linewidth=2)
        
        ax6.axvline(x=t_hours[current_time_idx], color='red', linestyle='--', 
                   label='Current Time', alpha=0.7)
        ax6.set_title('Temperature Evolution at Key Locations')
        ax6.set_xlabel('Time (hours)')
        ax6.set_ylabel('Temperature (°C)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Fractional Order Distribution
        ax7 = fig.add_subplot(gs[1, 3])
        alpha_fields = urban_dataset['alpha_fields']
        
        # Create histogram of alpha values
        alpha_T_flat = alpha_fields['alpha_T'].flatten()
        alpha_C_flat = alpha_fields['alpha_C'].flatten()
        alpha_H_flat = alpha_fields['alpha_H'].flatten()
        
        ax7.hist(alpha_T_flat, bins=20, alpha=0.7, label='α_T (thermal)', density=True)
        ax7.hist(alpha_C_flat, bins=20, alpha=0.7, label='α_C (pollutant)', density=True)
        ax7.hist(alpha_H_flat, bins=20, alpha=0.7, label='α_H (humidity)', density=True)
        
        ax7.set_title('Fractional Order Distribution')
        ax7.set_xlabel('α value')
        ax7.set_ylabel('Density')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Urban Heat Island Analysis
        ax8 = fig.add_subplot(gs[2, 0])
        
        # Compute temperature anomaly from spatial mean
        temp_mean = np.mean(temp_data)
        temp_anomaly = temp_data - temp_mean
        
        anomaly_im = ax8.contourf(X, Y, temp_anomaly, levels=20, 
                                 cmap='RdBu_r', vmin=-5, vmax=5)
        ax8.set_title('Heat Island Effect (°C)')
        ax8.set_xlabel('x (km)')
        ax8.set_ylabel('y (km)')
        plt.colorbar(anomaly_im, ax=ax8, shrink=self.config.colorbar_shrink)
        
        # 9. Pollutant vs Temperature Correlation
        ax9 = fig.add_subplot(gs[2, 1])
        
        temp_flat = temp_data.flatten()
        pollutant_flat = pollutant_data.flatten()
        
        ax9.scatter(temp_flat, pollutant_flat, alpha=0.6, s=20)
        
        # Add correlation line
        correlation = np.corrcoef(temp_flat, pollutant_flat)[0, 1]
        z = np.polyfit(temp_flat, pollutant_flat, 1)
        p = np.poly1d(z)
        ax9.plot(temp_flat, p(temp_flat), "r--", alpha=0.8)
        
        ax9.set_title(f'Temperature-Pollutant Correlation\nr = {correlation:.3f}')
        ax9.set_xlabel('Temperature (°C)')
        ax9.set_ylabel('Pollutant (μg/m³)')
        ax9.grid(True, alpha=0.3)
        
        # 10. Multi-field Alpha Comparison
        ax10 = fig.add_subplot(gs[2, 2])
        
        # Compare alpha fields spatially
        alpha_diff_TC = alpha_fields['alpha_T'] - alpha_fields['alpha_C']
        alpha_diff_im = ax10.contourf(X, Y, alpha_diff_TC, levels=20, 
                                     cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax10.set_title('α_T - α_C Difference')
        ax10.set_xlabel('x (km)')
        ax10.set_ylabel('y (km)')
        plt.colorbar(alpha_diff_im, ax=ax10, shrink=self.config.colorbar_shrink)
        
        # 11. Statistics Summary
        ax11 = fig.add_subplot(gs[2, 3])
        ax11.axis('off')
        
        # Compute summary statistics
        stats_text = f"""Summary Statistics (t = {t_hours[current_time_idx]:.1f}h)
        
Temperature:
  Mean: {np.mean(temp_data):.1f}°C
  Std:  {np.std(temp_data):.1f}°C
  Range: [{np.min(temp_data):.1f}, {np.max(temp_data):.1f}]
  
Pollutant:
  Mean: {np.mean(pollutant_data):.1f} μg/m³
  Std:  {np.std(pollutant_data):.1f} μg/m³
  Range: [{np.min(pollutant_data):.1f}, {np.max(pollutant_data):.1f}]
  
Fractional Orders:
  α_T: {np.mean(alpha_T_flat):.2f} ± {np.std(alpha_T_flat):.2f}
  α_C: {np.mean(alpha_C_flat):.2f} ± {np.std(alpha_C_flat):.2f}
  α_H: {np.mean(alpha_H_flat):.2f} ± {np.std(alpha_H_flat):.2f}
  
Urban Analysis:
  Heat Island Max: {np.max(temp_anomaly):.1f}°C
  Temp-Pollutant r: {correlation:.3f}
  Active Sensors: {len(urban_dataset.get('sensor_data', {}).get('x_sensors', []))}
        """
        
        ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.export_dpi, bbox_inches='tight')
            print(f"Urban overview dashboard saved to {save_path}")
        
        plt.show()
        self.figures['urban_overview'] = fig
        
        return fig
    
    def create_temporal_animation(self,
                                urban_dataset: Dict,
                                field_name: str = 'temperature',
                                save_path: Optional[str] = None) -> None:
        """Create animated visualization of temporal evolution."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for animation")
            return
        
        print(f"Creating temporal animation for {field_name}...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        X = urban_dataset['coordinates']['X']
        Y = urban_dataset['coordinates']['Y']
        t_hours = urban_dataset['coordinates']['t']
        
        if field_name == 'temperature':
            data_series = urban_dataset['temperature_series']
            cmap = self.config.temperature_cmap
            title = 'Temperature (°C)'
        elif field_name == 'pollutant':
            data_series = urban_dataset['pollutant_series']
            cmap = self.config.pollutant_cmap
            title = 'Pollutant Concentration (μg/m³)'
        else:
            print(f"Field {field_name} not supported for animation")
            return
        
        # Set up initial plots
        vmin, vmax = np.min(data_series), np.max(data_series)
        
        # Spatial plot
        im = ax1.contourf(X, Y, data_series[0], levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title(f'{title} - t = {t_hours[0]:.1f}h')
        ax1.set_xlabel('x (km)')
        ax1.set_ylabel('y (km)')
        cbar = plt.colorbar(im, ax=ax1)
        
        # Add sensor locations if available
        if 'sensor_data' in urban_dataset and self.config.show_sensor_locations:
            sensor_x = urban_dataset['sensor_data']['x_sensors']
            sensor_y = urban_dataset['sensor_data']['y_sensors']
            sensor_points = ax1.scatter(sensor_x, sensor_y, c='white', s=30, 
                                      marker='o', edgecolors='black', linewidth=1, zorder=5)
        
        # Temporal evolution at center point
        center_idx_x = X.shape[1] // 2
        center_idx_y = X.shape[0] // 2
        center_values = [data_series[t][center_idx_y, center_idx_x] for t in range(len(t_hours))]
        
        line, = ax2.plot(t_hours, center_values, 'b-', linewidth=2, label='City Center')
        point, = ax2.plot(t_hours[0], center_values[0], 'ro', markersize=8)
        
        ax2.set_title('Temporal Evolution at City Center')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel(title)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Animation function
        def animate(frame):
            ax1.clear()
            
            # Update spatial plot
            im = ax1.contourf(X, Y, data_series[frame], levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
            ax1.set_title(f'{title} - t = {t_hours[frame]:.1f}h')
            ax1.set_xlabel('x (km)')
            ax1.set_ylabel('y (km)')
            
            # Re-add sensor locations
            if 'sensor_data' in urban_dataset and self.config.show_sensor_locations:
                ax1.scatter(sensor_x, sensor_y, c='white', s=30, 
                           marker='o', edgecolors='black', linewidth=1, zorder=5)
            
            # Update temporal plot
            point.set_data([t_hours[frame]], [center_values[frame]])
            
            return im, point
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(t_hours),
                                     interval=self.config.animation_interval,
                                     blit=False, repeat=True)
        
        if save_path:
            # Save as GIF
            gif_path = save_path.replace('.png', '.gif').replace('.jpg', '.gif')
            anim.save(gif_path, writer='pillow', fps=5)
            print(f"Animation saved to {gif_path}")
        
        plt.tight_layout()
        plt.show()
        
        self.figures[f'{field_name}_animation'] = fig
        return anim
    
    def create_3d_urban_visualization(self,
                                    urban_dataset: Dict,
                                    field_name: str = 'temperature',
                                    time_idx: int = None,
                                    save_path: Optional[str] = None) -> None:
        """Create 3D visualization of urban fields."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for 3D visualization")
            return
        
        fig = plt.figure(figsize=self.config.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        X = urban_dataset['coordinates']['X']
        Y = urban_dataset['coordinates']['Y']
        
        if time_idx is None:
            time_idx = len(urban_dataset['coordinates']['t']) // 2
        
        if field_name == 'temperature':
            Z = urban_dataset['temperature_series'][time_idx]
            title = f"3D Temperature Distribution (t = {urban_dataset['coordinates']['t'][time_idx]:.1f}h)"
            cmap = self.config.temperature_cmap
        elif field_name == 'pollutant':
            Z = urban_dataset['pollutant_series'][time_idx]
            title = f"3D Pollutant Distribution (t = {urban_dataset['coordinates']['t'][time_idx]:.1f}h)"
            cmap = self.config.pollutant_cmap
        elif field_name.startswith('alpha'):
            Z = urban_dataset['alpha_fields'][field_name]
            title = f"3D {field_name} Distribution"
            cmap = self.config.alpha_cmap
        else:
            print(f"Field {field_name} not supported for 3D visualization")
            return
        
        # Create 3D surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add contour lines at base
        ax.contour(X, Y, Z, levels=10, linewidths=0.5, colors='black', 
                  offset=np.min(Z), alpha=0.5)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel(field_name)
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=20)
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.export_dpi, bbox_inches='tight')
            print(f"3D visualization saved to {save_path}")
        
        plt.show()
        self.figures[f'3d_{field_name}'] = fig
        
        return fig
    
    def create_real_time_dashboard(self,
                                 current_estimates: Dict,
                                 sensor_readings: List = None,
                                 performance_metrics: Dict = None,
                                 save_path: Optional[str] = None) -> None:
        """Create real-time monitoring dashboard."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for real-time dashboard")
            return
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Smart City Real-Time Dashboard', fontsize=16, fontweight='bold')
        
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Current field estimates
        if 'fields' in current_estimates and current_estimates['fields']:
            fields = current_estimates['fields']
            
            # Temperature
            ax1 = fig.add_subplot(gs[0, 0])
            if 'temperature' in fields:
                temp_data = np.array(fields['temperature']).reshape((20, 20))  # Assume 20x20 grid
                temp_im = ax1.imshow(temp_data, cmap=self.config.temperature_cmap, 
                                   origin='lower', extent=[0, 10, 0, 10])
                ax1.set_title('Temperature (°C)')
                plt.colorbar(temp_im, ax=ax1, shrink=0.8)
            
            # Pollutant
            ax2 = fig.add_subplot(gs[0, 1])
            if 'pollutant' in fields:
                pollutant_data = np.array(fields['pollutant']).reshape((20, 20))
                pollutant_im = ax2.imshow(pollutant_data, cmap=self.config.pollutant_cmap,
                                        origin='lower', extent=[0, 10, 0, 10])
                ax2.set_title('Pollutant (μg/m³)')
                plt.colorbar(pollutant_im, ax=ax2, shrink=0.8)
            
            # Humidity
            ax3 = fig.add_subplot(gs[0, 2])
            if 'humidity' in fields:
                humidity_data = np.array(fields['humidity']).reshape((20, 20))
                humidity_im = ax3.imshow(humidity_data, cmap=self.config.humidity_cmap,
                                       origin='lower', extent=[0, 10, 0, 10])
                ax3.set_title('Humidity (%)')
                plt.colorbar(humidity_im, ax=ax3, shrink=0.8)
        
        # Sensor status
        ax4 = fig.add_subplot(gs[0, 3])
        if sensor_readings:
            # Plot recent sensor readings
            times = [r.timestamp for r in sensor_readings[-20:]]  # Last 20 readings
            temp_values = [r.measurements.get('temperature', np.nan) for r in sensor_readings[-20:]]
            
            if times and not all(np.isnan(temp_values)):
                ax4.plot(times, temp_values, 'o-', label='Temperature')
                ax4.set_title('Recent Sensor Readings')
                ax4.set_ylabel('Temperature (°C)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No sensor data\navailable', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Sensor Status')
        
        # Performance metrics
        ax5 = fig.add_subplot(gs[1, :2])
        if performance_metrics and 'no_data' not in performance_metrics:
            metrics_text = f"""Performance Metrics
            
Total Updates: {performance_metrics.get('total_updates', 0)}
Avg Processing Time: {performance_metrics.get('avg_processing_time', 0)*1000:.1f} ms
Recent Readings: {performance_metrics.get('recent_readings_count', 0)}
Active Fields: {len(performance_metrics.get('active_fields', []))}
Buffer Utilization: {performance_metrics.get('buffer_utilization', 0)*100:.1f}%
            """
        else:
            metrics_text = "Performance metrics\nnot available"
        
        ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax5.set_title('System Performance')
        ax5.axis('off')
        
        # Confidence intervals
        ax6 = fig.add_subplot(gs[1, 2:])
        if 'confidence' in current_estimates and current_estimates['confidence']:
            confidence = current_estimates['confidence']
            fields_conf = list(confidence.keys())
            mae_values = [confidence[f].get('mae', 0) for f in fields_conf]
            
            bars = ax6.bar(fields_conf, mae_values, color=['red', 'green', 'blue'][:len(fields_conf)])
            ax6.set_title('Model Uncertainty (MAE)')
            ax6.set_ylabel('Mean Absolute Error')
            
            # Add value labels on bars
            for bar, val in zip(bars, mae_values):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom')
        else:
            ax6.text(0.5, 0.5, 'No confidence\ndata available', 
                    transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title('Model Uncertainty')
        
        # System status
        ax7 = fig.add_subplot(gs[2, :])
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        status_text = f"""System Status - {current_time}
        
🟢 Data Assimilation: Active
🟢 Model Inference: Running  
🟢 Sensor Network: Online
🟢 Alert System: Monitoring
        
Last Update: {current_estimates.get('last_update', 'Unknown')}
Grid Resolution: {performance_metrics.get('grid_size', 'Unknown') if performance_metrics else 'Unknown'}
        """
        
        ax7.text(0.05, 0.95, status_text, transform=ax7.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax7.set_title('System Status')
        ax7.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.export_dpi, bbox_inches='tight')
            print(f"Real-time dashboard saved to {save_path}")
        
        plt.show()
        self.figures['real_time_dashboard'] = fig
        
        return fig
    
    def create_experiment_comparison(self,
                                   experiment_results: Dict[str, Dict],
                                   save_path: Optional[str] = None) -> None:
        """Create comparison visualization of multiple experiments."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for experiment comparison")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Smart City Experiment Comparison', fontsize=16, fontweight='bold')
        
        # Extract experiment names and metrics
        exp_names = list(experiment_results.keys())
        
        # 1. Training Loss Comparison
        ax = axes[0, 0]
        for exp_name, results in experiment_results.items():
            if 'training_history' in results and 'total_loss' in results['training_history']:
                history = results['training_history']['total_loss']
                ax.plot(history, label=exp_name, linewidth=2)
        
        ax.set_title('Training Loss Evolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. Final Loss Comparison
        ax = axes[0, 1]
        final_losses = []
        for exp_name, results in experiment_results.items():
            if 'best_loss' in results:
                final_losses.append(results['best_loss'])
            else:
                final_losses.append(np.nan)
        
        bars = ax.bar(exp_names, final_losses, color=['blue', 'green', 'red', 'orange', 'purple'][:len(exp_names)])
        ax.set_title('Final Validation Loss')
        ax.set_ylabel('Loss Value')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, final_losses):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom')
        
        # 3. Training Time Comparison
        ax = axes[0, 2]
        training_times = []
        for exp_name, results in experiment_results.items():
            if 'training_time' in results:
                training_times.append(results['training_time'])
            else:
                training_times.append(0)
        
        bars = ax.bar(exp_names, training_times, color=['blue', 'green', 'red', 'orange', 'purple'][:len(exp_names)])
        ax.set_title('Training Time')
        ax.set_ylabel('Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Convergence Analysis
        ax = axes[1, 0]
        for exp_name, results in experiment_results.items():
            if 'training_history' in results and 'validation_loss' in results['training_history']:
                val_history = results['training_history']['validation_loss']
                if len(val_history) > 0:
                    ax.plot(val_history, label=exp_name, linewidth=2, marker='o', markersize=4)
        
        ax.set_title('Validation Loss Convergence')
        ax.set_xlabel('Validation Step')
        ax.set_ylabel('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 5. Experiment Summary Table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary table
        table_data = []
        headers = ['Experiment', 'Best Loss', 'Time (s)', 'Iterations']
        
        for exp_name, results in experiment_results.items():
            row = [
                exp_name[:15] + '...' if len(exp_name) > 15 else exp_name,
                f"{results.get('best_loss', 0):.4f}",
                f"{results.get('training_time', 0):.1f}",
                f"{results.get('total_iterations', 0)}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center', 
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax.set_title('Experiment Summary')
        
        # 6. Performance Metrics
        ax = axes[1, 2]
        
        # Plot multiple metrics if available
        metrics_to_plot = ['best_loss', 'training_time', 'total_iterations']
        metric_values = {metric: [] for metric in metrics_to_plot}
        
        for exp_name, results in experiment_results.items():
            for metric in metrics_to_plot:
                metric_values[metric].append(results.get(metric, 0))
        
        # Normalize values for comparison
        x_pos = np.arange(len(exp_names))
        bar_width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = np.array(metric_values[metric])
            if np.max(values) > 0:
                normalized_values = values / np.max(values)
                ax.bar(x_pos + i*bar_width, normalized_values, bar_width, 
                      label=metric.replace('_', ' ').title())
        
        ax.set_title('Normalized Performance Metrics')
        ax.set_ylabel('Normalized Value')
        ax.set_xticks(x_pos + bar_width)
        ax.set_xticklabels([name[:10] for name in exp_names], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.export_dpi, bbox_inches='tight')
            print(f"Experiment comparison saved to {save_path}")
        
        plt.show()
        self.figures['experiment_comparison'] = fig
        
        return fig
    
    def save_all_figures(self, output_dir: str):
        """Save all created figures to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            filepath = os.path.join(output_dir, f'{name}.{self.config.export_format}')
            fig.savefig(filepath, dpi=self.config.export_dpi, bbox_inches='tight')
            print(f"Saved {name} to {filepath}")
    
    def close_all_figures(self):
        """Close all figures and clear memory."""
        if HAS_MATPLOTLIB:
            plt.close('all')
        self.figures.clear()


def create_visualization_demo():
    """Create demonstration of visualization capabilities."""
    print("Smart City Visualization Demo")
    print("=" * 40)
    
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization demo.")
        return
    
    # Create demo data
    from urban_data_generator import UrbanClimateGenerator
    
    generator = UrbanClimateGenerator(nx=31, ny=31, nt=13)
    urban_dataset = generator.generate_complete_urban_scenario('mixed_city', 'summer_day')
    
    # Create visualizer
    config = VisualizationConfig(
        figure_size=(16, 12),
        temperature_cmap='hot',
        pollutant_cmap='Reds',
        humidity_cmap='Blues'
    )
    
    visualizer = SmartCityVisualizer(config)
    
    print("Creating urban overview dashboard...")
    visualizer.create_urban_overview_dashboard(urban_dataset)
    
    print("Creating 3D visualization...")
    visualizer.create_3d_urban_visualization(urban_dataset, 'temperature')
    
    # Create mock real-time data
    print("Creating real-time dashboard...")
    current_estimates = {
        'fields': {
            'temperature': np.random.normal(22, 3, 400),
            'pollutant': np.random.normal(45, 15, 400),
            'humidity': np.random.normal(65, 10, 400)
        },
        'confidence': {
            'temperature': {'mae': 0.8},
            'pollutant': {'mae': 5.2},
            'humidity': {'mae': 3.1}
        },
        'last_update': '5.2 seconds ago'
    }
    
    performance_metrics = {
        'total_updates': 150,
        'avg_processing_time': 0.025,
        'recent_readings_count': 25,
        'active_fields': ['temperature', 'pollutant', 'humidity'],
        'buffer_utilization': 0.65,
        'grid_size': 400
    }
    
    visualizer.create_real_time_dashboard(current_estimates, 
                                        performance_metrics=performance_metrics)
    
    # Create mock experiment comparison
    print("Creating experiment comparison...")
    mock_experiments = {
        'baseline_mixed_city': {
            'best_loss': 0.856,
            'training_time': 45.2,
            'total_iterations': 1200,
            'training_history': {
                'total_loss': np.exp(-np.linspace(0, 5, 100)) + 0.1*np.random.rand(100),
                'validation_loss': np.exp(-np.linspace(0, 4, 20)) + 0.1*np.random.rand(20)
            }
        },
        'heat_island_downtown': {
            'best_loss': 0.734,
            'training_time': 62.8,
            'total_iterations': 1580,
            'training_history': {
                'total_loss': np.exp(-np.linspace(0, 6, 120)) + 0.1*np.random.rand(120),
                'validation_loss': np.exp(-np.linspace(0, 5, 25)) + 0.1*np.random.rand(25)
            }
        },
        'air_quality_industrial': {
            'best_loss': 0.912,
            'training_time': 38.5,
            'total_iterations': 950,
            'training_history': {
                'total_loss': np.exp(-np.linspace(0, 4, 80)) + 0.1*np.random.rand(80),
                'validation_loss': np.exp(-np.linspace(0, 3.5, 18)) + 0.1*np.random.rand(18)
            }
        }
    }
    
    visualizer.create_experiment_comparison(mock_experiments)
    
    print("\nVisualization demo completed successfully!")
    print("All visualization capabilities demonstrated.")
    
    return visualizer


if __name__ == "__main__":
    demo_visualizer = create_visualization_demo()