"""
IoT Integration and Real-Time Data Processing for Smart City Applications

This module provides real-time sensor data integration, streaming analytics,
and live urban climate monitoring capabilities for the variable-order
fractional PDE discovery framework.

Author: Sakeeb Rahman
Date: 2025
"""

import numpy as np
import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from queue import Queue, Empty
import asyncio
from datetime import datetime, timedelta
import csv
import os


@dataclass
class SensorReading:
    """Individual sensor measurement."""
    sensor_id: str
    timestamp: datetime
    location: Tuple[float, float]  # (x, y) coordinates in km
    measurements: Dict[str, float]  # field_name -> value
    uncertainty: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'measurements': self.measurements,
            'uncertainty': self.uncertainty,
            'metadata': self.metadata
        }


@dataclass
class SensorNetwork:
    """Sensor network configuration and metadata."""
    network_id: str
    sensors: Dict[str, Dict]  # sensor_id -> sensor_config
    field_types: List[str]
    spatial_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    update_frequency: float  # seconds
    data_retention_hours: float = 24.0
    
    def get_sensor_locations(self) -> np.ndarray:
        """Get array of sensor locations."""
        locations = []
        for sensor_config in self.sensors.values():
            locations.append(sensor_config['location'])
        return np.array(locations)


class SensorDataStream(ABC):
    """Abstract base class for sensor data streams."""
    
    @abstractmethod
    async def start_stream(self):
        """Start the data stream."""
        pass
    
    @abstractmethod
    async def stop_stream(self):
        """Stop the data stream."""
        pass
    
    @abstractmethod
    async def get_reading(self) -> Optional[SensorReading]:
        """Get next sensor reading."""
        pass


class MockSensorStream(SensorDataStream):
    """Mock sensor stream for testing and development."""
    
    def __init__(self, sensor_network: SensorNetwork, noise_level: float = 0.1):
        self.sensor_network = sensor_network
        self.noise_level = noise_level
        self.is_running = False
        self.current_time = datetime.now()
        
        # Simulate baseline conditions
        self.baseline_conditions = {
            'temperature': 22.0,  # °C
            'pollutant': 45.0,    # μg/m³
            'humidity': 65.0,     # %
            'wind_speed': 3.0,    # m/s
            'pressure': 1013.25   # hPa
        }
        
    async def start_stream(self):
        """Start mock data generation."""
        self.is_running = True
        print(f"Started mock sensor stream for network: {self.sensor_network.network_id}")
        
    async def stop_stream(self):
        """Stop mock data generation."""
        self.is_running = False
        print(f"Stopped mock sensor stream for network: {self.sensor_network.network_id}")
        
    async def get_reading(self) -> Optional[SensorReading]:
        """Generate mock sensor reading."""
        if not self.is_running:
            return None
            
        # Simulate time progression
        self.current_time += timedelta(seconds=self.sensor_network.update_frequency)
        
        # Select random sensor
        sensor_ids = list(self.sensor_network.sensors.keys())
        if not sensor_ids:
            return None
            
        sensor_id = np.random.choice(sensor_ids)
        sensor_config = self.sensor_network.sensors[sensor_id]
        
        # Generate realistic measurements with spatial and temporal patterns
        measurements = {}
        uncertainty = {}
        
        for field in self.sensor_network.field_types:
            if field in self.baseline_conditions:
                base_value = self.baseline_conditions[field]
                
                # Add diurnal cycle for temperature
                if field == 'temperature':
                    hour = self.current_time.hour
                    diurnal = 5.0 * np.sin(2 * np.pi * (hour - 6) / 24)
                    base_value += diurnal
                
                # Add spatial variation
                x, y = sensor_config['location']
                spatial_factor = 0.1 * np.sin(x) * np.cos(y)
                
                # Add noise
                noise = self.noise_level * base_value * np.random.randn()
                
                value = base_value + spatial_factor + noise
                
                # Field-specific constraints
                if field == 'temperature':
                    value = max(-10, min(50, value))
                    uncertainty[field] = 0.5  # ±0.5°C
                elif field == 'pollutant':
                    value = max(0, value)
                    uncertainty[field] = 5.0  # ±5 μg/m³
                elif field == 'humidity':
                    value = max(0, min(100, value))
                    uncertainty[field] = 3.0  # ±3%
                
                measurements[field] = value
        
        return SensorReading(
            sensor_id=sensor_id,
            timestamp=self.current_time,
            location=sensor_config['location'],
            measurements=measurements,
            uncertainty=uncertainty,
            metadata={'stream_type': 'mock', 'network_id': self.sensor_network.network_id}
        )


class RealTimePDEInference:
    """Real-time fractional PDE inference engine."""
    
    def __init__(self,
                 solution_network,
                 order_network,
                 loss_function,
                 spatial_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 update_interval: float = 60.0):
        """
        Initialize real-time inference engine.
        
        Args:
            solution_network: Trained solution network
            order_network: Trained order network
            loss_function: Loss function for optimization
            spatial_bounds: Domain boundaries ((x_min, x_max), (y_min, y_max))
            update_interval: Model update interval in seconds
        """
        self.solution_network = solution_network
        self.order_network = order_network
        self.loss_function = loss_function
        self.spatial_bounds = spatial_bounds
        self.update_interval = update_interval
        
        self.recent_readings = []
        self.max_readings_buffer = 1000
        self.last_update_time = time.time()
        
        # Real-time state
        self.current_field_estimates = {}
        self.current_alpha_estimates = {}
        self.confidence_intervals = {}
        
        # Performance tracking
        self.inference_times = []
        self.update_counts = 0
        
    def add_sensor_reading(self, reading: SensorReading):
        """Add new sensor reading to the buffer."""
        self.recent_readings.append(reading)
        
        # Maintain buffer size
        if len(self.recent_readings) > self.max_readings_buffer:
            self.recent_readings = self.recent_readings[-self.max_readings_buffer:]
        
        # Check if model update is needed
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            self._update_model()
            self.last_update_time = current_time
    
    def _update_model(self):
        """Update model with recent sensor data."""
        if len(self.recent_readings) < 10:  # Need minimum data
            return
            
        start_time = time.time()
        
        # Extract recent data for online learning
        recent_coords = []
        recent_observations = {field: [] for field in ['temperature', 'pollutant', 'humidity']}
        
        cutoff_time = datetime.now() - timedelta(hours=1)  # Use last hour of data
        
        for reading in self.recent_readings:
            if reading.timestamp > cutoff_time:
                x, y = reading.location
                t = (reading.timestamp.hour + reading.timestamp.minute/60.0)  # Hour of day
                
                recent_coords.append([x, y, t])
                
                for field in recent_observations.keys():
                    if field in reading.measurements:
                        recent_observations[field].append(reading.measurements[field])
                    else:
                        recent_observations[field].append(np.nan)
        
        if len(recent_coords) == 0:
            return
            
        coords_array = np.array(recent_coords)
        coords_2d = coords_array[:, :2]
        
        # Get current model predictions
        solution_predictions = self.solution_network.forward(coords_array)
        alpha_predictions = self.order_network.forward(coords_2d)
        
        # Update field estimates
        self.current_field_estimates = solution_predictions.copy()
        self.current_alpha_estimates = alpha_predictions.copy()
        
        # Simple confidence estimation based on recent prediction accuracy
        for field in solution_predictions.keys():
            if field in recent_observations:
                obs_values = np.array([x for x in recent_observations[field] if not np.isnan(x)])
                pred_values = solution_predictions[field][:len(obs_values)]
                
                if len(obs_values) > 0:
                    residuals = obs_values - pred_values
                    std_error = np.std(residuals) if len(residuals) > 1 else 1.0
                    self.confidence_intervals[field] = {
                        'std_error': std_error,
                        'mae': np.mean(np.abs(residuals)) if len(residuals) > 0 else 1.0
                    }
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.update_counts += 1
        
        # Keep inference time history manageable
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
    
    def get_current_estimates(self, coordinates: np.ndarray) -> Dict:
        """Get current field and alpha estimates at given coordinates."""
        if len(self.current_field_estimates) == 0:
            # No estimates available yet, use network directly
            solution_estimates = self.solution_network.forward(coordinates)
            alpha_estimates = self.order_network.forward(coordinates[:, :2])
            
            return {
                'fields': solution_estimates,
                'alpha': alpha_estimates,
                'confidence': {},
                'last_update': 'initial'
            }
        
        # Use cached estimates (would interpolate in full implementation)
        return {
            'fields': self.current_field_estimates,
            'alpha': self.current_alpha_estimates,
            'confidence': self.confidence_intervals,
            'last_update': time.time() - self.last_update_time
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get real-time performance metrics."""
        if len(self.inference_times) == 0:
            return {'no_data': True}
            
        return {
            'total_updates': self.update_counts,
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'recent_readings': len(self.recent_readings),
            'buffer_utilization': len(self.recent_readings) / self.max_readings_buffer
        }


class SmartCityDigitalTwin:
    """Digital twin for real-time smart city monitoring."""
    
    def __init__(self,
                 sensor_networks: List[SensorNetwork],
                 inference_engine: RealTimePDEInference,
                 alert_thresholds: Dict[str, Tuple[float, float]] = None):
        """
        Initialize smart city digital twin.
        
        Args:
            sensor_networks: List of sensor networks to monitor
            inference_engine: Real-time PDE inference engine
            alert_thresholds: Alert thresholds for each field (min, max)
        """
        self.sensor_networks = sensor_networks
        self.inference_engine = inference_engine
        self.alert_thresholds = alert_thresholds or {
            'temperature': (-5.0, 45.0),
            'pollutant': (0.0, 150.0),
            'humidity': (10.0, 95.0)
        }
        
        self.data_streams = {}
        self.is_running = False
        self.reading_queue = Queue()
        self.alert_queue = Queue()
        
        # Monitoring statistics
        self.total_readings = 0
        self.alerts_generated = 0
        self.start_time = None
        
    async def start_monitoring(self):
        """Start real-time monitoring."""
        print("Starting Smart City Digital Twin monitoring...")
        self.is_running = True
        self.start_time = time.time()
        
        # Start sensor streams
        for network in self.sensor_networks:
            stream = MockSensorStream(network)  # Would use real streams in production
            await stream.start_stream()
            self.data_streams[network.network_id] = stream
        
        # Start processing tasks
        asyncio.create_task(self._data_collection_loop())
        asyncio.create_task(self._data_processing_loop())
        asyncio.create_task(self._alert_monitoring_loop())
        
        print(f"Monitoring started for {len(self.sensor_networks)} sensor networks")
        
    async def stop_monitoring(self):
        """Stop real-time monitoring."""
        print("Stopping Smart City Digital Twin monitoring...")
        self.is_running = False
        
        # Stop sensor streams
        for stream in self.data_streams.values():
            await stream.stop_stream()
        
        self.data_streams.clear()
        print("Monitoring stopped")
        
    async def _data_collection_loop(self):
        """Continuous data collection from sensor streams."""
        while self.is_running:
            for network_id, stream in self.data_streams.items():
                try:
                    reading = await stream.get_reading()
                    if reading:
                        self.reading_queue.put(reading)
                        self.total_readings += 1
                except Exception as e:
                    print(f"Error collecting data from {network_id}: {e}")
            
            await asyncio.sleep(1.0)  # Check every second
    
    async def _data_processing_loop(self):
        """Process incoming sensor readings."""
        while self.is_running:
            try:
                # Process readings from queue
                while not self.reading_queue.empty():
                    reading = self.reading_queue.get_nowait()
                    
                    # Add to inference engine
                    self.inference_engine.add_sensor_reading(reading)
                    
                    # Check for alerts
                    self._check_alerts(reading)
                    
            except Empty:
                pass
            except Exception as e:
                print(f"Error processing data: {e}")
            
            await asyncio.sleep(0.1)  # Process every 100ms
    
    async def _alert_monitoring_loop(self):
        """Monitor and handle alerts."""
        while self.is_running:
            try:
                while not self.alert_queue.empty():
                    alert = self.alert_queue.get_nowait()
                    await self._handle_alert(alert)
                    
            except Empty:
                pass
            except Exception as e:
                print(f"Error in alert monitoring: {e}")
            
            await asyncio.sleep(5.0)  # Check alerts every 5 seconds
    
    def _check_alerts(self, reading: SensorReading):
        """Check if reading triggers any alerts."""
        for field, value in reading.measurements.items():
            if field in self.alert_thresholds:
                min_val, max_val = self.alert_thresholds[field]
                
                if value < min_val or value > max_val:
                    alert = {
                        'type': 'threshold_violation',
                        'field': field,
                        'value': value,
                        'threshold': self.alert_thresholds[field],
                        'sensor_id': reading.sensor_id,
                        'location': reading.location,
                        'timestamp': reading.timestamp,
                        'severity': 'high' if value < min_val * 0.8 or value > max_val * 1.2 else 'medium'
                    }
                    
                    self.alert_queue.put(alert)
                    self.alerts_generated += 1
    
    async def _handle_alert(self, alert: Dict):
        """Handle generated alert."""
        print(f"🚨 ALERT: {alert['field']} = {alert['value']:.2f} at {alert['location']} "
              f"(threshold: {alert['threshold']}) - Severity: {alert['severity']}")
        
        # In production, would integrate with city management systems
        # - Send notifications
        # - Trigger automated responses
        # - Update dashboards
        # - Log to monitoring systems
    
    def get_current_city_state(self) -> Dict:
        """Get current city-wide state estimates."""
        # Create representative grid for city-wide estimates
        x_min, x_max = self.inference_engine.spatial_bounds[0]
        y_min, y_max = self.inference_engine.spatial_bounds[1]
        
        grid_size = 20
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        
        X, Y = np.meshgrid(x_grid, y_grid)
        coords = np.column_stack([X.flatten(), Y.flatten(), 
                                np.full(X.size, datetime.now().hour)])
        
        # Get current estimates
        estimates = self.inference_engine.get_current_estimates(coords)
        
        return {
            'spatial_grid': {'x': x_grid, 'y': y_grid, 'X': X, 'Y': Y},
            'field_estimates': estimates['fields'],
            'alpha_estimates': estimates['alpha'],
            'confidence': estimates['confidence'],
            'performance': self.inference_engine.get_performance_metrics(),
            'monitoring_stats': {
                'total_readings': self.total_readings,
                'alerts_generated': self.alerts_generated,
                'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
                'active_networks': len(self.data_streams)
            }
        }


def create_demo_sensor_network() -> SensorNetwork:
    """Create demonstration sensor network for testing."""
    sensors = {}
    
    # Create distributed sensor locations across city
    locations = [
        (2.0, 2.0),   # Downtown area
        (5.0, 5.0),   # City center
        (8.0, 2.0),   # Industrial area
        (2.0, 8.0),   # Residential area
        (8.0, 8.0),   # Suburban area
        (5.0, 2.0),   # Commercial area
        (5.0, 8.0),   # Park area
        (1.0, 5.0),   # River area
        (9.0, 5.0),   # Airport area
        (5.0, 1.0),   # Port area
    ]
    
    for i, location in enumerate(locations):
        sensor_id = f"sensor_{i+1:03d}"
        sensors[sensor_id] = {
            'location': location,
            'sensor_type': 'multi_parameter',
            'installation_date': '2024-01-01',
            'last_calibration': '2024-12-01',
            'status': 'active'
        }
    
    return SensorNetwork(
        network_id='demo_city_network',
        sensors=sensors,
        field_types=['temperature', 'pollutant', 'humidity'],
        spatial_bounds=((0.0, 10.0), (0.0, 10.0)),
        update_frequency=30.0,  # 30 seconds
        data_retention_hours=24.0
    )


async def run_iot_demo():
    """Run IoT integration demonstration."""
    print("Smart City IoT Integration Demo")
    print("=" * 50)
    
    # Create sensor network
    sensor_network = create_demo_sensor_network()
    print(f"Created sensor network with {len(sensor_network.sensors)} sensors")
    
    # Create networks for inference (simplified for demo)
    from smart_city_networks import SmartCityNetworkFactory
    from smart_city_loss_functions import create_smart_city_loss
    
    solution_network = SmartCityNetworkFactory.create_multi_physics_solution_network()
    order_network = SmartCityNetworkFactory.create_multi_field_order_network()
    loss_function = create_smart_city_loss({})
    
    # Create inference engine
    inference_engine = RealTimePDEInference(
        solution_network, order_network, loss_function,
        spatial_bounds=((0.0, 10.0), (0.0, 10.0)),
        update_interval=30.0
    )
    
    # Create digital twin
    digital_twin = SmartCityDigitalTwin(
        sensor_networks=[sensor_network],
        inference_engine=inference_engine
    )
    
    # Start monitoring
    await digital_twin.start_monitoring()
    
    # Run demo for 30 seconds
    print("Running live monitoring demo...")
    for i in range(6):  # 6 iterations of 5 seconds each
        await asyncio.sleep(5)
        
        # Get current city state
        city_state = digital_twin.get_current_city_state()
        stats = city_state['monitoring_stats']
        
        print(f"Demo step {i+1}/6:")
        print(f"  Readings processed: {stats['total_readings']}")
        print(f"  Alerts generated: {stats['alerts_generated']}")
        print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
        
        if city_state['field_estimates']:
            for field, values in city_state['field_estimates'].items():
                if len(values) > 0:
                    print(f"  {field}: {np.mean(values):.2f} ± {np.std(values):.2f}")
    
    # Stop monitoring
    await digital_twin.stop_monitoring()
    
    print("\nIoT Demo completed successfully!")
    print(f"Total readings: {digital_twin.total_readings}")
    print(f"Total alerts: {digital_twin.alerts_generated}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_iot_demo())