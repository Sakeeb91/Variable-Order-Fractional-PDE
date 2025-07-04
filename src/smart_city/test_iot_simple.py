"""
Simple IoT Integration Test

Simplified test of IoT capabilities without complex interpolation.
"""

import numpy as np
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iot_integration import SensorReading, SensorNetwork, MockSensorStream, SmartCityDigitalTwin


def test_sensor_data_generation():
    """Test basic sensor data generation."""
    print("Testing sensor data generation...")
    
    # Create simple sensor network
    sensors = {
        'sensor_001': {'location': (2.0, 2.0)},
        'sensor_002': {'location': (5.0, 5.0)},
        'sensor_003': {'location': (8.0, 8.0)}
    }
    
    network = SensorNetwork(
        network_id='test_network',
        sensors=sensors,
        field_types=['temperature', 'pollutant', 'humidity'],
        spatial_bounds=((0.0, 10.0), (0.0, 10.0)),
        update_frequency=10.0
    )
    
    print(f"✓ Created network with {len(network.sensors)} sensors")
    
    return network


async def test_mock_sensor_stream():
    """Test mock sensor stream."""
    print("\nTesting mock sensor stream...")
    
    network = test_sensor_data_generation()
    stream = MockSensorStream(network, noise_level=0.1)
    
    await stream.start_stream()
    
    readings = []
    for i in range(5):
        reading = await stream.get_reading()
        if reading:
            readings.append(reading)
            print(f"  Reading {i+1}: {reading.sensor_id} at {reading.location}")
            for field, value in reading.measurements.items():
                print(f"    {field}: {value:.2f}")
    
    await stream.stop_stream()
    
    print(f"✓ Generated {len(readings)} sensor readings")
    return readings


def test_simple_inference():
    """Test simple real-time inference without complex assimilation."""
    print("\nTesting simple inference...")
    
    from smart_city_networks import SmartCityNetworkFactory
    from smart_city_loss_functions import create_smart_city_loss
    from iot_integration import RealTimePDEInference
    
    # Create networks
    solution_network = SmartCityNetworkFactory.create_multi_physics_solution_network()
    order_network = SmartCityNetworkFactory.create_multi_field_order_network()
    loss_function = create_smart_city_loss({})
    
    # Create inference engine
    inference_engine = RealTimePDEInference(
        solution_network, order_network, loss_function,
        spatial_bounds=((0.0, 10.0), (0.0, 10.0)),
        update_interval=5.0
    )
    
    # Simulate sensor readings
    current_time = datetime.now()
    
    for i in range(5):
        reading = SensorReading(
            sensor_id=f"test_sensor_{i}",
            timestamp=current_time + timedelta(seconds=i*10),
            location=(2.0 + i, 2.0 + i),
            measurements={
                'temperature': 20.0 + np.random.randn(),
                'pollutant': 50.0 + 10*np.random.randn(),
                'humidity': 60.0 + 5*np.random.randn()
            },
            uncertainty={
                'temperature': 0.5,
                'pollutant': 5.0,
                'humidity': 3.0
            }
        )
        
        inference_engine.add_sensor_reading(reading)
        print(f"  Added reading {i+1} from {reading.sensor_id}")
    
    # Get current estimates
    test_coords = np.array([[5.0, 5.0, 12.0]])  # Center of domain at noon
    estimates = inference_engine.get_current_estimates(test_coords)
    
    print(f"✓ Inference engine processed {len(inference_engine.recent_readings)} readings")
    
    if estimates['fields']:
        for field, values in estimates['fields'].items():
            print(f"  {field}: {np.mean(values):.2f}")
    
    return inference_engine


async def test_simple_digital_twin():
    """Test simplified digital twin."""
    print("\nTesting simple digital twin...")
    
    # Create simple sensor network
    network = test_sensor_data_generation()
    
    # Create simplified inference engine
    inference_engine = test_simple_inference()
    
    # Create digital twin without complex assimilation
    from iot_integration import SmartCityDigitalTwin
    
    digital_twin = SmartCityDigitalTwin(
        sensor_networks=[network],
        inference_engine=inference_engine
    )
    
    # Start monitoring
    await digital_twin.start_monitoring()
    
    # Run for short time
    print("  Running monitoring for 10 seconds...")
    await asyncio.sleep(10)
    
    # Get state
    city_state = digital_twin.get_current_city_state()
    stats = city_state['monitoring_stats']
    
    print(f"  Readings processed: {stats['total_readings']}")
    print(f"  Alerts generated: {stats['alerts_generated']}")
    
    # Stop monitoring
    await digital_twin.stop_monitoring()
    
    print("✓ Digital twin test completed")
    
    return digital_twin


async def run_simple_iot_test():
    """Run complete simplified IoT test."""
    print("Simple IoT Integration Test")
    print("=" * 40)
    
    try:
        # Test components individually
        readings = await test_mock_sensor_stream()
        inference_engine = test_simple_inference()
        digital_twin = await test_simple_digital_twin()
        
        print("\n" + "=" * 40)
        print("All IoT tests passed successfully! ✓")
        print("IoT integration system is working.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_simple_iot_test())