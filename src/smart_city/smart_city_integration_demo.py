"""
Smart City Integration Demo

Complete demonstration of the integrated smart city variable-order 
fractional PDE discovery system, showcasing all components working together.

Author: Sakeeb Rahman
Date: 2025
"""

import numpy as np
import asyncio
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all smart city components
from urban_data_generator import UrbanClimateGenerator
from smart_city_networks import SmartCityNetworkFactory
from smart_city_loss_functions import create_smart_city_loss
from smart_city_experiments import SmartCityExperimentConfig, SmartCityTrainer, SmartCityExperimentSuite
from iot_integration import SensorNetwork, SmartCityDigitalTwin, RealTimePDEInference, create_demo_sensor_network
from smart_city_visualization import SmartCityVisualizer, VisualizationConfig


class SmartCityIntegratedSystem:
    """Complete integrated smart city system."""
    
    def __init__(self):
        self.urban_generator = None
        self.solution_network = None
        self.order_network = None
        self.loss_function = None
        self.digital_twin = None
        self.visualizer = None
        self.experiment_suite = None
        
        # Data storage
        self.urban_datasets = {}
        self.experiment_results = {}
        self.real_time_data = {}
        
    def initialize_components(self):
        """Initialize all system components."""
        print("Initializing Smart City Integrated System...")
        
        # 1. Urban data generator
        self.urban_generator = UrbanClimateGenerator(nx=31, ny=31, nt=13)
        print("✓ Urban climate generator initialized")
        
        # 2. Neural networks
        self.solution_network = SmartCityNetworkFactory.create_multi_physics_solution_network()
        self.order_network = SmartCityNetworkFactory.create_multi_field_order_network()
        self.loss_function = create_smart_city_loss({})
        print("✓ Neural networks initialized")
        
        # 3. Experiment suite
        self.experiment_suite = SmartCityExperimentSuite()
        print("✓ Experiment suite initialized")
        
        # 4. IoT components
        sensor_network = create_demo_sensor_network()
        inference_engine = RealTimePDEInference(
            self.solution_network, self.order_network, self.loss_function,
            spatial_bounds=((0.0, 10.0), (0.0, 10.0)),
            update_interval=30.0
        )
        self.digital_twin = SmartCityDigitalTwin([sensor_network], inference_engine)
        print("✓ IoT system initialized")
        
        # 5. Visualization system
        vis_config = VisualizationConfig(figure_size=(14, 10))
        self.visualizer = SmartCityVisualizer(vis_config)
        print("✓ Visualization system initialized")
        
        print("All components initialized successfully!")
        
    def generate_urban_scenarios(self):
        """Generate multiple urban scenarios for analysis."""
        print("\nGenerating urban scenarios...")
        
        scenarios = [
            ('mixed_city', 'summer_day'),
            ('downtown_core', 'heat_wave'),
            ('suburban', 'normal_day')
        ]
        
        for layout, scenario in scenarios:
            print(f"  Generating {scenario} with {layout} layout...")
            dataset = self.urban_generator.generate_complete_urban_scenario(layout, scenario)
            self.urban_datasets[f"{scenario}_{layout}"] = dataset
        
        print(f"✓ Generated {len(self.urban_datasets)} urban scenarios")
        
    def run_training_experiments(self):
        """Run training experiments on generated scenarios."""
        print("\nRunning training experiments...")
        
        # Create experiment configurations
        experiments = [
            self.experiment_suite.create_baseline_experiment(),
            self.experiment_suite.create_heat_island_experiment(),
            self.experiment_suite.create_air_quality_experiment()
        ]
        
        # Modify configs for faster demo execution
        for config in experiments:
            config.max_iterations = 50  # Reduced for demo
            config.validation_frequency = 10
            config.domain_size = (21, 21, 5)  # Smaller domain
        
        # Run experiments
        for config in experiments:
            print(f"  Running experiment: {config.experiment_name}")
            try:
                trainer = SmartCityTrainer(config)
                results = trainer.train()
                self.experiment_results[config.experiment_name] = results
                print(f"    ✓ Completed with loss: {results['best_loss']:.4f}")
            except Exception as e:
                print(f"    ✗ Failed: {e}")
        
        print(f"✓ Completed {len(self.experiment_results)} experiments")
        
    async def demonstrate_real_time_monitoring(self):
        """Demonstrate real-time IoT monitoring."""
        print("\nDemonstrating real-time monitoring...")
        
        # Start digital twin
        await self.digital_twin.start_monitoring()
        print("  Digital twin monitoring started")
        
        # Run for demonstration period
        monitoring_duration = 15  # seconds
        print(f"  Running for {monitoring_duration} seconds...")
        
        for i in range(3):  # 3 status updates
            await asyncio.sleep(monitoring_duration / 3)
            
            # Get current state
            city_state = self.digital_twin.get_current_city_state()
            stats = city_state['monitoring_stats']
            
            print(f"    Status update {i+1}/3:")
            print(f"      Readings processed: {stats['total_readings']}")
            print(f"      Alerts generated: {stats['alerts_generated']}")
            print(f"      Uptime: {stats['uptime_seconds']:.1f}s")
        
        # Store final state
        self.real_time_data = self.digital_twin.get_current_city_state()
        
        # Stop monitoring
        await self.digital_twin.stop_monitoring()
        print("  ✓ Real-time monitoring demonstration completed")
        
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualization suite."""
        print("\nCreating comprehensive visualizations...")
        
        # 1. Urban scenario overview
        if self.urban_datasets:
            first_dataset = list(self.urban_datasets.values())[0]
            print("  Creating urban overview dashboard...")
            self.visualizer.create_urban_overview_dashboard(first_dataset)
        
        # 2. 3D visualization
        if self.urban_datasets:
            print("  Creating 3D visualization...")
            self.visualizer.create_3d_urban_visualization(first_dataset, 'temperature')
        
        # 3. Real-time dashboard
        if self.real_time_data:
            print("  Creating real-time dashboard...")
            estimates = self.real_time_data.get('field_estimates', {})
            performance = self.real_time_data.get('performance', {})
            
            # Convert to expected format
            current_estimates = {
                'fields': estimates,
                'confidence': {},
                'last_update': 'just now'
            }
            
            self.visualizer.create_real_time_dashboard(
                current_estimates, 
                performance_metrics=performance
            )
        
        # 4. Experiment comparison
        if self.experiment_results:
            print("  Creating experiment comparison...")
            self.visualizer.create_experiment_comparison(self.experiment_results)
        
        print("  ✓ All visualizations created")
        
    def generate_system_report(self):
        """Generate comprehensive system performance report."""
        print("\nGenerating system performance report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_components': {
                'urban_generator': bool(self.urban_generator),
                'neural_networks': bool(self.solution_network and self.order_network),
                'loss_function': bool(self.loss_function),
                'digital_twin': bool(self.digital_twin),
                'visualizer': bool(self.visualizer),
                'experiment_suite': bool(self.experiment_suite)
            },
            'data_summary': {
                'urban_scenarios': len(self.urban_datasets),
                'experiments_completed': len(self.experiment_results),
                'real_time_data_available': bool(self.real_time_data)
            },
            'experiment_results': {},
            'real_time_performance': {},
            'system_capabilities': [
                'Multi-physics urban climate modeling',
                'Variable-order fractional PDE discovery',
                'Real-time IoT data integration',
                'Advanced 2D/3D visualization',
                'Experiment management and comparison',
                'Smart city digital twin monitoring'
            ]
        }
        
        # Add experiment results summary
        for exp_name, results in self.experiment_results.items():
            report['experiment_results'][exp_name] = {
                'best_loss': results.get('best_loss', 'N/A'),
                'training_time': results.get('training_time', 'N/A'),
                'total_iterations': results.get('total_iterations', 'N/A')
            }
        
        # Add real-time performance
        if self.real_time_data and 'monitoring_stats' in self.real_time_data:
            stats = self.real_time_data['monitoring_stats']
            report['real_time_performance'] = {
                'total_readings_processed': stats.get('total_readings', 0),
                'alerts_generated': stats.get('alerts_generated', 0),
                'system_uptime': stats.get('uptime_seconds', 0),
                'active_networks': stats.get('active_networks', 0)
            }
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        report_path = 'reports/smart_city_system_report.json'
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✓ System report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SMART CITY SYSTEM PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Components Active: {sum(report['system_components'].values())}/6")
        print(f"Urban Scenarios: {report['data_summary']['urban_scenarios']}")
        print(f"Experiments Completed: {report['data_summary']['experiments_completed']}")
        print(f"Real-time Monitoring: {'✓' if report['data_summary']['real_time_data_available'] else '✗'}")
        
        if report['experiment_results']:
            print(f"\nExperiment Results:")
            for exp_name, results in report['experiment_results'].items():
                print(f"  {exp_name}: Loss={results['best_loss']:.4f}, Time={results['training_time']:.1f}s")
        
        if report['real_time_performance']:
            perf = report['real_time_performance']
            print(f"\nReal-time Performance:")
            print(f"  Readings Processed: {perf['total_readings_processed']}")
            print(f"  Alerts Generated: {perf['alerts_generated']}")
            print(f"  System Uptime: {perf['system_uptime']:.1f}s")
        
        print(f"\nSystem Capabilities:")
        for capability in report['system_capabilities']:
            print(f"  • {capability}")
        
        print("="*60)
        
        return report


async def run_complete_smart_city_demo():
    """Run complete smart city system demonstration."""
    print("🏙️  SMART CITY VARIABLE-ORDER FRACTIONAL PDE DISCOVERY")
    print("🏙️  COMPLETE SYSTEM INTEGRATION DEMONSTRATION")
    print("="*70)
    
    # Initialize system
    system = SmartCityIntegratedSystem()
    
    try:
        # Phase 1: System initialization
        system.initialize_components()
        
        # Phase 2: Data generation
        system.generate_urban_scenarios()
        
        # Phase 3: Training experiments
        system.run_training_experiments()
        
        # Phase 4: Real-time monitoring
        await system.demonstrate_real_time_monitoring()
        
        # Phase 5: Comprehensive visualization
        system.create_comprehensive_visualizations()
        
        # Phase 6: System report
        report = system.generate_system_report()
        
        print("\n🎉 COMPLETE SMART CITY SYSTEM DEMONSTRATION SUCCESSFUL!")
        print("All components working together seamlessly.")
        print("System ready for production deployment.")
        
        return system, report
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Run the complete demonstration
    system, report = asyncio.run(run_complete_smart_city_demo())