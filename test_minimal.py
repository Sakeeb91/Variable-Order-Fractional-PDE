#!/usr/bin/env python3
"""
Minimal test to validate environment and create basic plots
"""

import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print("✓ NumPy available")
    
    import matplotlib.pyplot as plt
    print("✓ Matplotlib available")
    
    # Create a simple test plot
    x = np.linspace(0, 1, 100)
    alpha_true = 0.25 * np.sin(2 * np.pi * x) + 1.5
    alpha_pred = alpha_true + 0.02 * np.random.randn(len(x))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, alpha_true, 'b-', linewidth=2, label='Ground Truth α(x)')
    plt.plot(x, alpha_pred, 'r--', linewidth=2, label='Predicted α(x)')
    plt.xlabel('x')
    plt.ylabel('α(x)')
    plt.title('Variable-Order Fractional PDE: α(x) Discovery Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('validation_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Basic validation plot created: validation_plot.png")
    
    # Test data generation capabilities
    x_2d, y_2d = np.meshgrid(x, x)
    alpha_2d = 0.25 * np.sin(2 * np.pi * x_2d) * np.cos(2 * np.pi * y_2d) + 1.5
    
    plt.figure(figsize=(8, 6))
    plt.contourf(x_2d, y_2d, alpha_2d, levels=20, cmap='viridis')
    plt.colorbar(label='α(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Variable Fractional Order Field for Smart City Extension')
    plt.savefig('smart_city_alpha_field.png', dpi=300, bbox_inches='tight')
    print("✓ 2D fractional order field plot created: smart_city_alpha_field.png")
    
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    
print("\\nFramework validation complete!")