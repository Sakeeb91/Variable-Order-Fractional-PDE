# Contributing to Variable-Order Fractional PDE Discovery

We welcome contributions to this research project! This document provides guidelines for contributing to the codebase.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Variable-Order-Fractional-PDE.git
   cd Variable-Order-Fractional-PDE
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines for Python code
- Use type hints where appropriate
- Maximum line length: 88 characters (Black formatter standard)
- Use descriptive variable and function names

### Code Formatting

We use automated code formatting:

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/
```

### Documentation

- Add docstrings to all functions and classes using Google style
- Update README.md if adding new features
- Document any new mathematical formulations
- Add comments for complex algorithms

Example docstring format:
```python
def fractional_derivative(u, alpha, dx):
    """Compute fractional derivative using Grünwald-Letnikov formula.
    
    Args:
        u (torch.Tensor): Input function values
        alpha (float): Fractional order
        dx (float): Spatial step size
        
    Returns:
        torch.Tensor: Fractional derivative approximation
        
    References:
        Pang et al. (2019). fPINNs: Fractional Physics-Informed Neural Networks.
    """
```

### Testing

- Add unit tests for new functions in `tests/`
- Ensure all tests pass before submitting:
  ```bash
  pytest tests/
  ```
- Aim for >80% code coverage for critical components

### Mathematical Accuracy

- Verify numerical implementations against analytical solutions
- Include convergence tests for new algorithms
- Document numerical stability considerations
- Validate against established benchmarks

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version and OS
- Complete error message and stack trace
- Minimal code example to reproduce
- Expected vs actual behavior

### Feature Requests

For new features, please:
- Check existing issues first
- Describe the mathematical/scientific motivation
- Provide implementation suggestions if possible
- Consider backward compatibility

### Code Contributions

#### Neural Network Architectures
- Follow the dual-network pattern (solution + order networks)
- Include proper initialization strategies
- Add activation function constraints for physical bounds

#### Loss Functions
- Implement vectorized computations when possible
- Include numerical stability checks
- Document mathematical derivations

#### Experimental Setups
- Provide reproducible results with fixed random seeds
- Include parameter sensitivity analysis
- Generate comprehensive visualizations

#### Visualization
- Follow the plotting standards in `visuals/plots_description.md`
- Save plots in multiple formats (PNG, PDF, SVG)
- Include interactive versions where appropriate

## Submission Process

1. **Ensure code quality**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   pytest tests/
   ```

2. **Commit with descriptive messages**:
   ```bash
   git commit -m "feat: Add variable-order GL discretization
   
   - Implement point-wise coefficient calculation
   - Add numerical stability improvements
   - Include convergence validation tests"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** with:
   - Clear title and description
   - Reference any related issues
   - Include test results and performance benchmarks
   - Add screenshots for visualization changes

## Review Process

Pull requests will be reviewed for:
- **Mathematical correctness**: Verify algorithms and formulations
- **Code quality**: Style, documentation, and testing
- **Performance**: Computational efficiency and scalability
- **Reproducibility**: Results can be replicated independently

## Research Ethics

- Properly cite all references and prior work
- Acknowledge contributions from collaborators
- Share data and results in accordance with open science principles
- Follow institutional guidelines for research publication

## Getting Help

- **Issues**: Use GitHub Issues for questions and bug reports
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact maintainers for sensitive topics

## Recognition

Contributors will be acknowledged in:
- `AUTHORS.md` file for significant contributions
- Publication acknowledgments for research contributions
- GitHub contributor statistics

Thank you for contributing to advancing fractional PDE discovery methods!