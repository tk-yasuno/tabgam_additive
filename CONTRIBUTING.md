# Contributing to TabGAM

Thank you for your interest in contributing to TabGAM! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/tabgam_additive.git
   cd tabgam_additive
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
tabgam_additive/
├── src/                    # Core source code
│   ├── models/            # Model implementations
│   ├── agentic_tuner.py   # Agentic ensemble optimizer
│   └── data_preprocessing.py  # Data pipeline
├── outputs/               # Results and figures
├── notebooks/             # Jupyter notebooks
└── tests/                 # Unit tests (TODO)
```

## Contribution Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and under 50 lines when possible

### Docstring Format
```python
def example_function(param1: int, param2: str) -> float:
    """
    Brief description of the function.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
    
    Returns
    -------
    float
        Description of return value
    """
    pass
```

### Adding New Models
When adding a new GAM implementation:

1. Create a new file in `src/models/`
2. Inherit from a common base class (if applicable)
3. Implement `fit()` and `predict()` methods
4. Add comprehensive docstrings
5. Update `README.md` with model description

Example:
```python
class NewGAMModel:
    """
    Brief description of the model.
    
    Mathematical formulation:
    h_i ≈ φ_0 + Σ φ_j(x_ij)
    """
    
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X, y):
        """Train the model."""
        pass
    
    def predict(self, X):
        """Make predictions."""
        pass
```

### Testing
- Add unit tests for new features
- Ensure all tests pass before submitting PR
- Run tests with: `pytest tests/`

### Documentation
- Update `README.md` if adding new features
- Update `METHODOLOGY.md` if changing algorithms
- Add lessons learned to appropriate Lesson*.md files

## Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: Add new feature description"
   ```
   
   Commit message format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `refactor:` Code refactoring
   - `test:` Adding tests

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   - Create PR on GitHub
   - Provide clear description of changes
   - Reference related issues

5. **Code Review**
   - Address reviewer comments
   - Update PR as needed

## Areas for Contribution

### High Priority
- [ ] Add unit tests for all models
- [ ] Improve data pipeline performance
- [ ] Add more optimization strategies to AgenticTuner
- [ ] Implement cross-validation utilities

### Medium Priority
- [ ] Add more GAM variants (e.g., SplineGAM, TreeGAM)
- [ ] Improve SHAP visualization options
- [ ] Add model explainability tools
- [ ] Create tutorial notebooks

### Low Priority
- [ ] Add CLI interface
- [ ] Create web dashboard
- [ ] Add data augmentation methods
- [ ] Optimize memory usage

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Full error traceback
- Minimal reproducible example
- Expected vs. actual behavior

## Questions?

Feel free to open an issue for:
- Feature requests
- Bug reports
- Documentation improvements
- General questions

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help create a welcoming environment

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to TabGAM! 🚀
