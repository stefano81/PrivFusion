# Contributing to PrivFusion

Thank you for your interest in contributing to PrivFusion! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/data-alignment-poc.git
   cd data-alignment-poc
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ibm-research/data-alignment-poc.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

1. Create and activate a virtual environment:
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Or using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode:
   ```bash
   # Using uv sync (recommended - fastest)
   uv sync --extra dev

   # Or using uv pip
   uv pip install -e .[dev]

   # Or using pip
   pip install -e .[dev]
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Add example notebooks or scripts
- **Performance improvements**: Optimize existing code

### Contribution Workflow

1. **Check existing issues** to see if your contribution is already being worked on
2. **Create an issue** if one doesn't exist for your contribution
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following our coding standards
5. **Write or update tests** for your changes
6. **Run tests** to ensure everything passes
7. **Commit your changes** with clear, descriptive messages
8. **Push to your fork** and create a pull request

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: Maximum 120 characters
- **Imports**: Use absolute imports, group by standard library, third-party, and local
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use the following tools (configured in `setup.cfg` and `.pre-commit-config.yaml`):

- **ruff**: Linting, formatting, and import sorting
- **ty**: Type checking (Astral's fast type checker)

Run formatters before committing:
```bash
ruff check src/ tests/
ruff format src/ tests/
ty check src/
```

### Docstring Example

```python
def consolidate(self, datasets: dict[str, Any], llm: LLM) -> pd.DataFrame:
    """Consolidate multiple datasets into a unified schema.

    Args:
        datasets: Dictionary mapping dataset names to their data and metadata.
            Each entry should contain 'data' (pd.DataFrame) and 'info'
            (DatasetInformation) keys.
        llm: Language model instance to use for semantic analysis.

    Returns:
        DataFrame containing the consolidation results with cluster assignments,
        normalized feature names, and transformation recommendations.

    Raises:
        ValueError: If datasets are malformed or consolidation fails.

    Example:
        >>> consolidator = Consolidator()
        >>> result = consolidator.consolidate(datasets, llm)
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names: `test_<functionality>_<scenario>`
- Use pytest fixtures for common setup
- Mock external dependencies (LLM calls, API requests)

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_mapping.py

# Run specific test
pytest tests/test_mapping.py::test_cluster_features
```

### Test Example

```python
import pytest
from privfusion.consolidater import Consolidator

def test_consolidate_creates_clusters():
    """Test that consolidate correctly identifies feature clusters."""
    # Arrange
    consolidator = Consolidator()
    mock_datasets = create_mock_datasets()
    mock_llm = MockLLM()

    # Act
    result = consolidator.consolidate(mock_datasets, mock_llm)

    # Assert
    assert 'cluster_id' in result.columns
    assert result['cluster_id'].nunique() > 0
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed APIs or added features
2. **Add tests** for new functionality
3. **Run the full test suite** and ensure all tests pass
4. **Update CHANGELOG.md** with your changes
5. **Ensure your code follows** our coding standards

### PR Description Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran and their results

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
```

### Review Process

1. At least one maintainer must review and approve your PR
2. All CI checks must pass
3. Address any feedback from reviewers
4. Once approved, a maintainer will merge your PR

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Clear title** describing the issue
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, package versions)
- **Error messages** or stack traces
- **Minimal code example** that reproduces the issue

### Feature Requests

When requesting features, please include:

- **Clear description** of the feature
- **Use case** explaining why this feature would be useful
- **Proposed implementation** (if you have ideas)
- **Alternatives considered**

### Issue Template

```markdown
## Description
Clear description of the issue or feature request

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- Package version: [e.g., 0.0.1]

## Steps to Reproduce (for bugs)
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Additional Context
Any other relevant information
```

## Development Tips

### Working with LLMs

- **Mock LLM calls** in tests to avoid API costs and ensure reproducibility
- **Use small models** (like Ollama) for local development
- **Cache responses** during development to speed up iteration

### Debugging

- Use the logging module for debug output
- Set `logging.basicConfig(level=logging.DEBUG)` for verbose output
- Use notebooks for interactive debugging

### Performance

- Profile code before optimizing
- Use appropriate data structures (e.g., sets for membership tests)
- Consider memory usage for large datasets

## Questions?

If you have questions about contributing, please:

1. Check existing documentation and issues
2. Ask in the issue tracker
3. Reach out to maintainers

Thank you for contributing to PrivFusion! 🎉
