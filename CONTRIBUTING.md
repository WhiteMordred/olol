# Contributing to OLOL

Thank you for considering contributing to OLOL! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/K2/olol.git
   cd olol
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   uv pip install -e ".[dev,test]"
   ```

## Development Process

1. Create a new branch for your feature
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests to verify your changes
   ```bash
   pytest -v
   ```

4. Format and lint your code
   ```bash
   black .
   isort .
   ruff check .
   mypy src/ tests/
   ```

5. Submit a pull request

## Code Style Guidelines

- Follow PEP 8 and use Black for consistent formatting
- Add type hints to all functions and methods
- Write docstrings in Google style format
- Keep functions and methods small and focused
- Follow the existing patterns in the codebase

## Testing

- Add tests for all new functionality
- Maintain at least 80% code coverage
- Write both unit and integration tests as appropriate

## Documentation

- Update documentation to reflect your changes
- Add examples for new features
- Keep the README up to date

## Protocol Buffer Changes

When changing proto files:

1. Update the proto files in `src/olol/proto/`
2. Regenerate the Python code:
   ```bash
   python -m olol.utils.protoc
   ```

## Code of Conduct

Please be respectful and inclusive when contributing. Harassment or offensive behavior will not be tolerated.