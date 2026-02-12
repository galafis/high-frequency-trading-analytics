# Tests Directory

This directory contains automated tests for the high-frequency trading analytics system.

## Test Files

- `test_features.py` - Tests for the feature engineering pipeline (`src/data/features.py`)
- `test_validate_data.py` - Tests for the CSV data validation script (`src/validate_data.py`)

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov
```

### Run All Tests
```bash
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Run Specific Test Files
```bash
pytest tests/test_features.py -v
pytest tests/test_validate_data.py -v
```

## Testing Guidelines

1. Each new feature or bug fix should include relevant tests.
2. Use descriptive test names that explain what is being validated.
3. Keep tests isolated — avoid depending on external services or network calls.
