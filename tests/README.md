# Tests Directory

This directory contains automated tests for the high-frequency trading analytics system.

## Test Structure

### Unit Tests (`unit/`)
Individual component tests for:
- `test_models/` - Machine learning and reinforcement learning models
- `test_strategies/` - Trading strategies
- `test_data/` - Data processing modules
- `test_execution/` - Order execution components
- `test_backtesting/` - Backtesting framework
- `test_utils/` - Utility functions

### Integration Tests (`integration/`)
Tests for component interactions:
- `test_strategy_integration.py` - Strategy + model integration
- `test_data_pipeline.py` - End-to-end data processing
- `test_backtesting_integration.py` - Full backtesting workflows

### Performance Tests (`performance/`)
Performance and benchmarking tests:
- `test_latency.py` - Latency optimization tests
- `test_throughput.py` - Throughput benchmarks
- `test_memory_usage.py` - Memory usage optimization

### Strategy Tests (`strategies/`)
Specific strategy validation:
- `test_market_making.py` - Market making strategy tests
- `test_arbitrage.py` - Statistical arbitrage tests
- `test_momentum.py` - Momentum strategy tests

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov pytest-xdist
```

### Run All Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run in parallel
pytest -n auto tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Performance tests only
pytest tests/performance/

# Strategy tests only
pytest tests/strategies/
```

### Run Specific Test Files
```bash
# Test specific models
pytest tests/unit/test_models/test_reinforcement_learning.py

# Test specific strategies
pytest tests/strategies/test_market_making.py
```

## Test Configuration

- `conftest.py` - Shared fixtures and configuration
- `pytest.ini` - Pytest configuration
- Test data fixtures in `fixtures/`

## Testing Guidelines

1. **Unit Tests**: Fast, isolated, test single functions/methods
2. **Integration Tests**: Test component interactions, may use external resources
3. **Performance Tests**: Benchmark critical paths, ensure SLA compliance
4. **Strategy Tests**: Validate trading logic with historical data

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Nightly builds

Target coverage: 90%+ for critical components
