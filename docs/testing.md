# SOTA AI — Testing Guide

## Running Tests

### Unit Tests
```bash
python -m pytest tests/
```

### Specific Tests
```bash
python -m pytest tests/test_attention.py
```

### With Coverage
```bash
python -m pytest --cov=sota tests/
```

---

## Writing Tests

### Test Structure
```python
def test_function():
    # Arrange
    input_data = ...

    # Act
    result = function(input_data)

    # Assert
    assert result == expected
```

### Test Coverage
Aim for >80% coverage on critical paths.

---

## Continuous Integration

Tests run automatically on:
- Pull requests
- Main branch pushes

---

*Good tests ensure code quality.*
