# Test Suite for Myo Handwriting Recognition

This directory contains comprehensive tests for the Myo Handwriting Recognition project.

## Test Structure

### `test_preprocessing.py`

Tests for data preprocessing functions:

- EMG data preprocessing
- Quaternion data preprocessing
- Feature extraction
- Sequence creation with weighted combinations

### `test_position_only_model.py`

Tests for the position-only (quaternion-only) model:

- Model creation and architecture
- Quaternion preprocessing
- Model training
- Prediction functionality
- Model saving/loading

### `test_gui_components.py`

Tests for GUI components (without requiring full GUI):

- Data validation
- Data saving/loading
- Model loading
- Prediction logic
- Error handling

### `test_integration.py`

Integration tests for the full pipeline:

- Complete EMG + quaternion pipeline
- Position-only pipeline
- Data save/load pipeline
- Model save/load pipeline
- Error handling
- Performance testing

## Running Tests

### Run All Tests

```bash
python run_tests.py
```

### Run Specific Test Module

```bash
python run_tests.py test_preprocessing
python run_tests.py test_position_only_model
python run_tests.py test_gui_components
python run_tests.py test_integration
```

### Run Individual Test Files

```bash
python -m unittest tests.test_preprocessing
python -m unittest tests.test_position_only_model
python -m unittest tests.test_gui_components
python -m unittest tests.test_integration
```

## Test Coverage

The test suite covers:

1. **Data Processing**

   - EMG preprocessing with filtering and normalization
   - Quaternion data handling
   - Feature extraction
   - Data validation

2. **Model Functionality**

   - Model creation and architecture
   - Training pipeline
   - Prediction functionality
   - Model saving/loading

3. **GUI Components**

   - Data validation
   - File I/O operations
   - Error handling
   - Prediction logic

4. **Integration**
   - Full pipeline testing
   - Performance testing
   - Error handling throughout pipeline
   - Data integrity verification

## Adding New Tests

When adding new functionality:

1. **Create unit tests** for individual functions
2. **Add integration tests** for new pipelines
3. **Test error conditions** and edge cases
4. **Verify performance** for new features

### Example Test Structure

```python
def test_new_functionality(self):
    """Test description"""
    # Arrange
    input_data = create_test_data()

    # Act
    result = new_function(input_data)

    # Assert
    self.assertIsNotNone(result)
    self.assertEqual(result.shape, expected_shape)
```

## Test Best Practices

1. **Use descriptive test names** that explain what is being tested
2. **Test both success and failure cases**
3. **Use setUp()** for common test data
4. **Clean up resources** in tearDown() if needed
5. **Test edge cases** (empty data, invalid inputs, etc.)
6. **Verify performance** for time-critical operations

## Continuous Integration

These tests can be integrated into CI/CD pipelines to:

- Catch regressions early
- Ensure code quality
- Verify functionality across environments
- Provide confidence for refactoring

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure project root is in Python path
2. **Missing dependencies**: Install required packages (tensorflow, numpy, etc.)
3. **Test failures**: Check if recent changes broke existing functionality
4. **Performance issues**: Tests may be slow on first run due to model compilation

### Debugging Tests

Add debug prints or use pdb:

```python
import pdb; pdb.set_trace()  # Add breakpoint
print(f"Debug: {variable}")  # Add debug prints
```
