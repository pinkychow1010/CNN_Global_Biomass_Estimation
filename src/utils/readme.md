# Utils Folder

## Overview
The `utils` folder contains various scripts essential for data processing, model training, feature engineering, and evaluation in the CNN-based biomass prediction project. Below is a description of each script and its functionality.

## Folder Structure
```
utils/
├── __init__.py           # Package initialization
├── config.py             # Configuration management for CNN training
├── data.py               # Spatio-temporal matching & image quality check functions
├── datasets.py           # Dataset objects, feature engineering, samplers, early stopping, LR scheduler
├── load_sampler.py       # Test data sampler
├── losses.py             # Multiple loss functions (RMSE, MSE, Gaussian NLL)
├── parser.py             # CLI configuration parser & validation
├── predict.py            # Biomass prediction using trained model
├── processing_utils.py   # Training setup utilities (tracking, logging)
├── read_dataset.py       # Test dataset utilities
├── read_hdf5.py          # HDF5 data extraction
├── sampler.py            # Balanced data sampler for CNN training
├── scheduler.py          # Custom learning rate scheduler
├── test_model.py         # Full workflow for CNN debugging & testing
├── training.py           # Main script for training & validation
├── transformer.py        # Data augmentation utilities
```

## Script Descriptions
- **`config.py`**: Manages training configurations, dependencies, and shortcuts.
- **`data.py`**: Functions for spatio-temporal matching and image quality checks.
- **`datasets.py`**: Dataset setup, geolocation extraction, feature engineering, random sampling, training time tracking, and Monte Carlo dropout.
- **`load_sampler.py`**: Handles test data sampling.
- **`losses.py`**: Implements multiple loss functions (RMSE, MSE, Gaussian Negative Log Likelihood).
- **`parser.py`**: Parses and validates command-line configuration inputs.
- **`predict.py`**: Runs biomass prediction using a trained model checkpoint.
- **`processing_utils.py`**: Utility functions for training setup, including tracking and logging.
- **`read_dataset.py`**: Functions for reading test datasets.
- **`read_hdf5.py`**: Extracts data from HDF5 files.
- **`sampler.py`**: Generates balanced data samplers considering AGB, land use, and spatial locations.
- **`scheduler.py`**: Implements a custom learning rate scheduler inspired by CosineAnnealingWarmUpRestarts.
- **`test_model.py`**: Full workflow for debugging and testing the CNN model.
- **`training.py`**: Main script orchestrating training and validation steps.
- **`transformer.py`**: Defines data augmentation techniques.
