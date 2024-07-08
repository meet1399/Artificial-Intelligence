# Multi-layer Perceptron for New York Housing Market Data

## Overview
This project involves implementing a multi-layer perceptron (MLP) from scratch to perform a classification task on real-world data from the New York housing market. The objective is to predict the number of bedrooms (BEDS) for given properties using the other 16 features available in the dataset.

## Requirements
- **Language:** Python
- **Libraries:** Only standard libraries such as NumPy (or equivalent for non-Python languages) are allowed. No machine learning libraries like Scikit-learn, TensorFlow, or PyTorch.
- **Runtime Constraint:** The model must be trained and tested within a 5-minute window on Vocareum.

## Dataset
The dataset contains 4801 real estate sales entries with the following features:
- BROKERTITLE
- TYPE
- PRICE
- BATH
- PROPERTYSQFT
- ADDRESS
- STATE
- MAIN_ADDRESS
- ADMINISTRATIVE_AREA_LEVEL_2
- LOCALITY
- SUBLOCALITY
- STREET_NAME
- LONG_NAME
- FORMATTED_ADDRESS
- LATITUDE
- LONGITUDE

The task is to predict the number of bedrooms (BEDS) using the remaining features.

## Grading Criteria
1. **Primary Prediction Task (70%)**: Performance relative to a benchmark model.
2. **Hyperparameter Report (30%)**: Document the exploration and results of different hyperparameters.

## Model Implementation
The neural network should be a feed-forward model with potentially multiple hidden layers. Key hyperparameters include:
- Learning rate
- Mini-batch size
- Number of epochs
- Number of hidden layers and nodes per layer

## Usage
1. **Training the Model:**
   - Read `train_data.csv` and `train_label.csv` for training data.
   - Train the neural network model.
2. **Testing the Model:**
   - Read `test_data.csv` for test data.
   - Output predictions to `output.csv`.
