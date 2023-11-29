# Linear Regression Models Comparison

This repository contains Python code for comparing the performance of a custom Linear Regression model and the Linear Regression model provided by scikit-learn. The comparison is based on the Root Mean Squared Error (RMSE) metric.

## Overview

- `LinearRegression` class in the `Classes` directory implements a simple Linear Regression model.
- `test_lr` function in the `Models` directory tests the custom Linear Regression model on a given test set and calculates RMSE.
- `scikitlearn_lr` function in the `Models` directory uses scikit-learn's Linear Regression model to calculate RMSE.
- `train_Test_Sets` function loads the California housing dataset, splits it into training and test sets, and returns them.
- `main` function runs the models 20 times, calculates the mean and standard deviation of RMSE, and prints the results.

## Files

- `Classes/linear_regression.py`: Contains the custom Linear Regression class.
- `Models/test_lr.py`: Implements the testing of the custom Linear Regression model.
- `Models/scikitlearn_lr.py`: Implements the testing of the scikit-learn Linear Regression model.
- `main.py`: The main script that orchestrates the training, testing, and result display.

## Usage

1. Install the required dependencies:

   ```bash
   pip install numpy scikit-learn pandas
   ```

2. Run the main script:

   ```bash
   python main.py
   ```

## Results

The script compares the performance of the custom Linear Regression model and the scikit-learn Linear Regression model based on the California housing dataset. It prints the mean and standard deviation of RMSE for each model after 20 runs.

Example Output:

```
=========== {My Linear Regression Model} ===========
Mean RMSE: 0.1234
Standard Deviation of RMSE: 0.0456

=========== {SciKit Learn Linear Regression Model} ===========
Mean RMSE: 0.0876
Standard Deviation of RMSE: 0.0321
```

The results I get from the `LinearRegression` compared to the `scikit-learn/LinearRegression`
have small differences beyond the eighth decimal number.
