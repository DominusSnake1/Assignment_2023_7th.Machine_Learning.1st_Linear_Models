import numpy as np
from Models import test_lr as my_lr
from Models import scikitlearn_lr as sl_lr
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def train_Test_Sets():
    """
    Load the California housing dataset, split it into training and test sets, and return them.

    :return: List containing [X_train, X_test, y_train, y_test]
    """

    data = fetch_california_housing(as_frame=True)

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return [np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)]


def main():
    # Lists to store RMSE values
    test_RMSE_values = []
    sl_RMSE_values = []

    # Generate training and test sets
    sets = train_Test_Sets()

    # Run the models 20 times and store RMSE values
    for _ in range(20):
        test_RMSE = my_lr.test_lr(sets)
        test_RMSE_values.append(test_RMSE)

        sl_RMSE = sl_lr.scikitlearn_lr(sets)
        sl_RMSE_values.append(sl_RMSE)

    # Calculate mean and standard deviation of RMSE values
    test_mean_RMSE = np.mean(test_RMSE_values)
    test_std_RMSE = np.std(test_RMSE_values)

    sl_mean_RMSE = np.mean(sl_RMSE_values)
    sl_std_RMSE = np.std(sl_RMSE_values)

    # Print results
    print("==========={My Linear Regression Model}===========")
    print(f"Mean RMSE: {test_mean_RMSE}")
    print(f"Standard Deviation of RMSE: {test_std_RMSE}\n")

    print("==========={SciKit Learn Linear Regression Model}===========")
    print(f"Mean RMSE: {sl_mean_RMSE}")
    print(f"Standard Deviation of RMSE: {sl_std_RMSE}\n")


if __name__ == "__main__":
    main()
