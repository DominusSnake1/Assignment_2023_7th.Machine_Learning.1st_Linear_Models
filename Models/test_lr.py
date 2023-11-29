import numpy as np
from Classes.linear_regression import LinearRegression


def test_lr(sets):
    """
    Test a Linear Regression model on the provided test set and calculate the Root Mean Squared Error (RMSE).

    :param sets: A tuple containing the test set (X_test, y_test)
    :return: The Root Mean Squared Error (RMSE) of the model on the test set
    """

    # Unpack the test set
    X_train, X_test, y_train, y_test = sets

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model on the training set (assuming you have a separate training set)
    model.fit(X_train, y_train)

    # Make predictions on the test set and calculate Mean Squared Error (MSE)
    y_hat, MSE = model.evaluate(X_test, y_test)

    # Calculate Root Mean Squared Error (RMSE)
    RMSE = np.sqrt(MSE)

    return RMSE
