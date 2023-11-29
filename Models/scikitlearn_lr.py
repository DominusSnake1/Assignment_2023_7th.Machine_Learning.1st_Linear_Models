import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def scikitlearn_lr(sets):
    """
    Train a Linear Regression model using the provided training sets and evaluate it on the test set.

    :param sets: Train and Test Sets, a tuple (X_train, X_test, y_train, y_test)
    :return: The Square Root of the Mean Squared Error (RMSE)
    """

    # Unpack the sets
    X_train, X_test, y_train, y_test = sets

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    MSE = mean_squared_error(y_test, y_pred)

    # Calculate Square Root of Mean Squared Error (RMSE)
    RMSE = np.sqrt(MSE)

    return RMSE
