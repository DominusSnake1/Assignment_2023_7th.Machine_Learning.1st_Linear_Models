import numpy as np
from Classes.linear_regression import LinearRegression


def test_lr(sets):
    X_train, X_test, y_train, y_test = sets

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_hat, MSE = model.evaluate(X_test, y_test)

    RMSE = np.sqrt(MSE)

    return RMSE
