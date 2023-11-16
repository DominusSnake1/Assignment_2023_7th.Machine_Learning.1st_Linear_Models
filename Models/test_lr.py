import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from Classes.linear_regression import LinearRegression


def test_lr():
    df = fetch_california_housing(as_frame=True)

    X = df.data
    y = df.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model.fit(X_train, y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_hat, MSE = model.evaluate(X_test, y_test)

    RMSE = np.sqrt(MSE)

    return RMSE
