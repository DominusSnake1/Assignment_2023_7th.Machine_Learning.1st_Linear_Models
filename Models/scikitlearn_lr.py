import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def scikitlearn_lr():
    data = fetch_california_housing()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model.fit(X_train, y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = model.predict(X_test)

    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    return RMSE
