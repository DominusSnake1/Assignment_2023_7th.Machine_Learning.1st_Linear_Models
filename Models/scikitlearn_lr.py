import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def scikitlearn_lr(sets):
    X_train, X_test, y_train, y_test = sets

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    return RMSE
