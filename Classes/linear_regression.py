import numpy as np


class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)):
            raise ValueError("Input data must be numpy arrays")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Input data dimensions are not compatible")

        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate((X, ones_column), axis=1)

        XT = np.transpose(X)
        XT_X = np.dot(XT, X)
        XT_X_inv = np.linalg.inv(XT_X)
        XT_y = np.dot(XT, y)
        theta = np.dot(XT_X_inv, XT_y)

        self.w = theta[:-1]
        self.b = theta[-1]

    def predict(self, X):
        if (self.w is None) or (self.b is None):
            raise ValueError("Model has not been trained. Call 'fit' to train the model.")

        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array")

        X_w = np.dot(X, self.w)
        y_hat = X_w + self.b
        return y_hat

    def evaluate(self, X, y):
        if (self.w is None) or (self.b is None):
            raise ValueError("Model has not been trained. Call 'fit' to train the model.")

        y_hat = self.predict(X)
        MSE = np.mean((y - y_hat) ** 2)
        return y_hat, MSE
