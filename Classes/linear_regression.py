import numpy as np


class LinearRegression:
    def __init__(self):
        # Initialize weights and bias
        self.w = None  # Weights
        self.b = None  # Bias

    def fit(self, X, y):
        """
        Calculates theta and separates the weights and the bias.

        :param X: Design Matrix
        :param y: Output Vector
        """

        # Check if input data is in the correct format
        if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)):
            raise ValueError("Input data must be numpy arrays")

        # Check if dimensions of input data are compatible
        if X.shape[0] != y.shape[0]:
            raise ValueError("Input data dimensions are not compatible")

        # Add a column of ones to X for the bias term
        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate((X, ones_column), axis=1)

        # Calculate the parameters using the normal equation
        XT = np.transpose(X)
        XT_X = np.dot(XT, X)
        XT_X_inv = np.linalg.inv(XT_X)
        XT_y = np.dot(XT, y)
        theta = np.dot(XT_X_inv, XT_y)

        # Separate weights and bias from the parameters
        self.w = theta[:-1]
        self.b = theta[-1]

    def predict(self, X):
        """
        Makes the predictions.

        :param X: Design Matrix
        :return: Predictions
        """

        # Check if the model has been trained
        if (self.w is None) or (self.b is None):
            raise ValueError("Model has not been trained. Call 'fit' to train the model.")

        # Check if input data is in the correct format
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array")

        # Make predictions using the learned parameters
        X_w = np.dot(X, self.w)
        y_hat = X_w + self.b
        return y_hat

    def evaluate(self, X, y):
        """
        Makes the predictions and calculates the Mean Squared Error.

        :param X: Design Matrix
        :param y: Output Vector
        :return: Predictions, Mean Squared Error
        """
        # Check if the model has been trained
        if (self.w is None) or (self.b is None):
            raise ValueError("Model has not been trained. Call 'fit' to train the model.")

        # Make predictions and calculate Mean Squared Error
        y_hat = self.predict(X)
        MSE = np.mean((y - y_hat) ** 2)
        return y_hat, MSE
