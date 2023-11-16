import numpy as np
import test_lr as lr
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def main():
    RMSE_values = []

    for _ in range(20):
        RMSE = lr.test_lr()
        RMSE_values.append(RMSE)

    mean_RMSE = np.mean(RMSE_values)
    std_RMSE = np.std(RMSE_values)

    print("==========={My Linear Regression Model}===========")
    print(f"Mean RMSE: {mean_RMSE}")
    print(f"Standard Deviation of RMSE: {std_RMSE}\n")


def scikitlearn_LR():
    RMSE_values = []

    for _ in range(20):
        data = lr.fetch_california_housing()
        X = data.data
        y = data.target

        X_train, X_test, y_train, y_test = lr.train_test_split(X, y, test_size=0.3, random_state=42)

        model = LinearRegression()

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model.fit(X_train, y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        y_pred = model.predict(X_test)

        MSE = mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(MSE)

        RMSE_values.append(RMSE)

    mean_RMSE = np.mean(RMSE_values)
    std_RMSE = np.std(RMSE_values)

    print("==========={SciKit Learn Linear Regression Model}===========")
    print(f"Mean RMSE: {mean_RMSE}")
    print(f"Standard Deviation of RMSE: {std_RMSE}\n")


if __name__ == "__main__":
    main()
    scikitlearn_LR()
    print("Το LinearRegression που υλοποιήσα φαίνεται ότι παράγει πολύ μικρά σφάλματα στο Mean RMSE,\nκαι ένα πολύ μικρό αλλά όχι ακριβώς μηδενικό Standard Deviation of RMSE.\nΕνώ το κανονικό LinearRegression από τη βιβλιοθήκη scikit-learn παράγει πολύ παρόμοια αλλά όχι τα ίδια ακριβώς αποτελέσματα.")