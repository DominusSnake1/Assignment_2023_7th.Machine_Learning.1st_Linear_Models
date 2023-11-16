import numpy as np
from Models import test_lr as my_lr
from Models import scikitlearn_lr as sl_lr


def main():
    RMSE_values = []

    for _ in range(20):
        RMSE = my_lr.test_lr()
        RMSE_values.append(RMSE)

    mean_RMSE = np.mean(RMSE_values)
    std_RMSE = np.std(RMSE_values)

    print("==========={My Linear Regression Model}===========")
    print(f"Mean RMSE: {mean_RMSE}")
    print(f"Standard Deviation of RMSE: {std_RMSE}\n")


def scikitlearn_LR():
    RMSE_values = []

    for _ in range(20):
        RMSE = sl_lr.scikitlearn_lr()
        RMSE_values.append(RMSE)

    mean_RMSE = np.mean(RMSE_values)
    std_RMSE = np.std(RMSE_values)

    print("==========={SciKit Learn Linear Regression Model}===========")
    print(f"Mean RMSE: {mean_RMSE}")
    print(f"Standard Deviation of RMSE: {std_RMSE}\n")


if __name__ == "__main__":
    main()
    scikitlearn_LR()
