import numpy as np
from Models import test_lr as my_lr
from Models import scikitlearn_lr as sl_lr


def main():
    test_RMSE_values = []
    sl_RMSE_values = []

    for _ in range(20):
        test_RMSE = my_lr.test_lr()
        test_RMSE_values.append(test_RMSE)

        sl_RMSE = sl_lr.scikitlearn_lr()
        sl_RMSE_values.append(sl_RMSE)

    test_mean_RMSE = np.mean(test_RMSE_values)
    test_std_RMSE = np.std(test_RMSE_values)

    sl_mean_RMSE = np.mean(sl_RMSE_values)
    sl_std_RMSE = np.std(sl_RMSE_values)

    print("==========={My Linear Regression Model}===========")
    print(f"Mean RMSE: {test_mean_RMSE}")
    print(f"Standard Deviation of RMSE: {test_std_RMSE}\n")

    print("==========={SciKit Learn Linear Regression Model}===========")
    print(f"Mean RMSE: {sl_mean_RMSE}")
    print(f"Standard Deviation of RMSE: {sl_std_RMSE}\n")


if __name__ == "__main__":
    main()
