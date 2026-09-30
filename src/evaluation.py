from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error



def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_true,
        y_pred
    ) * 100

    return mae, mape