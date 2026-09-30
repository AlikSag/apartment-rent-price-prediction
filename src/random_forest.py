import time

from sklearn.ensemble import RandomForestRegressor

from src.evaluation import evaluate_regression

def random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators,
    max_depth
):
    start_time = time.perf_counter()

    model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42,
    n_jobs=-1
    )

    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    end_time = time.perf_counter()
    mae, mape = evaluate_regression(y_val, prediction)

    return {
        "N_estimators": n_estimators,
        "max_depth": max_depth,
        "MAE": mae,
        "MAPE": mape,
        "time": end_time - start_time,
    }


def tune_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators_values,
    max_depth_values
):
    results = []

    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:

            result = random_forest(
                X_train,
                y_train,
                X_val,
                y_val,
                n_estimators=n_estimators,
                max_depth=max_depth
            )

            results.append(result)

    return results