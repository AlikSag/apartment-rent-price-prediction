import time

from sklearn.tree import DecisionTreeRegressor

from src.evaluation import evaluate_regression

def sklearn_decision_tree(
    X_train,
    y_train,
    X_val,
    y_val,
    max_depth,
    min_samples_split,
    min_samples_leaf
):
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    prediction = model.predict(X_val)

    end_time = time.perf_counter()

    mae, mape = evaluate_regression(
        y_val,
        prediction
    )

    return {
        "Max depth": max_depth,
        "Min samples split": min_samples_split,
        "Min samples leaf": min_samples_leaf,
        "MAE": mae,
        "MAPE": mape,
        "Time": end_time - start_time
    }

def tune_sklearn_decision_tree(
    X_train,
    y_train,
    X_val,
    y_val,
    max_depth_values,
    min_samples_split_values,
    min_samples_leaf_values
):
    results = []

    for depth in max_depth_values:
        for split in min_samples_split_values:
            for leaf in min_samples_leaf_values:

                result = sklearn_decision_tree(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    max_depth=depth,
                    min_samples_split=split,
                    min_samples_leaf=leaf
                )

                results.append(result)

    return results