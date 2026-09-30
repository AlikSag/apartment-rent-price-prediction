import time

from src.custom_decision_tree import (
    build_tree,
    predict_all,
)

from src.evaluation import (
    evaluate_regression
)

import time

from src.custom_decision_tree import (
    build_tree,
    predict_all,
)

from src.evaluation import (
    evaluate_regression
)


def tune_decision_tree(
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

                start_time = time.perf_counter()

                tree = build_tree(
                    X_train,
                    y_train.to_numpy(),
                    max_depth=depth,
                    min_samples_split=split,
                    min_samples_leaf=leaf,
                )

                predictions = predict_all(
                    tree,
                    X_val
                )

                end_time = time.perf_counter()
                total_time = end_time - start_time

                mae, mape = evaluate_regression(
                    y_val.to_numpy(),
                    predictions
                )

                results.append({
                    "Max depth": depth,
                    "Min samples split": split,
                    "Min samples leaf": leaf,
                    "MAE": mae,
                    "MAPE": mape,
                    "Time": total_time,
                })

    return results
