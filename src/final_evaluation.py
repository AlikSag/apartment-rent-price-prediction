import os
import time
import numpy as np
import pandas as pd

from src.custom_knn import (
    find_neighbors,
    predict_from_neighbors,
)

from src.custom_decision_tree import (
    build_tree,
    predict_all,
)

from src.sklearn_knn import sklearn_knn
from src.sklearn_dt import sklearn_decision_tree
from src.random_forest import random_forest

from src.evaluation import evaluate_regression


def final_evaluation(
    X_train,
    X_test,
    X_train_knn,
    X_test_knn,
    X_train_dt,
    X_test_dt,
    y_train,
    y_test,
    knn_cache_path,
):

    results = []

    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    X_train_knn_np = X_train_knn.to_numpy()
    X_test_knn_np = X_test_knn.to_numpy()


    # ============================================================
    # 1. CUSTOM KNN
    # ============================================================

    if os.path.exists(knn_cache_path):

        print("Loading cached TEST neighbors...")

        test_neighbors = np.load(
            knn_cache_path
        )

    else:

        print("Calculating TEST neighbors...")

        test_neighbors = find_neighbors(
            X_train_knn_np,
            X_test_knn_np,
            max_k=2,
        )

        os.makedirs(
            os.path.dirname(knn_cache_path),
            exist_ok=True,
        )

        np.save(
            knn_cache_path,
            test_neighbors,
        )

        print("TEST neighbors saved.")

    start_time = time.perf_counter()

    custom_knn_predictions = predict_from_neighbors(
        test_neighbors,
        y_train_np,
        k=2,
    )

    custom_knn_time = time.perf_counter() - start_time

    mae, mape = evaluate_regression(
        y_test_np,
        custom_knn_predictions,
    )

    results.append({
        "Model": "Custom KNN",
        "Parameters": "k=2",
        "MAE": mae,
        "MAPE": mape,
        "Time": custom_knn_time,
    })


    # ============================================================
    # 2. SKLEARN KNN
    # ============================================================

    sklearn_knn_result = sklearn_knn(
        2,
        X_train_knn,
        y_train,
        X_test_knn,
        y_test,
    )

    results.append({
        "Model": "Sklearn KNN",
        "Parameters": "k=2",
        "MAE": sklearn_knn_result["mae"],
        "MAPE": sklearn_knn_result["mape"],
        "Time": np.nan,
    })


    # ============================================================
    # 3. CUSTOM DECISION TREE
    # ============================================================

    print("Training final Custom DT...")

    start_time = time.perf_counter()

    custom_tree = build_tree(
        X_train,
        y_train_np,
        max_depth=35,
        min_samples_split=5,
        min_samples_leaf=2,
    )

    custom_dt_predictions = predict_all(
        custom_tree,
        X_test,
    )

    custom_dt_time = time.perf_counter() - start_time

    mae, mape = evaluate_regression(
        y_test_np,
        custom_dt_predictions,
    )

    results.append({
        "Model": "Custom DT",
        "Parameters": "depth=35, split=5, leaf=2",
        "MAE": mae,
        "MAPE": mape,
        "Time": custom_dt_time,
    })


    # ============================================================
    # 4. SKLEARN DECISION TREE
    # ============================================================

    sklearn_dt_result = sklearn_decision_tree(
        X_train_dt,
        y_train,
        X_test_dt,
        y_test,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
    )

    results.append({
        "Model": "Sklearn DT",
        "Parameters": "depth=30, split=5, leaf=2",
        "MAE": sklearn_dt_result["MAE"],
        "MAPE": sklearn_dt_result["MAPE"],
        "Time": sklearn_dt_result["Time"],
    })


    # ============================================================
    # 5. RANDOM FOREST
    # ============================================================

    rf_result = random_forest(
        X_train_dt,
        y_train,
        X_test_dt,
        y_test,
        n_estimators=300,
        max_depth=None,
    )

    results.append({
        "Model": "Random Forest",
        "Parameters": "n_estimators=300, depth=None",
        "MAE": rf_result["MAE"],
        "MAPE": rf_result["MAPE"],
        "Time": rf_result["time"],
    })


    # ============================================================
    # FINAL TABLE
    # ============================================================

    results_df = pd.DataFrame(
        results
    )

    return results_df