import os
import numpy as np
import pandas as pd

from src.preprocessing import (
    prepare_base_data,
    split_data,
    input_missing,
    prepare_knn_data,
    prepare_dt_data,
)

from src.custom_knn import (
    find_neighbors,
    predict_from_neighbors,
)

from src.evaluation import (
    evaluate_regression,
)

from src.knn_tunning import (
    tune_knn,
)

from src.sklearn_knn import (
    sklearn_knn,
)

from src.custom_decision_tree import (
    build_tree,
    predict_all,
)

from src.dt_tunning import (
    tune_decision_tree,
)

from src.sklearn_dt import (
    sklearn_decision_tree,
    tune_sklearn_decision_tree,
)

from src.random_forest import (
    random_forest,
    tune_random_forest,
    )

file_path = (
    "data/"
    "apartments_for_rent_classified_100K.csv"
)

cache_path = (
    "outputs/custom_knn/"
    "val_neighbors_k50.npy"
)

from src.final_evaluation import final_evaluation
from src.save_results import save_results

def main():

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    print("Loading data...")

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="latin1",
        low_memory=False,
    )


    # ============================================================
    # 2. CREATE ABT
    # ============================================================

    base_data = prepare_base_data(
        df
    )


    # ============================================================
    # 3. SPLIT DATA
    # ============================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(
        base_data
    )


    # ============================================================
    # 4. MISSING VALUE IMPUTATION
    # ============================================================

    (
        X_train,
        X_val,
        X_test,
    ) = input_missing(
        X_train,
        X_val,
        X_test,
    )


    # ============================================================
    # 5. KNN-SPECIFIC PREPROCESSING
    # ============================================================

    (
        X_train_knn,
        X_val_knn,
        X_test_knn,
    ) = prepare_knn_data(
        X_train,
        X_val,
        X_test,
    )


    # ============================================================
    # 6. CONVERT CUSTOM KNN DATA TO NUMPY
    # ============================================================

    X_train_np = X_train_knn.to_numpy()
    X_val_np = X_val_knn.to_numpy()

    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()


    # ============================================================
    # 7. CUSTOM KNN NEIGHBOR SEARCH / CACHE
    # ============================================================

    max_k = 50

    if os.path.exists(cache_path):

        print("Loading cached neighbors...")

        neighbors = np.load(
            cache_path
        )

    else:

        print("Calculating neighbors...")

        neighbors = find_neighbors(
            X_train_np,
            X_val_np,
            max_k,
        )

        np.save(
            cache_path,
            neighbors,
        )

        print("Neighbors saved.")

    print(
        "Neighbors shape:",
        neighbors.shape,
    )


    # ============================================================
    # 8. CUSTOM KNN SINGLE EXAMPLE
    # ============================================================

    predictions = predict_from_neighbors(
        neighbors,
        y_train_np,
        k=5,
    )

    mae, mape = evaluate_regression(
        y_val_np,
        predictions,
    )

    print("\nCUSTOM KNN")
    print("MAE:", mae)
    print("MAPE:", mape)


    # ============================================================
    # 9. CUSTOM KNN HYPERPARAMETER TUNING
    # ============================================================

    k_values = list(
        range(1, 16)
    )

    custom_knn_results = tune_knn(
        neighbors,
        y_train_np,
        k_values,
        y_val_np,
    )

    save_results(
        custom_knn_results,
        "outputs/tuning/custom_knn_tuning.csv"
    )

    print("\n" + "=" * 70)
    print("CUSTOM KNN RESULTS")
    print("=" * 70)

    for result in custom_knn_results:
        print(result)


    # ============================================================
    # 10. SKLEARN KNN HYPERPARAMETER TUNING
    # ============================================================

    sklearn_knn_results = []

    for k_value in k_values:

        result = sklearn_knn(
            k_value,
            X_train_knn,
            y_train,
            X_val_knn,
            y_val,
        )

        sklearn_knn_results.append(
            result
        )

    save_results(
        sklearn_knn_results,
        "outputs/tuning/sklearn_knn_tuning.csv"
    )

    print("\n" + "=" * 70)
    print("SKLEARN KNN RESULTS")
    print("=" * 70)

    for result in sklearn_knn_results:
        print(result)


    # ============================================================
    # 11. CUSTOM DT SINGLE EXAMPLE
    # ============================================================

    tree = build_tree(
        X_train,
        y_train.to_numpy(),
        max_depth=35,
        min_samples_split=5,
        min_samples_leaf=2,
    )

    predictions = predict_all(
        tree,
        X_val,
    )

    mae, mape = evaluate_regression(
        y_val.to_numpy(),
        predictions,
    )

    print("\nCUSTOM DECISION TREE")
    print("MAE:", mae)
    print("MAPE:", mape)


    # ============================================================
    # 12. CUSTOM DT HYPERPARAMETER TUNING
    # ============================================================

    max_depth_values = [
        25,
        30,
        35,
    ]

    min_samples_split_values = [
        5,
        10,
    ]

    min_samples_leaf_values = [
        2,
        5,
    ]

    custom_dt_results = tune_decision_tree(
        X_train,
        y_train,
        X_val,
        y_val,
        max_depth_values,
        min_samples_split_values,
        min_samples_leaf_values,
    )

    save_results(
        custom_dt_results,
        "outputs/tuning/custom_dt_tuning.csv"
    )

    print("\n" + "=" * 70)
    print("CUSTOM DECISION TREE RESULTS")
    print("=" * 70)

    for result in custom_dt_results:
        print(result)


    # ============================================================
    # 13. SKLEARN DT PREPROCESSING
    # ============================================================

    (
        X_train_dt,
        X_val_dt,
        X_test_dt,
    ) = prepare_dt_data(
        X_train,
        X_val,
        X_test,
    )


    # ============================================================
    # 14. SKLEARN DT HYPERPARAMETER TUNING
    # ============================================================

    sklearn_dt_results = tune_sklearn_decision_tree(
        X_train_dt,
        y_train,
        X_val_dt,
        y_val,
        max_depth_values,
        min_samples_split_values,
        min_samples_leaf_values,
    )

    save_results(
        sklearn_dt_results,
        "outputs/tuning/sklearn_dt_tuning.csv"
    )

    print("\n" + "=" * 70)
    print("SKLEARN DECISION TREE RESULTS")
    print("=" * 70)

    for result in sklearn_dt_results:
        print(result)

    rf_result = random_forest(
    X_train_dt,
    y_train,
    X_val_dt,
    y_val,
    n_estimators=100,
    max_depth=30
)

    print("\nRANDOM FOREST")
    print(rf_result)


    n_estimators_values = [
    50,
    100,
    200,
    300
    ]

    rf_max_depth_values = [
        10,
        20,
        30,
        40,
        None
    ]

    rf_results = tune_random_forest(
        X_train_dt,
        y_train,
        X_val_dt,
        y_val,
        n_estimators_values,
        rf_max_depth_values
    )

    save_results(
        rf_results,
        "outputs/tuning/random_forest_tuning.csv"
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST RESULTS")
    print("=" * 70)

    for result in rf_results:
        print(result)

    test_cache_path = (
    "outputs/custom_knn/"
    "test_neighbors_k2.npy"
    )

    final_results = final_evaluation(
        X_train=X_train,
        X_test=X_test,
        X_train_knn=X_train_knn,
        X_test_knn=X_test_knn,
        X_train_dt=X_train_dt,
        X_test_dt=X_test_dt,
        y_train=y_train,
        y_test=y_test,
        knn_cache_path=test_cache_path,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        final_results.to_string(
            index=False
        )
    )

    os.makedirs(
        "outputs/final",
        exist_ok=True,
    )

    final_results.to_csv(
        "outputs/final/test_results.csv",
        index=False,
    )

    print(
        "\nFinal results saved to "
        "outputs/final/test_results.csv"
    )


if __name__ == "__main__":
    main()
