import pandas as pd

from src.data_analysis import (
    categorical_features,
    numeric_features,
    numeric_quality,
    categorical_quality,
    plot_histogram,
    plot_scatter,
    plot_box,
    plot_categorical
)

from src.preprocessing import (
    prepare_base_data,
    split_data,
    input_missing,
    prepare_knn_data
)


file_path = "data/apartments_for_rent_classified_100K.csv"


def main():

    # ============================================================
    # 1. LOAD RAW DATA
    # ============================================================

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    print("\n" + "=" * 70)
    print("RAW DATA")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nHead:")
    print(df.head())

    print("\nColumns:")
    print(df.columns)

    print("\nInfo:")
    df.info()

    print("\nDuplicated rows:")
    print(df.duplicated().sum())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nMissing values in percent:")
    print(df.isna().mean() * 100)


    # ============================================================
    # 2. NUMERIC DATA QUALITY REPORT
    # ============================================================

    print("\n" + "=" * 70)
    print("NUMERIC QUALITY REPORT")
    print("=" * 70)

    numeric_results = []

    for numeric_feature in numeric_features:
        results = numeric_quality(
            df[numeric_feature]
        )

        numeric_results.append(results)

    numeric_report = pd.DataFrame(
        numeric_results
    )

    print(numeric_report)


    # ============================================================
    # 3. CATEGORICAL DATA QUALITY REPORT
    # ============================================================

    print("\n" + "=" * 70)
    print("CATEGORICAL QUALITY REPORT")
    print("=" * 70)

    categorical_results = []

    for categorical_feature in categorical_features:
        results = categorical_quality(
            df[categorical_feature]
        )

        categorical_results.append(results)

    categorical_report = pd.DataFrame(
        categorical_results
    )

    print(categorical_report)


    # ============================================================
    # 4. NUMERIC VISUALIZATIONS
    # ============================================================

    print("\nCreating numeric plots...")

    for numeric_feature in numeric_features:

        plot_histogram(
            df[numeric_feature]
        )

        if numeric_feature != "price":
            plot_scatter(
                df,
                numeric_feature,
                "price"
            )

        plot_box(
            df[numeric_feature]
        )


    # ============================================================
    # 5. MANUAL OUTLIER / EXTREME VALUE INSPECTION
    # ============================================================

    print("\n" + "=" * 70)
    print("EXTREME VALUE INSPECTION")
    print("=" * 70)

    columns_most_expensive_ones = [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms",
        "cityname",
        "state"
    ]

    columns_most_expensive = [
        "title",
        "body",
        "price_type",
        "price_display",
        "address",
        "source"
    ]

    # Check largest price and square_feet values
    for feature in [
        "price",
        "square_feet"
    ]:

        print(
            f"\nTOP 10 BY {feature}"
        )

        top_10_expensive = df.nlargest(
            10,
            feature
        )

        print(
            top_10_expensive[
                columns_most_expensive_ones
            ]
        )


    # Most expensive listing
    print("\nMOST EXPENSIVE LISTING")

    most_expensive_price = df.nlargest(
        1,
        "price"
    )

    print(
        most_expensive_price[
            columns_most_expensive
        ]
    )

    print("\nBody:")

    print(
        most_expensive_price[
            "body"
        ].iloc[0]
    )


    # Largest square feet listings
    print("\nLARGEST SQUARE FEET LISTINGS")

    biggest_square_feet = df.nlargest(
        10,
        "square_feet"
    )

    print(
        biggest_square_feet[
            columns_most_expensive
        ]
    )

    for i in range(10):

        print(
            f"\nListing {i + 1}:"
        )

        print(
            biggest_square_feet[
                "body"
            ].iloc[i]
        )


    # ============================================================
    # 6. CATEGORICAL VISUALIZATIONS
    # ============================================================

    print("\nCreating categorical plots...")

    for categorical_feature in categorical_features:

        plot_categorical(
            df[categorical_feature]
        )


    # ============================================================
    # 7. CREATE ANALYTICS BASE TABLE
    # ============================================================

    print("\n" + "=" * 70)
    print("ANALYTICS BASE TABLE")
    print("=" * 70)

    base_data = prepare_base_data(df)

    print("\nABT shape:")
    print(base_data.shape)

    print("\nABT columns:")
    print(base_data.columns)

    print("\nABT head:")
    print(base_data.head())

    print("\nMissing values in ABT:")
    print(base_data.isna().sum())


    # ============================================================
    # 8. TRAIN / VALIDATION / TEST SPLIT
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(
        base_data
    )

    print(
        "Train:",
        X_train.shape,
        y_train.shape
    )

    print(
        "Validation:",
        X_val.shape,
        y_val.shape
    )

    print(
        "Test:",
        X_test.shape,
        y_test.shape
    )


    # ============================================================
    # 9. MISSING VALUE IMPUTATION
    # ============================================================

    print("\n" + "=" * 70)
    print("MISSING VALUES BEFORE IMPUTATION")
    print("=" * 70)

    print("\nTrain:")
    print(X_train.isna().sum())

    print("\nValidation:")
    print(X_val.isna().sum())

    print("\nTest:")
    print(X_test.isna().sum())


    X_train, X_val, X_test = input_missing(
        X_train,
        X_val,
        X_test
    )


    print("\n" + "=" * 70)
    print("MISSING VALUES AFTER TRAIN-MEDIAN IMPUTATION")
    print("=" * 70)

    print("\nTrain:")
    print(X_train.isna().sum())

    print("\nValidation:")
    print(X_val.isna().sum())

    print("\nTest:")
    print(X_test.isna().sum())


    # ============================================================
    # 10. KNN-SPECIFIC PREPROCESSING
    # ============================================================

    print("\n" + "=" * 70)
    print("KNN-SPECIFIC PREPROCESSING")
    print("=" * 70)

    (
        X_train_knn,
        X_val_knn,
        X_test_knn
    ) = prepare_knn_data(
        X_train,
        X_val,
        X_test
    )

    print("\nKNN train head:")
    print(
        X_train_knn.head()
    )

    print(
        "\nTrain:",
        X_train_knn.shape
    )

    print(
        "Validation:",
        X_val_knn.shape
    )

    print(
        "Test:",
        X_test_knn.shape
    )


    # ============================================================
    # 11. STANDARDIZATION CHECK
    # ============================================================

    continuous_columns = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude"
    ]

    print("\nMean after StandardScaler:")

    print(
        X_train_knn[
            continuous_columns
        ].mean()
    )

    print("\nStd after StandardScaler:")

    print(
        X_train_knn[
            continuous_columns
        ].std()
    )


    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()