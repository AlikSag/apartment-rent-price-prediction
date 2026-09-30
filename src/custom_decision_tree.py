import numpy as np
import pandas as pd


class TreeNode:
    def __init__(self):
        self.feature: str | None = None
        self.threshold: float | None = None
        self.split_type: str | None = None

        self.left: TreeNode | None = None
        self.right: TreeNode | None = None

        self.leaf: bool = False
        self.value: float | None = None

        self.children: dict[object, TreeNode] | None = None


def mse(y):
    mean_value = np.mean(y)
    squared_error = (y - mean_value) ** 2
    return np.mean(squared_error)


def weighted_mse(left_y, right_y):
    if len(left_y) == 0 or len(right_y) == 0:
        return np.inf

    total_size = len(left_y) + len(right_y)

    left_weight = len(left_y) / total_size
    right_weight = len(right_y) / total_size

    score = (
        left_weight * mse(left_y)
        + right_weight * mse(right_y)
    )

    return score


def find_best_numeric_split(feature_values, y, min_samples_leaf):
    unique_values = np.unique(feature_values)

    thresholds = (
        unique_values[:-1] + unique_values[1:]
    ) / 2

    best_threshold = None
    best_score = np.inf

    for threshold in thresholds:
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold

        left_y = y[left_mask]
        right_y = y[right_mask]

        if (
            len(left_y) < min_samples_leaf
            or len(right_y) < min_samples_leaf
        ):
            continue

        score = weighted_mse(
            left_y,
            right_y
        )

        if score < best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def categorical_split_score(
    feature_values,
    y,
    min_samples_leaf
):
    unique_values = np.unique(feature_values)

    if len(unique_values) <= 1:
        return None, np.inf

    total_length = len(y)

    groups = {}
    score = 0

    for unique_value in unique_values:
        category_mask = feature_values == unique_value
        category_y = y[category_mask]

        if len(category_y) < min_samples_leaf:
            return None, np.inf

        groups[unique_value] = category_y

        category_weight = len(category_y) / total_length

        score += category_weight * mse(category_y)

    return groups, score


def find_best_split(X, y, min_samples_leaf):
    best_score = np.inf
    best_threshold = None
    best_feature_name = None
    best_groups = None
    best_split_type = None

    for feature_name in X.columns:

        feature_values = X[feature_name].to_numpy()

        if pd.api.types.is_numeric_dtype(X[feature_name]):

            feature_best_threshold, feature_score = (
                find_best_numeric_split(
                    feature_values,
                    y,
                    min_samples_leaf
                )
            )

            groups = None
            split_type = "numerical"

        else:

            groups, feature_score = categorical_split_score(
                feature_values,
                y,
                min_samples_leaf
            )

            feature_best_threshold = None
            split_type = "categorical"

        if feature_score < best_score:
            best_score = feature_score
            best_threshold = feature_best_threshold
            best_feature_name = feature_name
            best_groups = groups
            best_split_type = split_type

    return (
        best_feature_name,
        best_threshold,
        best_score,
        best_groups,
        best_split_type
    )


def split_data(X, y, feature_name, threshold):
    feature_values = X[feature_name].to_numpy()

    left_mask = feature_values <= threshold
    right_mask = feature_values > threshold

    X_left = X[left_mask]
    X_right = X[right_mask]

    y_left = y[left_mask]
    y_right = y[right_mask]

    return X_left, X_right, y_left, y_right


def build_tree(
    X,
    y,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    depth=0
):
    node = TreeNode()

    node.value = np.mean(y)

    if (
        depth >= max_depth
        or len(y) < min_samples_split
    ):
        node.leaf = True
        return node

    (
        feature_name,
        threshold,
        score,
        groups,
        split_type
    ) = find_best_split(
        X,
        y,
        min_samples_leaf
    )

    node.feature = feature_name
    node.threshold = threshold
    node.split_type = split_type

    if node.feature is None:
        node.leaf = True
        return node

    if node.split_type == "numerical":

        if node.threshold is None:
            node.leaf = True
            return node

        (
            X_left,
            X_right,
            y_left,
            y_right
        ) = split_data(
            X,
            y,
            node.feature,
            node.threshold
        )

        node.left = build_tree(
            X_left,
            y_left,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            depth + 1
        )

        node.right = build_tree(
            X_right,
            y_right,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            depth + 1
        )

    else:

        node.children = {}

        if groups is None:
            node.leaf = True
            return node

        for category in groups:
            feature = X[node.feature]

            mask = feature == category

            X_categorical = X[mask]
            y_categorical = y[mask]

            node.children[category] = build_tree(
                X_categorical,
                y_categorical,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                depth + 1
            )

    return node


def predict_one(node, row):
    if node.leaf:
        return node.value

    feature_value = row[node.feature]

    if node.split_type == "numerical":

        if feature_value <= node.threshold:
            return predict_one(
                node.left,
                row
            )

        return predict_one(
            node.right,
            row
        )

    if feature_value not in node.children:
        return node.value

    return predict_one(
        node.children[feature_value],
        row
    )


def predict_all(tree, X):
    predictions = []

    for i in range(X.shape[0]):
        row = X.iloc[i]

        prediction = predict_one(
            tree,
            row
        )

        predictions.append(prediction)

    return predictions


