
from src.custom_knn import predict_from_neighbors
from src.evaluation import evaluate_regression

def tune_knn(neighbors,y_train, k_values, y_val):
    results = []

    for k_value in k_values:
        prediction = predict_from_neighbors(
            neighbors,
            y_train,
            k_value
        )
        mae, mape = evaluate_regression(
            y_val,
            prediction)
        
        results.append({
            "k":k_value,
            "mae": mae,
            "mape": mape,
        })
    return results