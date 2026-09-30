from sklearn.neighbors import KNeighborsRegressor
from src.evaluation import evaluate_regression

def sklearn_knn(n_neighbors, X_train_knn, y_train, X_val_knn, y_val):
    
    model = KNeighborsRegressor(n_neighbors=n_neighbors, metric="euclidean", weights= "uniform")

    model.fit(X_train_knn, y_train)
    predictions = model.predict(X_val_knn)
    mae, mape = evaluate_regression(y_val, predictions)
    return {"k value": n_neighbors,
                           "mae": mae,
                           "mape": mape}