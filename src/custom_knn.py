import numpy as np

def euclidean_distances(X_train, query, k):
    difference = X_train - query
    squared = difference**2
    summed = np.sum(squared, axis = 1)
    distances = np.sqrt(summed)
    nearest_neighbor = np.argpartition(distances, k-1)[:k]
    order = np.argsort(distances[nearest_neighbor])

    nearest_neighbor = nearest_neighbor[order]
    
    return nearest_neighbor




def find_neighbors(X_train, X_query, max_k):
    all_neighbors = np.empty(
        (X_query.shape[0],
        max_k,),
        dtype= np.int32)

    for i, query in enumerate(X_query):
        all_neighbors[i] = euclidean_distances(
            X_train,
            query,
            max_k
        )
    return all_neighbors


def predict_from_neighbors(neighbors, y_train, k):
    predictions = []

    for neighbor_indices in neighbors:
        nearest_k = neighbor_indices[:k]
        neighbor_prices = y_train[nearest_k]

        prediction = np.mean(neighbor_prices)

        predictions.append(prediction)

    return np.array(predictions)    