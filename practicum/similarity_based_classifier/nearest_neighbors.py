import numpy as np
from sklearn.neighbors import NearestNeighbors
import distances


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size=1):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.neighbors = None
        self.labeled_data = None

    def fit(self, X, y):
        self.labeled_data = (X, y)
        if self.strategy == 'brute':
            self.neighbors = NearestNeighbors(n_neighbors=self.neighbors, algorithm='brute', metric=self.metric).\
                fit(X, y)
        elif self.strategy != 'my_own':
            self.neighbors = NearestNeighbors(n_neighbors=self.neighbors, algorithm=self.strategy, metric='euclidean').\
                fit(X, y)

    def find_kneighbors(self, X, return_distance):
        if self.strategy == 'my_own':
            return_dist = np.empty((0, self.k), float)
            return_indices = np.empty((0, self.k), int)
            if self.metric == 'cosine':
                distances_matrix = distances.cosine_distance(X, self.labeled_data[0])
            else:
                distances_matrix = distances.euclidean_distance(X, self.labeled_data[0])

            for row in distances_matrix:
                kneighbors_indices = np.argpartition(row, self.k)[:self.k]
                indices_sorted = np.argsort(row[kneighbors_indices])

                return_dist = np.append(return_dist, [row[kneighbors_indices[indices_sorted]]], axis=0)
                return_indices = np.append(return_indices, [kneighbors_indices[indices_sorted]], axis=0)

            if return_distance:
                return return_dist, return_indices
            else:
                return return_indices

        return self.neighbors.kneighbors(X, self.k, return_distance)

    def predict(self, X):
        distances_matrix, indices_matrix = self.find_kneighbors(X, True)
        ans = np.empty(0, self.labeled_data[1].dtype)

        if self.weights:
            distances_matrix = np.true_divide(1, distances_matrix + 10 ** (-5))
            for i in range(indices_matrix.shape[0]):
                ans = np.append(ans, np.argmax(np.bincount(self.labeled_data[1][indices_matrix[i]],
                                                           distances_matrix[i])))
        else:
            for indices in indices_matrix:
                ans = np.append(ans, np.argmax(np.bincount(self.labeled_data[1][indices])))

        return ans


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [2, 2, 2], [1, 2, 3], [2, 2, 2], [0, 0, 0], [0, 0, -1]])
    y = np.array([2, 1, 1, 2, 2, 2, 2, 1, 1])
    a = np.array([[0, 1, 1], [0, 0, 0], [0, 0, -1]])
    neigh = KNNClassifier(1, 'my_own', 'euclidean', True)
    neigh.fit(X, y)
    print(neigh.predict(a))
