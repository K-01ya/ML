import numpy as np
from scipy.spatial import distance


def euclidean_distance(X, Y):
    arr = (np.linalg.norm(X[:, np.newaxis, :], axis=-1) ** 2 + np.linalg.norm(Y[np.newaxis, :, :], axis=-1) ** 2 - 2 *
           np.dot(X, np.transpose(Y)))
    arr[arr < 0] = 0
    return arr ** (1/2)


def cosine_distance(X, Y):
    return 1 - (np.dot(X, np.transpose(Y)) /
                ((np.linalg.norm(X[:, np.newaxis, :], axis=-1)) * (np.linalg.norm(Y[np.newaxis, :, :], axis=-1))))


if __name__ == "__main__":
    a = np.array([[1, 0, 0]])
    b = np.array([[1, 0, 0]])
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 4]])
    X = np.array([[0, 1, 1], [0, 0, 0], [0, 0, -1], [1, 2, 3]])
    print(cosine_distance(a, b), '\n', distance.pdist(np.concatenate((a, b)), 'cosine'), '\n')
    print(euclidean_distance(X, X), '\n', distance.pdist(np.concatenate((X, X)), 'euclidean'), '\n')
