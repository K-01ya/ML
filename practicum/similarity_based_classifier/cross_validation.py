import numpy as np
import nearest_neighbors


def kfold(n, n_folds):
    fold_indices = [0] * n_folds
    len_fold = n // n_folds + 1
    start_index = 0
    finish_index = len_fold
    test_indices = np.arange(finish_index)

    for i in range(n_folds):
        if i == (n % n_folds):
            len_fold -= 1
            finish_index -= 1
            test_indices = np.arange(start_index, finish_index)

        train_indices = np.concatenate((np.arange(start_index), np.arange(finish_index, n)))
        fold_indices[i] = (train_indices, np.copy(test_indices))

        test_indices += len_fold
        start_index += len_fold
        finish_index += len_fold

    return fold_indices


def knn_cross_val_score(X, y, k_list, score, cv=3, **kvargs):
    if type(cv) is int:
        cv = kfold(y.shape[0], cv)
    elif type(cv) is not list:
        cv = kfold(y.shape[0], 3)
    kfold_score = {}
    for k in k_list:
        kfold_score[k] = np.array([])

    for train, test in cv:
        clf = nearest_neighbors.KNNClassifier(k_list[-1], **kvargs)
        clf.fit(X[train], y[train])
        distances_matrix, indices_matrix = clf.find_kneighbors(X[test], True)
        if clf.weights:
            distances_matrix = np.true_divide(1, distances_matrix + 10 ** (-5))

        for k in k_list:
            ans = np.empty(0, y.dtype)
            if clf.weights:
                for i in range(indices_matrix.shape[0]):
                    ans = np.append(ans, np.argmax(np.bincount(y[train][indices_matrix[i][:k]],
                                                               distances_matrix[i][:k])))
            else:
                for indices in indices_matrix:
                    ans = np.append(ans, np.argmax(np.bincount(y[train][indices[:k]])))
            kfold_score[k] = np.append(kfold_score[k], np.sum(ans == y[test]) / y[test].shape[0])

    return kfold_score


if __name__ == '__main__':
    X = np.array([[0, 1, 1], [0, 0, 0], [0, 0, -1], [1, 2, 3], [2, 2, 2], [1, 2, 3], [2, 2, 2], [0, 0, 0], [0, 0, -1]])
    y = np.array([2, 1, 1, 2, 2, 2, 2, 1, 1])
    # print(X, y)
    print(knn_cross_val_score(X, y, [1], 'accuracy', cv=3, strategy='my_own', metric='euclidean', weights=True,
                              test_block_size=4))
