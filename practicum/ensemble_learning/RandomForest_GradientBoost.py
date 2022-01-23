import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from numpy.random import default_rng


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []
        self.feature_number = None
        for i in range(self.n_estimators):
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth))

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self.feature_number = X.shape[1]
        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3

        for i in range(self.n_estimators):
            rng = default_rng(i)
            feature_sample_indices = rng.choice(self.feature_number,
                                                size=int(self.feature_subsample_size * self.feature_number),
                                                replace=False)
            self.trees[i].fit(X[:, feature_sample_indices], y)
        if X_val is not None:
            return ((self.predict(X_val) - y_val) ** 2).mean() ** (1/2)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        predictions = []
        for i in range(self.n_estimators):
            rng = default_rng(i)
            feature_sample_indices = rng.choice(self.feature_number, size=self.feature_subsample_size, replace=False)
            predictions.append(self.trees[i].predict(X[:, feature_sample_indices]))

        return np.array(predictions).mean(axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees = []
        self.feature_number = None
        for i in range(self.n_estimators):
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth))

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1/3

        objects_number = X.shape[0]
        previous_predictions = y
        previous_step = 0
        step = 10
        for m in range(self.n_estimators):
            rng = default_rng(m)
            feature_sample_indices = rng.choice(self.feature_number,
                                                size=int(self.feature_subsample_size * self.feature_number),
                                                replace=False)
            self.trees[m].fit(X[:, feature_sample_indices], previous_predictions)
            f_m = self.trees[m].predict(X[feature_sample_indices])
            k = 0
            while abs(previous_step - step) > 0.01:
                previous_step = step
                k += 1
                step -= 2 / (objects_number * (k**(1/2))) * np.dot(previous_predictions - step * f_m - y, -f_m)
            previous_predictions = step * self.learning_rate * previous_predictions
        if X_val is not None:
            return ((self.predict(X_val) - y_val) ** 2).mean() ** (1/2)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predictions = []
        for i in range(self.n_estimators):
            rng = default_rng(i)
            feature_sample_indices = rng.choice(self.feature_number, size=self.feature_subsample_size, replace=False)
            predictions.append(self.trees[i].predict(X[:, feature_sample_indices]))
        return predictions[0] + self.learning_rate * np.array(predictions[1:]).sum(axis=0)
