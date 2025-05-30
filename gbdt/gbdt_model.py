from datetime import datetime
import abc
from random import sample
from math import exp, log
from gbdt.tree import construct_decision_tree
import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor



class GBDTMultiClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, task='multiclass'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.classes_ = None
        self.task = task

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)


        if self.task == 'binary' and n_classes > 2:
            import warnings
            warnings.warn("warning-not binary", UserWarning)
            y = (y == self.classes_[0]).astype(int)
            self.classes_ = np.array([self.classes_[0], 'other'])
            n_classes = 2

        F = np.zeros((X.shape[0], n_classes))
        y_one_hot = np.zeros((X.shape[0], n_classes))
        for i, label in enumerate(self.classes_):
            y_one_hot[:, i] = (y == label).astype(int)

        # 
        import pandas as pd
        tree_time_records = []

        for estimator_idx in range(self.n_estimators):
            # print(f"tree {estimator_idx + 1}/{self.n_estimators}...")
            iteration_start_time = time.time()

            trees_for_iteration = []
            avg_residuals = []

            for class_idx in range(n_classes):
                if self.task == 'binary':
                    residual = y - self._sigmoid(F[:, 0])
                elif self.task == 'multiclass':
                    residual = y_one_hot[:, class_idx] - self._softmax(F)[:, class_idx]
                avg_residual = np.mean(np.abs(residual))
                avg_residuals.append(avg_residual)

                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                # 权重
                if sample_weight is not None:
                    weighted_residual = residual * sample_weight
                    tree.fit(X, weighted_residual)
                else:
                    tree.fit(X, residual)
                trees_for_iteration.append(tree)

                if n_classes > 1:
                    F[:, class_idx] += self.learning_rate * tree.predict(X)
                else:
                    F[:, 0] += self.learning_rate * tree.predict(X)

            self.trees.append(trees_for_iteration)

            iteration_elapsed_time = time.time() - iteration_start_time
            # print(f"Tree {estimator_idx + 1},residual: {np.mean(avg_residuals):.6f},time: {iteration_elapsed_time:.2f} seconds.")
            tree_time_records.append({"tree_index": estimator_idx + 1, "train_time": iteration_elapsed_time})

        return pd.DataFrame(tree_time_records)

    def predict(self, X):
        F = np.zeros((X.shape[0], len(self.classes_)))

        for trees_for_iteration in self.trees:
            for class_idx, tree in enumerate(trees_for_iteration):
                F[:, class_idx] += self.learning_rate * tree.predict(X)

        if self.task == 'multiclass':
            probs = self._softmax(F)
            return self.classes_[np.argmax(probs, axis=1)]
        elif self.task == 'binary':
            if len(self.classes_) > 2:
                import warnings
                warnings.warn("not binary", UserWarning)
                probs = self._sigmoid(F[:, 0])
                return np.where(probs > 0.5, self.classes_[0], 'other')
            else:
                probs = self._sigmoid(F[:, 0])
                return self.classes_[(probs > 0.5).astype(int)]
        else:
            raise ValueError(f"Unknown task type: {self.task}")

    def _softmax(self, F):
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / np.sum(exp_F, axis=1, keepdims=True)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
