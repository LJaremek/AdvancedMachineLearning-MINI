"""
Boosting algorithms:
1) AdaBoost
2) Gradient Boosting
3) XGBoost

B - total number of iterations
n - liczba rekordów
"""
from math import log

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class AdaBoost:
    def _init_weights(self, how_many: int) -> list[float]:
        return [1/how_many for _ in range(how_many)]

    def compute_tree_error(
            self,
            tree: DecisionTreeClassifier,
            weights: list[float],
            x: np.array,
            y: np.array
            ) -> float:
        """
        Computer weighted classification error for k-th tree (classifier).

        Output:
         * epsilon-k
        """
        n = len(x)

        total_error = 0
        for i in range(n):
            y_pred = tree.predict(x[i].reshape(1, -1))
            condition = int(y_pred != y[i])
            total_error += condition * weights[i]

        return total_error

    def fit(self, x, y, trees_count: int = 10, trees_depth: int = 1) -> None:
        self.b = trees_count
        n = len(x)

        self.weights = self._init_weights(n)

        self.trees = []
        for _ in range(self.b):

            # Fit tree
            tree = DecisionTreeClassifier(max_depth=trees_depth)
            tree.fit(x, y, sample_weight=self.weights)

            # Calculate scaling facotr
            epsilon = self.compute_tree_error(tree, self.weights, x, y)
            scaling_factor = epsilon / (1-epsilon)

            self.trees.append(
                (tree, scaling_factor)
            )

            # Update weights
            for i in range(n):
                if tree.predict(x[i].reshape(1, -1)) == y[i]:
                    self.weights[i] *= scaling_factor

            for i in range(len(self.weights)):
                sum_weights = sum(self.weights)
                self.weights[i] /= sum_weights

    def predict(self, x) -> np.array:
        predictions = []

        for sample in x:
            classes = {}

            for tree, scaling_factor in self.trees:
                y_pred = tree.predict(sample.reshape(1, -1))
                if y_pred[0] in classes:
                    classes[y_pred[0]] += log(1/scaling_factor)
                else:
                    classes[y_pred[0]] = log(1/scaling_factor)

            y_pred = max(classes, key=classes.get)
            predictions.append(y_pred)

        return np.array(predictions)


if __name__ == "__main__":
    np.random.seed(0)

    my_adaboost = AdaBoost()
    adaboost = AdaBoostClassifier()

    results = {
        "MyAdaBoost": [],
        "AdaBoost": []
        }

    for _ in range(50):
        x = np.random.rand(100, 4)
        y = np.where(x[:, 0] + x[:, 1] > 1, 1, 0)

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42
            )

        adaboost.fit(X_train, y_train)
        my_adaboost.fit(X_train, y_train)

        y_pred = adaboost.predict(X_test)
        my_y_pred = my_adaboost.predict(X_test)

        results["AdaBoost"].append(accuracy_score(y_test, y_pred))
        results["MyAdaBoost"].append(accuracy_score(y_test, my_y_pred))

    print("My AdaBoost:", np.mean(results["MyAdaBoost"]))
    print("AdaBoost:", np.mean(results["AdaBoost"]))
