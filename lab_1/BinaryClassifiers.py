from abc import ABC, abstractmethod

import numpy as np


class BinaryClassifier(ABC):
    def normal_pdf(self, x: float, avg: float, std: float) -> float:
        """
        Probability Density Function (PDF).
        More on Wikipedia: https://w.wiki/9F5W

        Input:
         * x: float - feature value
         * avg: float - mean of the features
         * std: float - varianvce of the features

        Output:
         * y: float - result of PDF
        """
        prefactor = 1 / (std * np.sqrt(2 * np.pi))
        exponent = np.exp(-((x - avg) ** 2) / (2 * std ** 2))
        return prefactor * exponent

    def fit(self, x: list[list], y: list) -> None:
        """
        Fit model using given data.
        Because it is a binary classification, there should be 2 classes.

        Input:
         * x: list[list] - list of records, every record is a group of features
         * y: list - list of labels for the given records (x variable)
        """
        assert len(x) == len(y)
        assert len(np.unique(y)) == 2

        self.classes = np.unique(y)
        self.x = np.array(x)
        self.y = np.array(y)

    @abstractmethod
    def predict_proba(self, x: list, the_class: str) -> float:
        """
        Calculating probability for the given class for the given x record.

        Input:
         * x: list - list of features
         * the_class: str - asked class

        Output:
         * probability: float - probability, that x belongs to the_class
        """
        ...

    def predict(self, x: list) -> str:
        """
        Predicting class based on the given x.

        Input:
         * x: list - list of features

        Output:
         * predicted class: str
        """
        results = {the_class: 0 for the_class in self.classes}

        for the_class in self.classes:
            results[the_class] = self.predict_proba(x, the_class)

        return max(results, key=results.get)

    @abstractmethod
    def get_params(self) -> dict:
        ...


class LDA(BinaryClassifier):
    def _calc_avg(self) -> dict[str, np.array]:
        """
        Calculating mean of parameters for every class in the model data.
        Could be executed only after fit method was executed
            (because we need data).
        """
        return {
            the_class: np.mean(self.x[self.y == the_class], axis=0)
            for the_class in self.classes
        }

    def _calc_std(self) -> dict[str, np.array]:
        """
        Calculating std of parameters for every class in the model data.
        Could be executed only after fit method was executed
            (because we need data).
        """
        return {
            the_class: np.std(self.x[self.y == the_class], axis=0)
            for the_class in self.classes
        }

    def _calc_cov_matrix(self) -> np.array:
        """
        Calculating covariance matrix for the model data.
        Could be executed only after:
         * fit method was executed
         * _calc_avg method was executed
         * _calc_std method was executed
        (because we need data, mean and std of the data).
        """
        n, features = self.x.shape
        cov_matrix = np.zeros((features, features))

        for the_class in self.classes:
            class_data = self.x[self.y == the_class]
            cov_matrix += (
                np.cov(class_data, rowvar=False, ddof=1) *
                (len(class_data) - 1)
            )

        cov_matrix /= n - len(self.classes)
        return cov_matrix

    def fit(self, x: list[list], y: list) -> None:
        super().fit(x, y)

        self.class_avg: dict[str, np.array] = self._calc_avg()
        self.class_std: dict[str, np.array] = self._calc_std()
        self.cov_matrix: np.array = self._calc_cov_matrix()

    def find_border(self) -> tuple[float, float, float]:
        """
        Calculating border line between classes.
        Function calculatin parameters: a, b, c
        Which you can use to calc the line based on x points:
            y(x) = -(c + a*x) / b

        Output:
         * parameters: tuple[float, flaot, float]
        """
        cov_inv = np.linalg.inv(self.cov_matrix)

        a = (self.class_avg[0] - self.class_avg[1]).dot(cov_inv)[0]
        b = (self.class_avg[0] - self.class_avg[1]).dot(cov_inv)[1]
        c = (
            -0.5 *
            (
                self.class_avg[0].dot(cov_inv).dot(self.class_avg[0]) -
                self.class_avg[1].dot(cov_inv).dot(self.class_avg[1])
            )
        )

        return a, b, c

    def predict_proba(self, x: list, the_class: str) -> float:
        class_prob = 1

        for f_i, feature in enumerate(x):
            prob = self.normal_pdf(
                feature,
                self.class_avg[the_class][f_i],
                self.class_std[the_class][f_i],
                )

            class_prob *= prob

        return class_prob

    def get_params(self) -> dict:
        return {
            "class_avg": self.class_avg,
            "class_std": self.class_std
        }


class QDA(BinaryClassifier):
    def _calc_avg(self) -> dict[str, np.array]:
        """
        Calculating mean of parameters for every class in the model data.
        Could be executed only after fit method was executed
            (because we need data).
        """
        return {
            the_class: np.mean(self.x[self.y == the_class], axis=0)
            for the_class in self.classes
        }

    def _calc_cov_matrix(self) -> dict[str, np.array]:
        """
        Calculate a separate covariance matrix for each class.
        """
        cov_matrix = {}

        for the_class in self.classes:
            class_data = self.x[self.y == the_class]
            cov_matrix[the_class] = np.cov(class_data, rowvar=False, ddof=1)

        return cov_matrix

    def fit(self, x: list[list], y: list) -> None:
        super().fit(x, y)

        self.class_avg = self._calc_avg()
        self.cov_matrix = self._calc_cov_matrix()

    def predict_proba(self, x: list, the_class: str) -> float:
        cov_inv = np.linalg.inv(self.cov_matrix[the_class])
        diff = x - self.class_avg[the_class]

        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        prefactor = 1 / np.sqrt(
            (2 * np.pi)**len(x) *
            np.linalg.det(self.cov_matrix[the_class])
            )

        return np.exp(exponent) * prefactor

    def get_params(self) -> dict:
        return {
            "class_avg": self.class_avg,
            "cov_matrix": self.cov_matrix
        }

    def find_border(self) -> tuple[np.array, np.array, np.array]:
        x_min = self.x[:, 0].min() - 1
        x_max = self.x[:, 0].max() + 1

        y_min = self.x[:, 1].min() - 1
        y_max = self.x[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
            )

        z = np.array(
            [
                self.predict([a, b])
                for a, b in zip(np.ravel(xx), np.ravel(yy))
            ]
            ).reshape(xx.shape)

        return xx, yy, z


class NaiveBayes(BinaryClassifier):
    def fit(self, x: list[list], y: list) -> None:
        super().fit(x, y)
        self.feature_stats = {}

        for the_class in self.classes:
            class_data = self.x[self.y == the_class]
            self.feature_stats[the_class] = [
                (np.mean(feature), np.std(feature, ddof=1))
                for feature in zip(*class_data)
                ]

    def predict_proba(self, x: list, the_class: str) -> float:
        log_prob = 0
        for i, feature in enumerate(x):
            avg, std = self.feature_stats[the_class][i]
            log_prob += np.log(self.normal_pdf(feature, avg, std))
        return np.exp(log_prob)

    def get_params(self) -> dict:
        return self.feature_stats

    def find_border(self) -> tuple[np.array, np.array, np.array]:
        x_min = self.x[:, 0].min() - 1
        x_max = self.x[:, 0].max() + 1

        y_min = self.x[:, 1].min() - 1
        y_max = self.x[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
            )

        z = np.array(
            [
                self.predict([a, b])
                for a, b in zip(np.ravel(xx), np.ravel(yy))
            ]
            ).reshape(xx.shape)

        return xx, yy, z
