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
        assert len(x) == len(y)

        self.classes = np.unique(y)
        self.x = x
        self.y = y

    @abstractmethod
    def predict_proba(self, x: list, the_class: str) -> float:
        ...

    def predict(self, x: list) -> str:
        results = {the_class: 0 for the_class in self.classes}

        for the_class in self.classes:
            results[the_class] = self.predict_proba(x, the_class)

        return max(results, key=results.get)

    @abstractmethod
    def get_params(self) -> dict:
        ...
