# Bayesian Classification

## 1. Implementation of LDA, QDA and NB (Naive Bayes) methods for binary classification (classes 0 and 1)
File: [BinaryClassifiers.py](https://github.com/LJaremek/AdvancedMachineLearning-MINI/blob/main/task_1/BinaryClassifiers.py)
```python
class BinaryClassifier(ABC):
    ...


class LDA(BinaryClassifier):
    ...


class QDA(BinaryClassifier):
    ...

class NaiveBayes(BinaryClassifier):
    ...

```

## 2. Comparison of LDA, QDA and NB methods on simulated data

### 2.1. Generate training and testing data
Generating data file - [tools.py](https://github.com/LJaremek/AdvancedMachineLearning-MINI/blob/main/task_1/tools.py)
```python
def generate_y(prob: float, n: int) -> np.array:
    ...


def generate_data_1(
        y: list[int],
        mean: float,
        alpha: float,
        variance: float,
        shuffle: bool = True
        ) -> np.array:
    ...


def generate_data_2(
        y: list[int],
        rho: float,
        alpha: float,
        shuffle: bool = True
        ) -> np.array:
    ...

```

### 2.2. Compare LDA, QDA, and NB for both schemes (compute accuracy on the testing set) for fixed value ρ = 0.5 and different values of a = 0.1, 0.5, 1, 2, 3, 5.

File: [BayesianSimulatedData1.ipynb](https://github.com/LJaremek/AdvancedMachineLearning-MINI/blob/main/task_1/BayesianSimulatedData1.ipynb)


### 2.3. Compare LDA, QDA, and NB for both schemes (compute accuracy on the testing set) for fixed value a = 2 and different values of ρ = 0, 0.1, 0.3, 0.5, 0.7, 0.9.

File: [BayesianSimulatedData2.ipynb](https://github.com/LJaremek/AdvancedMachineLearning-MINI/blob/main/task_1/BayesianSimulatedData2.ipynb)

### 2.4. For one chosen setting of parameters (e.g. a = 2, ρ = 0.5) generate a scatter plot showing observations from training set.

File: [BayesianSimulatedData3.ipynb](https://github.com/LJaremek/AdvancedMachineLearning-MINI/blob/main/task_1/BayesianSimulatedData3.ipynb)

![Comparing models on 2 datasets ](https://github.com/LJaremek/AdvancedMachineLearning-MINI/blob/main/task_1/plots/models_comparation_0_1_datasets.jpg)


## 3. Comparison of LDA, QDA and NB methods on real data.

Chosen datasets:
* [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris) - one class was deleted for binary Y
* [Abalone Dataset](https://archive.ics.uci.edu/dataset/1/abalone) - the goal is to predict whether age is greater or less than average age
* [Diabetes Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download) - gender should be predicted based on medical data

![Comparing models on 3 datasets](https://github.com/LJaremek/AdvancedMachineLearning-MINI/blob/main/task_1/plots/custom_datasets.png)
