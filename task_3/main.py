import random
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
"""
Performance meassures:
    Accuracy -> P(y^ = y)
    Error -> P(y^ != y)
    Recall -> P(y^ = + | y = +) True Positive Rate
    Precision -> P(y = + | y^ = +)
    ROC Curve

Estimation methods:
    Split data (train test)
    Cross Validation
    Bootstrap
"""

SEED = 1337
np.random.seed(SEED)
random.seed(SEED)


def generate_data(b: float, k: int, n: int) -> tuple[np.array, np.array]:
    b_list = np.array([b]*5 + [0]*k)

    x = np.array([
        np.array([np.random.normal(0, 1) for _ in range(5 + k)])
        for _ in range(n)
    ])

    p = np.array([
        1 / (1 + math.exp(-np.matmul(b_list, x_record)))
        for x_record in x
    ])

    y = np.array([
        np.random.binomial(1, p_value)
        for p_value in p
    ])

    return x, y


def refitting(model, x: np.array, y: np.array) -> float:
    model.fit(x, y)

    preds = []
    for x_record in x:
        pred = model.predict(x_record.reshape(1, -1))[0]
        preds.append(pred)

    results = y == np.array(preds)

    return results


def cross_validation(model, x: np.array, y: np.array) -> float:
    cv = StratifiedKFold(n_splits=10)
    results = cross_val_predict(model, x, y, cv=cv, method="predict_proba")

    return results[:, 1]


def bootstrap(
        model,
        x: np.array,
        y: np.array,
        use_632: bool = False
        ) -> np.array:
    from random import randint

    n_iterations = len(y)
    batch_size = 0.3
    size = len(x)
    results = []

    for _ in range(n_iterations):
        model = LogisticRegression(penalty="l2", C=1000)
        x_batch = []
        y_batch = []
        for _ in range(int(size*batch_size)):
            index = randint(0, int(size*batch_size))
            x_batch.append(x[index])
            y_batch.append(y[index])
        model.fit(x_batch, y_batch)
        y_pred = model.predict(x)
        accuracy = np.mean(y_pred == y)
        results.append(accuracy)

    return results


x, y = generate_data(1, 20, 1000)

lf = LogisticRegression(penalty="l2", C=1000)
ct = DecisionTreeClassifier()


def draw_roc(model, x: np.array, y: np.array, method) -> None:
    scores = method(model, x, y)
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    roc_label = f"ROC curve (area = {round(roc_auc, 3)})"
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=roc_label)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("test.png")


draw_roc(lf, x, y, bootstrap)
