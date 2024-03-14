import random
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

    return sum(results)/len(results)


def cross_validation(model, x: np.array, y: np.array) -> float:
    results = cross_val_score(model, x, y)

    return sum(results)/len(results)


def bootstrap(model, x: np.array, y: np.array, use_632: bool = False) -> float:
    accuracy_scores = []
    n_iterations = 100
    test_size = 0.25

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=0
    )

    original_accuracy = accuracy_score(
        y_test,
        model.fit(x_train, y_train).predict(x_test)
    )

    # Bootstrap
    for _ in range(n_iterations):
        indices = np.random.choice(
            range(len(x_train)),
            size=len(x_train),
            replace=True
        )

        x_train_bs = x_train[indices]
        y_train_bs = y_train[indices]

        model.fit(x_train_bs, y_train_bs)
        y_pred = model.predict(x_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    if use_632:
        bootstrap_score = np.mean([
            0.632 * acc + 0.368 * original_accuracy
            for acc in accuracy_scores
        ])
    else:
        bootstrap_score = np.mean(accuracy_scores)

    return bootstrap_score


x, y = generate_data(1, 20, 1000)

lf = LogisticRegression(penalty="l2", C=1000)
ct = DecisionTreeClassifier()


for model in (lf, ct):
    print(model)
    print(refitting(model, x, y))
    print(cross_validation(model, x, y))
    print(bootstrap(model, x, y))
    print(bootstrap(model, x, y, use_632=True))
