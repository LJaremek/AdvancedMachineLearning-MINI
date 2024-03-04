import numpy as np


def train_test_split(
        data: np.array,
        train_size: float
        ) -> tuple[np.array, np.array]:

    split_index = int(len(data)*train_size)
    train, test = data[:split_index], data[split_index:]

    return train, test


def nsd(mean: float, variance: float) -> float:
    """
    normal standard distribution
    """
    return np.random.normal(loc=mean, scale=np.sqrt(variance))


def generate_y(prob: float, n: int) -> np.array:
    return np.array([
            np.random.binomial(1, prob)
            for _ in range(n)
        ]).reshape(-1, 1)


def generate_data_1(
        y: list[int],
        mean: float,
        alpha: float,
        variance: float,
        shuffle: bool = True
        ) -> np.array:

    feature_1 = []
    feature_2 = []

    for y_record in y:
        if y_record == 0:
            feature_1.append(nsd(mean, variance))
            feature_2.append(nsd(mean, variance))
        else:
            feature_1.append(nsd(alpha, variance))
            feature_2.append(nsd(alpha, variance))

    feature_1 = np.array(feature_1).reshape(-1, 1)
    feature_2 = np.array(feature_2).reshape(-1, 1)

    data_1 = np.hstack((y, feature_1, feature_2))
    if shuffle:
        np.random.shuffle(data_1)

    return data_1


def generate_data_2(
        y: list[int],
        rho: float,
        alpha: float,
        shuffle: bool = True
        ) -> np.array:

    cov_matrix_0 = np.array([[1, rho], [rho, 1]])
    cov_matrix_1 = np.array([[1, -rho], [-rho, 1]])

    features = []

    for y_record in y:
        if y_record == 0:
            features.append(
                np.random.multivariate_normal([0, 0], cov_matrix_0),
            )
        else:
            features.append(
                np.random.multivariate_normal([alpha, alpha], cov_matrix_1)
            )

    data_2 = np.hstack((y, features))

    if shuffle:
        np.random.shuffle(data_2)

    return data_2
