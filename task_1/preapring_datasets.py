import numpy as np


def read_iris_data() -> np.array:
    """
    Source:
    https://archive.ics.uci.edu/dataset/53/iris
    """
    df = np.array(np.genfromtxt(
        "datasets/iris/iris.data",
        delimiter=",",
        dtype=None,
        encoding="utf-8"
        ).tolist()
    )

    labels, counts = np.unique(df[:, 4], return_counts=True)
    label_to_delete = labels[np.argmin(counts)]
    df = df[df[:, 4] != label_to_delete]

    labels = [label for label in labels if label != label_to_delete]

    df[:, 4] = np.where(df[:, 4] == labels[0], 1, 0)
    df = df.astype(float)

    return df


def read_abalone_data() -> np.array:
    """
    Source:
    https://archive.ics.uci.edu/dataset/1/abalone
    """
    df = np.array(np.genfromtxt(
        "datasets/abalone/abalone.data",
        delimiter=",",
        dtype=None,
        encoding="utf-8"
        ).tolist()
    )

    df[:, 0] = np.where(df[:, 0] == "F", 0, 1)
    df = df.astype(float)

    values, counts = np.unique(df[:, -1], return_counts=True)
    average = np.average(values)
    weighted_average = np.sum(values*counts)/np.sum(counts)
    medium_value = (average + weighted_average) / 2
    df[:, -1] = np.where(df[:, -1] > medium_value, 1, 0)

    return df


def read_diabetes_data() -> np.array:
    """
    Source:
    https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download  # noqa
    """
    df = np.array(np.genfromtxt(
        "datasets/archive/diabetes_012_health_indicators_BRFSS2015.csv",
        delimiter=",",
        dtype=None,
        encoding="utf-8"
        ).tolist()
    )

    df = df[1:]  # delete head row

    df[:, [-1, -4]] = df[:, [-4, -1]]  # put y (sex) at the end

    df = df.astype(float)

    return df


if __name__ == "__main__":
    print("Iris:")
    print(read_iris_data()[:5])

    print("Abalone:")
    print(read_abalone_data()[:5])

    print("Diabetes:")
    print(read_diabetes_data()[:5])
