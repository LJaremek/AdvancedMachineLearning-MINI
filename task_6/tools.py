import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_datasets(m_values):
    datasets = {}
    for m in m_values:
        x1_y0 = np.random.normal(0, 1, 500)
        x2_y0 = np.random.normal(0, 1, 500)
        y0 = np.zeros(500)

        x1_y1 = np.random.normal(m, 1, 500)
        x2_y1 = np.random.normal(m, 1, 500)
        y1 = np.ones(500)

        features = np.vstack((np.column_stack((x1_y0, x2_y0, y0)), 
                              np.column_stack((x1_y1, x2_y1, y1))))
        df = pd.DataFrame(features, columns=["x1", "x2", "y"])

        datasets[f"m={m}"] = df
    
    return datasets


def generate_circular_dataset(noise_level=0.1):
    x1_y1 = np.random.uniform(-1, 1, 500)
    x2_y1 = np.sqrt(1 - x1_y1**2) * np.random.choice([1, -1], 500)
    y1 = np.ones(500)

    x1_y0 = np.random.uniform(-2, 2, 500)
    x2_y0 = np.sqrt(4 - x1_y0**2) * np.random.choice([1, -1], 500)
    y0 = np.zeros(500)

    x1_y1 *= 1 + np.random.normal(0, noise_level, 500)
    x2_y1 *= 1 + np.random.normal(0, noise_level, 500)
    x1_y0 *= 1 + np.random.normal(0, noise_level, 500)
    x2_y0 *= 1 + np.random.normal(0, noise_level, 500)

    features = np.vstack((np.column_stack((x1_y0, x2_y0, y0)), 
                          np.column_stack((x1_y1, x2_y1, y1))))
    df = pd.DataFrame(features, columns=['x1', 'x2', 'y'])

    return df


def plot_datasets(datasets, m_values):
    num_plots = len(m_values)
    _, axes = plt.subplots(
        1, num_plots, figsize=(5 * num_plots, 4), sharey=True
        )

    if num_plots == 1:
        axes = [axes]

    for ax, m in zip(axes, m_values):
        df = datasets[f"m={m}"]

        ax.scatter(
            df[df["y"] == 0]["x1"], df[df["y"] == 0]["x2"],
            alpha=0.5, label="Class 0", color="blue"
            )

        ax.scatter(
            df[df["y"] == 1]["x1"], df[df["y"] == 1]["x2"],
            alpha=0.5, label="Class 1", color="red"
            )

        ax.set_title(f"Dataset with m={m}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_circular_dataset(df):
    plt.figure(figsize=(6, 6))
    plt.scatter(
        df[df["y"] == 0]["x1"], df[df["y"] == 0]["x2"],
        alpha=0.5, label="Class 0", color="blue"
        )

    plt.scatter(
        df[df["y"] == 1]["x1"], df[df["y"] == 1]["x2"],
        alpha=0.5, label="Class 1", color="red"
        )

    plt.title("Circular Dataset with Noise")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.axis("equal")
    plt.show()
