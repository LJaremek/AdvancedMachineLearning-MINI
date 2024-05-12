from tools import generate_datasets, generate_circular_dataset
from tools import plot_datasets, plot_circular_dataset

m_values = [0.5, 1, 2, 3]
datasets = generate_datasets(m_values)
plot_datasets(datasets, m_values)

df_circular = generate_circular_dataset(noise_level=0.1)
plot_circular_dataset(df_circular)
