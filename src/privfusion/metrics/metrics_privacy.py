import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

"""
Code source from
    https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py
"""


def scale_dataset(
    dataset: pd.DataFrame,
    data_percent: float = 0.1,
    random_seed: int = 42,
) -> pd.DataFrame:
    dataset = dataset.drop_duplicates(keep=False)
    if data_percent is not None:
        dataset = dataset.sample(
            n=int(len(dataset) * (0.01 * data_percent)),
            random_state=random_seed,
        ).to_numpy()

    scaler = StandardScaler()
    scaler.fit(dataset)
    dataset_scaled = scaler.transform(dataset)
    return dataset_scaled


def compute_smallest_distances(
    dataset: pd.DataFrame,
    other_dataset: pd.DataFrame = None,
) -> np.array:
    dist = metrics.pairwise_distances(
        dataset,
        Y=other_dataset,
        metric="minkowski",
        n_jobs=-1,
    )

    if other_dataset is None:
        # Remove distances of data points to themselves to avoid 0s
        rd_dist = dist[~np.eye(dist.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
    else:
        rd_dist = dist

    # Computing first and second smallest nearest neighbour distances
    smallest_two_indexes = [rd_dist[i].argsort()[:2] for i in range(len(rd_dist))]
    smallest_two_nn_distances = [rd_dist[i][smallest_two_indexes[i]] for i in range(len(rd_dist))]
    return smallest_two_nn_distances


def compute_DCR(
    dataset: pd.DataFrame,
    other_dataset: pd.DataFrame = None,
    data_percent: float = 0.1,
) -> float:
    dataset_scaled = scale_dataset(dataset, data_percent)
    other_dataset_scaled = None
    if other_dataset is not None:
        other_dataset_scaled = scale_dataset(other_dataset, data_percent)

    smallest_distances = compute_smallest_distances(
        dataset_scaled,
        other_dataset_scaled,
    )
    min_dist = np.array([i[0] for i in smallest_distances])
    fifth_perc_dcr = np.percentile(min_dist, 5)
    return fifth_perc_dcr


def compute_NNDR(
    dataset: pd.DataFrame,
    other_dataset: pd.DataFrame = None,
    data_percent: float = 0.1,
) -> float:
    dataset_scaled = scale_dataset(dataset, data_percent)
    other_dataset_scaled = None
    if other_dataset is not None:
        other_dataset_scaled = scale_dataset(other_dataset, data_percent)

    smallest_distances = compute_smallest_distances(
        dataset_scaled,
        other_dataset_scaled,
    )
    nn_ratio = np.array([i[0] / i[1] for i in smallest_distances])
    nn_fifth_perc_nndr = np.percentile(nn_ratio, 5)
    return nn_fifth_perc_nndr
