import random
from typing import List

from pandas import DataFrame

from src.constants.Constants import FEATURES


def random_removal_mean(dataset: DataFrame, removal_percentage: float):
    dataset_length: int = len(dataset)
    features_to_change: int = int(dataset_length * len(FEATURES) * removal_percentage)
    changed_row_collumns: List[(int, str)] = []

    means_list: List[float] = []
    for feature in FEATURES:
        means_list.append(dataset[feature].mean())

    for _ in range(features_to_change):
        row_index: int = random.randint(0, dataset_length-1)
        collumn_index: int = random.choice(range(0, len(FEATURES)))
        pos_to_change = (row_index, FEATURES[collumn_index])

        # check if pos_to_change has been visited
        while pos_to_change in changed_row_collumns:
            row_index: int = random.randint(0, dataset_length-1)
            collumn_index: int = random.choice(range(0, len(FEATURES)))
            pos_to_change = (row_index, FEATURES[collumn_index])

        dataset.loc[pos_to_change[0], pos_to_change[1]] = means_list[collumn_index]
        changed_row_collumns.append(pos_to_change)
