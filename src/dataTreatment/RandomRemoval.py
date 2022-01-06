import random
from typing import List

from src.models.Wine import Wine
from src.models.WineSet import WineSet
from matplotlib import pyplot as plt


# needs to be fixed, random removal is not removing rows of data, it is removing random features from random rows.
##
#
#
# #
def random_removal_mean(dataset: WineSet, removal_percentage: float) -> WineSet:
    dataset_length: int = len(dataset)
    features_to_change: int = int(dataset_length * len(Wine.FEATURES) * removal_percentage)
    dataframe = dataset.wine_dataframe
    changed_row_collumns: List[(int, str)] = []

    means_list: List[float] = []
    for feature in Wine.FEATURES:
        means_list.append(dataframe[feature].mean())

    for _ in range(features_to_change):
        row_index: int = random.randint(0, dataset_length-1)
        collumn_index: int = random.choice(range(0, len(Wine.FEATURES)))
        pos_to_change = (row_index, Wine.FEATURES[collumn_index])

        # check if pos_to_change has been visited
        while pos_to_change in changed_row_collumns:
            row_index: int = random.randint(0, dataset_length-1)
            collumn_index: int = random.choice(range(0, len(Wine.FEATURES)))
            pos_to_change = (row_index, Wine.FEATURES[collumn_index])

        print("initial", dataframe.loc[pos_to_change[0], pos_to_change[1]])
        dataframe.loc[row_index, Wine.FEATURES[collumn_index]] = means_list[collumn_index]
        print("with mean", dataframe.loc[pos_to_change[0], pos_to_change[1]])
        changed_row_collumns.append(pos_to_change)

    dataset.rebuild_from_dataframe()
    return dataset


def random_removal_2(dataset: WineSet, removal_percentage: float) -> WineSet:
    for wine in dataset:
        print(wine)
    # for _ in range(int(removal_percentage * len(dataset))):
    #    random_index = random.randrange(0, len(dataset))

    return dataset
