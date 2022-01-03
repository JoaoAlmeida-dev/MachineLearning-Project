import random
from typing import List

from src.models.Wine import Wine
from src.models.WineSet import WineSet


# needs to be fixed, random removal is not removing rows of data, it is removing random features from random rows.
def RandomRemoval(dataset: WineSet, removal_percentage: float):
    for _ in range(int(removal_percentage * len(dataset))):
        random_index = random.randrange(0, len(dataset))

    return dataset
