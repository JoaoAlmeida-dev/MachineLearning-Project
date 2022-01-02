import random
from typing import List

from src.models.Wine import Wine


def RandomRemoval(dataset: List[Wine], removal_percentage: float):
    for _ in range(int(removal_percentage * len(dataset))):
        random_index = random.randrange(0, len(dataset))
        dataset.pop(random_index)
    return dataset
