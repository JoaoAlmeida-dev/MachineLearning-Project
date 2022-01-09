import os
from typing import List

RESOURCES = 'res'

PROJECT_ROOT = os.path.split(__file__)[0]+"/../../"
RED_CSV = PROJECT_ROOT + RESOURCES + '/winequality-red.csv'
WHITE_CSV = PROJECT_ROOT + RESOURCES + '/winequality-white.csv'
MIN_MAX = PROJECT_ROOT + RESOURCES + '/winequality-minMax.csv'
CORRELATIONS = PROJECT_ROOT + RESOURCES + '/correlations.csv'

HEADERS: List[str] = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                      "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                      "quality", ]
FEATURES: List[str] = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                       "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                       ]
TARGET: str = "quality"

N_VARIABLES: int = len(HEADERS) - 1

