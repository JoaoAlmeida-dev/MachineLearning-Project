import os

RESOURCES = 'Resources'

PROJECT_ROOT = os.path.split(__file__)[0]+"/../../"
RED_CSV = PROJECT_ROOT + RESOURCES + '/winequality-red.csv'
WHITE_CSV = PROJECT_ROOT + RESOURCES + '/winequality-white.csv'
MIN_MAX = PROJECT_ROOT + RESOURCES + '/winequality-minMax.csv'
CATEGORIES = PROJECT_ROOT + RESOURCES + '/categories.csv'
CORRELATIONS = PROJECT_ROOT + RESOURCES + '/correlations.csv'